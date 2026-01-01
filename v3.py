# =========================================================
# v3.py — Standalone-driven Hybrid RAG (Sparse + Dense, E5 via transformers)
# - Router로 standalone_query 생성
# - Sparse(BM25) + Dense(E5) 동시 검색
# - RRF fusion
# - (옵션) ES 인덱스 생성 + E5 임베딩 색인까지 한 파일에서 처리
# =========================================================

import os
import json
import argparse
import traceback
from typing import List, Dict, Any
from collections import defaultdict

import torch
from elasticsearch import Elasticsearch, helpers
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Prompts
# -----------------------------
PERSONA_ROUTER = """
## Role: 지식 검색 전문가

- 당신의 임무는 답변이 아니라 "검색 성공"이다.
- 당신의 지식이나 추론을 사용하지 마라.
- 지식 질문이면 무조건 문서 검색을 수행한다.

### 판단 기준
- 사실, 개념, 정의, 원인, 설명 → needs_search=true
- 단순 인사/잡담 → needs_search=false
- 애매하면 무조건 needs_search=true

### Standalone Query 규칙
- 검색 엔진 친화적 키워드 나열
- 한국어 중심
- 영어 고유명사 → 한글 + 원어 병기
- 대명사/생략어 → 명시적 치환

### 출력(JSON only)
{
  "needs_search": true/false,
  "standalone_query": "...",
  "brief_reply": ""
}
""".strip()

PERSONA_QA = """
- 반드시 Reference 문서만 사용해 답변하라.
- Reference에 없는 내용은 추측하지 마라.
- 정보가 없으면 명확히 부족하다고 말하라.
- 한국어로 간결하게 작성하라.
""".strip()


# -----------------------------
# E5 embedder (transformers)
# -----------------------------
class E5Embedder:
    """
    E5 embedding via transformers (no sentence-transformers dependency).
    - query:  "query: <text>"
    - passage:"passage: <text>"
    - mean pooling + L2 normalize
    """
    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 256, batch_size: int = 64) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            model_out = self.model(**enc)
            token_embeddings = model_out.last_hidden_state  # (B, T, H)
            attention_mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)

            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts

            # L2 normalize
            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            out.extend(mean_pooled.detach().cpu().tolist())
        return out

    def encode_query(self, query: str) -> List[float]:
        return self.encode([f"query: {query}"], max_length=128, batch_size=1)[0]

    def encode_passage(self, passage: str) -> List[float]:
        return self.encode([f"passage: {passage}"], max_length=256, batch_size=1)[0]


# -----------------------------
# Elasticsearch index (settings/mappings)
# -----------------------------
def create_index_with_dense(es: Elasticsearch, index: str, dims: int, overwrite: bool):
    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter"]
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                }
            }
        }
    }

    mappings = {
        "properties": {
            "docid": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "nori"},
            "src": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
            # baseline 호환: 필드명은 embeddings 로 고정
            "embeddings": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine"
            }
        }
    }

    if es.indices.exists(index=index):
        if overwrite:
            es.indices.delete(index=index)
        else:
            return

    es.indices.create(index=index, settings=settings, mappings=mappings)


def bulk_index_documents(
    es: Elasticsearch,
    index: str,
    docs_path: str,
    embedder: E5Embedder,
    batch_size: int = 256,
):
    # load docs
    docs = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    # embed in batches
    contents = [d.get("content", "") for d in docs]
    embs = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        batch_texts = [f"passage: {t}" for t in batch]
        embs.extend(embedder.encode(batch_texts, max_length=256, batch_size=64))

    # bulk actions
    actions = []
    for d, e in zip(docs, embs):
        src = {
            "docid": d.get("docid", ""),
            "content": d.get("content", ""),
            "src": d.get("src", ""),
            "embeddings": e,
        }
        actions.append({"_index": index, "_source": src})

    helpers.bulk(es, actions)


# -----------------------------
# Retrieval
# -----------------------------
def sparse_retrieve(es: Elasticsearch, index: str, query: str, size: int) -> List[Dict[str, Any]]:
    resp = es.search(
        index=index,
        query={"match": {"content": {"query": query}}},
        size=size,
        sort="_score",
    )
    return resp.get("hits", {}).get("hits", []) or []


def dense_retrieve(
    es: Elasticsearch,
    index: str,
    query_emb: List[float],
    size: int,
    num_candidates: int,
) -> List[Dict[str, Any]]:
    resp = es.search(
        index=index,
        knn={
            "field": "embeddings",  # 반드시 embeddings
            "query_vector": query_emb,
            "k": size,
            "num_candidates": num_candidates,
        },
    )
    return resp.get("hits", {}).get("hits", []) or []


def rrf_fusion(
    sparse_hits: List[Dict[str, Any]],
    dense_hits: List[Dict[str, Any]],
    w_sparse: float,
    w_dense: float,
    k: int,
) -> List[str]:
    scores = defaultdict(float)

    for rank, h in enumerate(sparse_hits):
        docid = h.get("_source", {}).get("docid", "")
        if docid:
            scores[docid] += w_sparse / (k + rank + 1)

    for rank, h in enumerate(dense_hits):
        docid = h.get("_source", {}).get("docid", "")
        if docid:
            scores[docid] += w_dense / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docid for docid, _ in fused]


# -----------------------------
# Router JSON parse helper
# -----------------------------
def safe_parse_router_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    # 모델이 가끔 ```json ... ``` 로 감싸면 제거
    if text.startswith("```"):
        text = text.strip("`").strip()
        # 남아있는 json만 남기기
        if "\n" in text:
            lines = [ln for ln in text.splitlines() if ln.strip() and "```" not in ln]
            text = "\n".join(lines).strip()
    return json.loads(text)


# -----------------------------
# Core
# -----------------------------
def answer_question(
    client: OpenAI,
    es: Elasticsearch,
    embedder: E5Embedder,
    index: str,
    messages: List[Dict[str, str]],
    llm_model: str,
    sparse_candidates: int,
    dense_candidates: int,
    num_candidates: int,
    rrf_k: int,
    w_sparse: float,
    w_dense: float,
    topk_submit: int,
) -> Dict[str, Any]:

    out = {"standalone_query": "", "topk": [], "answer": "", "references": []}

    # 1) router
    try:
        router = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": PERSONA_ROUTER}] + (messages or []),
            temperature=0,
        )
        route = safe_parse_router_json(router.choices[0].message.content)
    except Exception:
        traceback.print_exc()
        # fallback: 무조건 검색 + 마지막 user를 query로
        route = {"needs_search": True, "standalone_query": "", "brief_reply": ""}

    if not route.get("needs_search", True):
        out["answer"] = route.get("brief_reply", "")
        return out

    query = (route.get("standalone_query") or "").strip()
    if not query:
        for m in reversed(messages or []):
            if m.get("role") == "user":
                query = (m.get("content") or "").strip()
                break

    out["standalone_query"] = query

    # 2) retrieve
    sparse_hits = sparse_retrieve(es, index, query, sparse_candidates)

    query_emb = embedder.encode_query(query)
    dense_hits = dense_retrieve(es, index, query_emb, dense_candidates, num_candidates)

    fused_docids = rrf_fusion(sparse_hits, dense_hits, w_sparse, w_dense, rrf_k)
    top_docids = fused_docids[:topk_submit]
    out["topk"] = top_docids

    # 3) build references/contexts
    doc_map = {}
    for h in sparse_hits + dense_hits:
        src = h.get("_source", {})
        did = src.get("docid", "")
        if did and did not in doc_map:
            doc_map[did] = src.get("content", "")

    contexts = [doc_map[d] for d in top_docids if d in doc_map]
    out["references"] = [{"docid": d, "content": doc_map[d]} for d in top_docids if d in doc_map]

    # 4) answer
    try:
        qa = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": PERSONA_QA},
                * (messages or []),
                {"role": "assistant", "content": json.dumps(contexts, ensure_ascii=False)},
            ],
            temperature=0,
        )
        out["answer"] = qa.choices[0].message.content or ""
    except Exception:
        traceback.print_exc()

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--out_path", required=True)

    parser.add_argument("--es_host", required=True)
    parser.add_argument("--es_username", required=True)
    parser.add_argument("--es_password", required=True)
    parser.add_argument("--ca_certs", required=True)

    parser.add_argument("--index", default="test")  # 기본은 test로
    parser.add_argument("--llm_model", default="gpt-4o-mini")

    parser.add_argument("--sparse_candidates", type=int, default=30)
    parser.add_argument("--dense_candidates", type=int, default=30)
    parser.add_argument("--num_candidates", type=int, default=100)

    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--w_sparse", type=float, default=0.7)
    parser.add_argument("--w_dense", type=float, default=0.3)
    parser.add_argument("--topk_submit", type=int, default=10)

    # indexing options (dense 필드가 없으면 이걸로 새로 만들고 넣어야 함)
    parser.add_argument("--docs_path", default="../data/documents.jsonl")
    parser.add_argument("--reindex", action="store_true", help="Create index with dense_vector and re-ingest documents")
    parser.add_argument("--overwrite_index", action="store_true", help="Delete existing index if exists")
    parser.add_argument("--embed_batch", type=int, default=256)

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()

    es = Elasticsearch(
        [args.es_host],
        basic_auth=(args.es_username, args.es_password),
        ca_certs=args.ca_certs,
    )

    # E5 base dims = 768
    embedder = E5Embedder("intfloat/e5-base-v2")

    # If embeddings field missing, you MUST reindex to use dense.
    if args.reindex:
        create_index_with_dense(es, args.index, dims=768, overwrite=args.overwrite_index)
        bulk_index_documents(es, args.index, args.docs_path, embedder, batch_size=args.embed_batch)

    with open(args.eval_path, "r", encoding="utf-8") as f, open(args.out_path, "w", encoding="utf-8") as out_f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            pred = answer_question(
                client=client,
                es=es,
                embedder=embedder,
                index=args.index,
                messages=sample.get("msg", []),
                llm_model=args.llm_model,
                sparse_candidates=args.sparse_candidates,
                dense_candidates=args.dense_candidates,
                num_candidates=args.num_candidates,
                rrf_k=args.rrf_k,
                w_sparse=args.w_sparse,
                w_dense=args.w_dense,
                topk_submit=args.topk_submit,
            )

            out_f.write(json.dumps({
                "eval_id": sample.get("eval_id"),
                "standalone_query": pred["standalone_query"],
                "topk": pred["topk"],
                "answer": pred["answer"],
                "references": pred["references"],
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
