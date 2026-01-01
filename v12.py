# =========================================================
# v12.py — Hybrid RAG (Multi-Query RRF + Sparse + Dense(E5) + Rerank)
# - v10 기반 (최고점 구조 유지)
# - Multi-pass query retrieval 추가
# - rerank / weight / candidate 그대로 유지
# =========================================================

import json
import argparse
from typing import List, Dict, Any
from collections import defaultdict

import torch
from elasticsearch import Elasticsearch
from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

# =============================
# Prompts (v3 그대로 유지)
# =============================
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

# =============================
# Utils
# =============================
def safe_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        lines = [l for l in text.splitlines() if "```" not in l]
        text = "\n".join(lines)
    return json.loads(text)


def last_user_msg(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


# =============================
# E5 Embedder
# =============================
class E5Embedder:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def query(self, q: str) -> List[float]:
        enc = self.tokenizer(
            [f"query: {q}"],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0].cpu().tolist()


# =============================
# Reranker
# =============================
class Reranker:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, docs: List[str]) -> List[float]:
        enc = self.tokenizer(
            [query] * len(docs),
            docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        return self.model(**enc).logits.squeeze(-1).cpu().tolist()


# =============================
# Retrieval
# =============================
def sparse_search(es, index, q, k):
    r = es.search(index=index, query={"match": {"content": q}}, size=k)
    return r["hits"]["hits"]


def dense_search(es, index, emb, k, cand):
    r = es.search(
        index=index,
        knn={
            "field": "embeddings",
            "query_vector": emb,
            "k": k,
            "num_candidates": cand,
        },
    )
    return r["hits"]["hits"]


def rrf(hit_lists, weight=1.0, k=60):
    score = defaultdict(float)
    for hits in hit_lists:
        for i, h in enumerate(hits):
            score[h["_source"]["docid"]] += weight / (k + i + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


# =============================
# Core
# =============================
def solve_one(sample, args, client, es, embedder, reranker):
    out = {"standalone_query": "", "topk": []}

    r = client.chat.completions.create(
        model=args.llm_model,
        messages=[{"role": "system", "content": PERSONA_ROUTER}] + sample["msg"],
        temperature=0,
    )
    route = safe_json(r.choices[0].message.content)

    if not route.get("needs_search", True):
        return out

    base_query = route.get("standalone_query") or last_user_msg(sample["msg"])
    out["standalone_query"] = base_query

    # 🔥 Multi-pass queries (minimal & safe)
    queries = [
        base_query,
        base_query + " 정의 원리 과정",
        base_query + " 과학 개념 메커니즘",
    ]

    all_sparse = []
    all_dense = []
    doc_map = {}

    for q in queries:
        s_hits = sparse_search(es, args.index, q, args.sparse_candidates)
        d_hits = dense_search(es, args.index, embedder.query(q), args.dense_candidates, args.num_candidates)

        all_sparse.append(s_hits)
        all_dense.append(d_hits)

        for h in s_hits + d_hits:
            doc_map[h["_source"]["docid"]] = h["_source"]["content"]

    # 🔥 Multi-query RRF
    fused_ids = rrf(all_sparse, args.w_sparse) + rrf(all_dense, args.w_dense)

    # 중복 제거 + 상위 유지
    seen = set()
    fused_ids = [x for x in fused_ids if not (x in seen or seen.add(x))]
    cand_ids = fused_ids[: args.rerank_candidates]
    cand_docs = [doc_map[i] for i in cand_ids if i in doc_map]

    scores = reranker.rerank(base_query, cand_docs)
    ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
    out["topk"] = [d for d, _ in ranked[: args.topk_submit]]

    return out


# =============================
# Main
# =============================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--es_host", required=True)
    p.add_argument("--es_username", required=True)
    p.add_argument("--es_password", required=True)
    p.add_argument("--ca_certs", required=True)

    p.add_argument("--index", default="test")
    p.add_argument("--llm_model", default="gpt-4o-mini")

    p.add_argument("--sparse_candidates", type=int, default=50)
    p.add_argument("--dense_candidates", type=int, default=50)
    p.add_argument("--num_candidates", type=int, default=200)
    p.add_argument("--rerank_candidates", type=int, default=80)
    p.add_argument("--topk_submit", type=int, default=10)

    p.add_argument("--w_sparse", type=float, default=0.55)
    p.add_argument("--w_dense", type=float, default=0.45)

    p.add_argument("--e5_model", default="intfloat/multilingual-e5-base")
    p.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3")

    args = p.parse_args()

    client = OpenAI()
    es = Elasticsearch(
        [args.es_host],
        basic_auth=(args.es_username, args.es_password),
        ca_certs=args.ca_certs,
    )

    embedder = E5Embedder(args.e5_model)
    reranker = Reranker(args.reranker_model)

    with open(args.eval_path, encoding="utf-8") as f, open(args.out_path, "w", encoding="utf-8") as out_f:
        for line in f:
            sample = json.loads(line)
            pred = solve_one(sample, args, client, es, embedder, reranker)
            out_f.write(json.dumps({
                "eval_id": sample["eval_id"],
                "standalone_query": pred["standalone_query"],
                "topk": pred["topk"],
                "answer": "",
                "references": []
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
