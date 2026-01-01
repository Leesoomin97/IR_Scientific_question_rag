# =========================================================
# v11.py — Hybrid RAG (Sparse + Dense(E5) + Rerank)
# - v3 프롬프트 유지
# - standalone_query 필수
# - multilingual E5
# - rerank로 top10 최적화
# - ❌ science gate 제거
# - ✅ Rule-based SEARCH_OFF (deterministic)
# - ✅ Rerank hard-cut (topN keep) then final top10
# - JSONL 출력 (파일 확장자는 .csv)
# =========================================================

import re
import json
import argparse
from typing import List, Dict, Any
from collections import defaultdict

import torch
from elasticsearch import Elasticsearch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


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
        text = "\n".join(lines).strip()
    return json.loads(text)


def last_user_msg(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


# =============================
# Rule-based SEARCH_OFF
# =============================
_EMOJI_RE = re.compile(
    "["  # basic emoji blocks
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

_SMALLTALK_PATTERNS = [
    r"^(hi|hello|hey|안녕|ㅎㅇ|하이)\b",
    r"(고마워|감사|땡큐|thank you)",
    r"(ㅋㅋ+|ㅎㅎ+|ㅠㅠ+|ㅜㅜ+)",
    r"(좋아\?|맞아\?|그치\?|그렇지\?)$",
    r"(뭐해|뭐함|잠|자니)$",
    r"(너는 누구|너 정체|모델 뭐야|너 gpt)",
    r"(번역해줘|요약해줘|첨삭해줘|고쳐줘)\b",
    r"(코드 짜줘|디버깅|에러|설치|pip|conda|환경설정)\b",
]

_SMALLTALK_RE = re.compile("|".join(f"(?:{p})" for p in _SMALLTALK_PATTERNS), flags=re.IGNORECASE)


def is_search_off(question: str, min_len: int) -> bool:
    """
    Deterministic filter to skip retrieval for obvious non-science / chit-chat / meta / very short queries.
    Conservative: only filters when it's highly likely not a document QA query.
    """
    q = (question or "").strip()
    if not q:
        return True

    # too short -> likely not a factual science question
    if len(q) < min_len:
        return True

    # emoji-only / emoji-heavy
    if _EMOJI_RE.sub("", q).strip() == "":
        return True

    # obvious smalltalk / meta / translation / programming help
    if _SMALLTALK_RE.search(q):
        return True

    return False


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
        if not docs:
            return []
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
    return r.get("hits", {}).get("hits", []) or []


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
    return r.get("hits", {}).get("hits", []) or []


def rrf(s_hits, d_hits, ws, wd, k=60):
    score = defaultdict(float)
    for i, h in enumerate(s_hits):
        src = h.get("_source", {})
        docid = src.get("docid", "")
        if docid:
            score[docid] += ws / (k + i + 1)
    for i, h in enumerate(d_hits):
        src = h.get("_source", {})
        docid = src.get("docid", "")
        if docid:
            score[docid] += wd / (k + i + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


# =============================
# Core
# =============================
def solve_one(sample, args, client, es, embedder, reranker):
    out = {"standalone_query": "", "topk": []}

    user_q = last_user_msg(sample.get("msg", []))

    # Rule-based SEARCH_OFF (before router to save tokens & stabilize)
    if args.enable_search_off and is_search_off(user_q, min_len=args.search_off_min_len):
        return out  # empty output

    # Router
    r = client.chat.completions.create(
        model=args.llm_model,
        messages=[{"role": "system", "content": PERSONA_ROUTER}] + sample["msg"],
        temperature=0,
    )
    route = safe_json(r.choices[0].message.content)

    if not route.get("needs_search", True):
        return out

    query = (route.get("standalone_query") or "").strip() or user_q
    out["standalone_query"] = query

    # Retrieval
    s_hits = sparse_search(es, args.index, query, args.sparse_candidates)
    d_hits = dense_search(es, args.index, embedder.query(query), args.dense_candidates, args.num_candidates)

    # doc map
    doc_map = {}
    for h in (s_hits + d_hits):
        src = h.get("_source", {})
        did = src.get("docid", "")
        if did and did not in doc_map:
            doc_map[did] = src.get("content", "") or ""

    fused = rrf(s_hits, d_hits, args.w_sparse, args.w_dense, k=args.rrf_k)

    # Candidate docs for rerank
    cand_ids = fused[: args.rerank_candidates]
    cand_docs = [doc_map[i] for i in cand_ids if i in doc_map]

    # Rerank
    scores = reranker.rerank(query, cand_docs)
    if not scores:
        out["topk"] = []
        return out

    ranked = sorted(zip(cand_ids[: len(scores)], scores), key=lambda x: x[1], reverse=True)

    # ✅ Hard cut: keep only topN after rerank, then take topk_submit
    keep_n = min(args.rerank_keep, len(ranked))
    kept = ranked[:keep_n]
    out["topk"] = [d for d, _ in kept[: args.topk_submit]]

    return out


# =============================
# Main (JSONL OUTPUT)
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

    # Retrieval params (keep your proven defaults)
    p.add_argument("--sparse_candidates", type=int, default=50)
    p.add_argument("--dense_candidates", type=int, default=50)
    p.add_argument("--num_candidates", type=int, default=200)

    # RRF + weights
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--w_sparse", type=float, default=0.55)
    p.add_argument("--w_dense", type=float, default=0.45)

    # Rerank
    p.add_argument("--rerank_candidates", type=int, default=80)
    p.add_argument("--rerank_keep", type=int, default=30, help="Hard-cut after rerank (keep top N before final topk)")
    p.add_argument("--topk_submit", type=int, default=10)

    # Models
    p.add_argument("--e5_model", default="intfloat/multilingual-e5-base")
    p.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3")

    # Rule-based SEARCH_OFF
    p.add_argument("--enable_search_off", action="store_true")
    p.add_argument("--search_off_min_len", type=int, default=6)

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
            line = line.strip()
            if not line:
                continue
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
