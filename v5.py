# =========================================================
# v5.py — Hybrid RAG (Sparse + Dense(E5) + Rerank)
# - v3 Router 프롬프트 유지
# - standalone_query 필수
# - multilingual E5
# - rerank로 top10 최적화
# - 비과학 질문은 무응답 (science gate)
# - JSONL 출력 (파일 확장자는 .csv)
# - FIX: rerank 후보 id-doc 정렬 불일치 버그 수정
# =========================================================

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
# Science Gate Prompt (v5 핵심 변경)
# =============================
SCIENCE_GATE_PROMPT = r"""
너는 'science gate' 분류기다.
입력 질문이 우리 코퍼스(자연과학 상식 문서)로 답할 수 있는 "과학 상식 질문"인지 판정하라.

[과학(true)으로 판단할 범위]
- 자연과학: 물리, 화학, 생물, 지구과학(천문/기후/지질/대기/해양 포함)
- 기초 의학/약학/보건(원리·원인·메커니즘 중심):
  - 질병/증상 원인, 인체 생리 작용, 약물 상호작용/부작용의 원리, 예방/면역의 과학적 설명
- 생명체/동물/식물/생태/환경:
  - 행동(사회화 포함), 학습, 생태 반응, 인간 활동(도시화 등)로 인한 자연계/생태 변화의 과학적 설명
- 농업/재배/사육/병해충 등 "생물학/환경 요인"에 근거한 원리 설명
- '무엇 때문에/어떻게 작동하는지/메커니즘/원인/정의/개념/분류'를 묻는 질문

[과학(false)으로 판단할 범위]
- 정책/제도/행정/예산/공교육/국가별 통계 비교 자체(사회·경제·정치·역사적 평가 포함)
- 연구방법론/조사설계/통계분석 방법 자체(코호트 효과를 고려한 디자인 방식 등)
- 컴퓨터과학/프로그래밍/알고리즘/암호학/정보보안/시스템 설계 및 취약점(예: 머클-담고르, MAC 취약점)
- 단순 사용법/요령/추천/구매 조언/개인 경험담/감정/잡담/번역 요청
  - 단, 생명체/인체/자연현상의 '원리 설명'을 요구하면 true 쪽으로 판정

[판정 원칙]
- 애매하면 true로 둔다. (오탐보다 미탐이 더 치명적)
- 설명 대상이 '자연계/인체/생명체/물질 변화'이면 true 쪽.
- 설명 대상이 '인공 시스템(소프트웨어/알고리즘/보안 구조) 또는 방법론(연구 설계)'이면 false 쪽.

출력(JSON only):
{
  "is_science": true/false
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
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m.get("content", "") or ""
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
def sparse_search(es: Elasticsearch, index: str, q: str, k: int):
    r = es.search(index=index, query={"match": {"content": q}}, size=k)
    return r.get("hits", {}).get("hits", []) or []


def dense_search(es: Elasticsearch, index: str, emb: List[float], k: int, cand: int):
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


def rrf(s_hits, d_hits, ws: float, wd: float, k: int = 60):
    score = defaultdict(float)
    for i, h in enumerate(s_hits):
        docid = (h.get("_source") or {}).get("docid", "")
        if docid:
            score[docid] += ws / (k + i + 1)
    for i, h in enumerate(d_hits):
        docid = (h.get("_source") or {}).get("docid", "")
        if docid:
            score[docid] += wd / (k + i + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


# =============================
# Science Gate
# =============================
def science_gate(client: OpenAI, model: str, question: str) -> bool:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SCIENCE_GATE_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        return safe_json(r.choices[0].message.content).get("is_science", True)
    except Exception:
        # gate 실패 시에는 미탐을 피하기 위해 true
        return True


# =============================
# Core
# =============================
def solve_one(sample: Dict[str, Any], args, client: OpenAI, es: Elasticsearch, embedder: E5Embedder, reranker: Reranker):
    out = {"standalone_query": "", "topk": []}

    msgs = sample.get("msg", []) or []

    # Router
    try:
        r = client.chat.completions.create(
            model=args.llm_model,
            messages=[{"role": "system", "content": PERSONA_ROUTER}] + msgs,
            temperature=0,
        )
        route = safe_json(r.choices[0].message.content)
    except Exception:
        route = {"needs_search": True, "standalone_query": ""}

    if not route.get("needs_search", True):
        return out

    query = (route.get("standalone_query") or "").strip()
    if not query:
        query = last_user_msg(msgs).strip()

    out["standalone_query"] = query

    if not query:
        return out

    # Retrieval
    s_hits = sparse_search(es, args.index, query, args.sparse_candidates)
    d_hits = dense_search(es, args.index, embedder.query(query), args.dense_candidates, args.num_candidates)

    doc_map = {}
    for h in (s_hits + d_hits):
        src = h.get("_source") or {}
        did = src.get("docid", "")
        if did and did not in doc_map:
            doc_map[did] = src.get("content", "") or ""

    fused = rrf(s_hits, d_hits, args.w_sparse, args.w_dense)
    cand_ids = fused[: args.rerank_candidates]

    # FIX: ids와 docs를 같은 필터 기준으로 정렬/정합 보장
    cand_pairs = [(did, doc_map[did]) for did in cand_ids if did in doc_map and doc_map[did]]
    if not cand_pairs:
        return out

    filtered_ids = [did for did, _ in cand_pairs]
    cand_docs = [doc for _, doc in cand_pairs]

    # Science gate (원문 질문 기준)
    if args.enable_science_gate:
        q_raw = last_user_msg(msgs).strip()
        if not science_gate(client, args.llm_model, q_raw):
            return out  # 무응답

    # Rerank
    scores = reranker.rerank(query, cand_docs)
    if not scores:
        return out

    ranked = sorted(zip(filtered_ids, scores), key=lambda x: x[1], reverse=True)
    out["topk"] = [d for d, _ in ranked[: args.topk_submit]]

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

    p.add_argument("--sparse_candidates", type=int, default=50)
    p.add_argument("--dense_candidates", type=int, default=50)
    p.add_argument("--num_candidates", type=int, default=200)
    p.add_argument("--rerank_candidates", type=int, default=80)
    p.add_argument("--topk_submit", type=int, default=10)

    p.add_argument("--w_sparse", type=float, default=0.7)
    p.add_argument("--w_dense", type=float, default=0.3)

    p.add_argument("--e5_model", default="intfloat/multilingual-e5-base")
    p.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--enable_science_gate", action="store_true")

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
                "eval_id": sample.get("eval_id"),
                "standalone_query": pred["standalone_query"],
                "topk": pred["topk"],
                "answer": "",
                "references": []
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
