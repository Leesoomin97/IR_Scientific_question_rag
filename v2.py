# =========================================================
# v2.6_full.py — Scientific RAG (LLM post-filter, jsonl-in-csv)
#
# 핵심:
# - 출력 포맷: "jsonl" (한 줄에 JSON 1개) / 단 파일 확장자는 .csv로 저장 가능
# - Gate 제거: 모든 샘플에서 검색 수행
# - LLM Standalone Query 생성: 멀티턴에서 마지막 발화가 애매한 문제 해결
# - Multilingual: KO query + EN translated query를 함께 RRF로 fusion
# - LLM 후단 판별: 비과학이면 topk=[]로 강제 (대회 MAP 로직 최적)
# - Debug 옵션 제공: --debug_out_path 로 원인 추적 로그 저장
# =========================================================

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from collections import defaultdict

from elasticsearch import Elasticsearch
from openai import OpenAI


# -----------------------------
# Prompts
# -----------------------------
STANDALONE_QUERY_PROMPT = """
너는 RAG 시스템의 쿼리 생성기다.
입력은 대화 메시지들이다. 마지막 사용자 메시지가 애매할 수 있으니, 전체 대화를 참고해
검색에 유리한 'standalone query'를 한국어로 간결하게 한 줄로 만들어라.

규칙:
- 가능한 한 짧고 검색 친화적으로
- 질문 의도를 명확히 (대명사/지시어/생략 보완)
- 불필요한 예의/군더더기 제거
- 출력은 텍스트 한 줄만 (따옴표/설명 금지)
""".strip()

LLM_POST_FILTER_PROMPT = """
다음 사용자 입력(대화 포함)이 과학 상식/자연과학/공학/의학/생물/물리/화학/지구과학 등
과학적 사실이나 원리 설명을 요구하면 YES,
인사, 감정표현, 잡담, 의견, 사회/정치/경제 일반대화면 NO.

매우 중요:
- 애매하면 YES로 판단하라. (FN을 최소화)
- 출력은 YES 또는 NO만.
""".strip()

PROMPT_TRANSLATE_TO_EN = """
Translate the given Korean query into a short, search-friendly English query.
Return ONLY the English query.
""".strip()


# -----------------------------
# Elasticsearch retrieval
# -----------------------------
def sparse_retrieve(es: Elasticsearch, index: str, query: str, size: int) -> List[Dict[str, Any]]:
    resp = es.search(
        index=index,
        query={"match": {"content": {"query": query}}},
        size=size,
    )
    return resp.get("hits", {}).get("hits", []) or []


# -----------------------------
# RRF
# -----------------------------
def reciprocal_rank_fusion(ranklists: Dict[str, List[str]], k: int = 60) -> List[str]:
    scores = defaultdict(float)
    for _, docids in ranklists.items():
        for rank, docid in enumerate(docids):
            scores[docid] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docid for docid, _ in fused]


# -----------------------------
# LLM helpers
# -----------------------------
def llm_make_standalone_query(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": STANDALONE_QUERY_PROMPT}] + (messages or []),
            temperature=0,
            timeout=20,
        )
        q = (res.choices[0].message.content or "").strip()
        q = q.replace("\n", " ").replace("\r", " ").strip()
        return q[:300]
    except Exception:
        return ""


def translate_to_en(client: OpenAI, model: str, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_TRANSLATE_TO_EN},
                {"role": "user", "content": text},
            ],
            temperature=0,
            timeout=20,
        )
        en = (res.choices[0].message.content or "").strip()
        en = en.replace("\n", " ").replace("\r", " ").strip()
        return en[:300]
    except Exception:
        return ""


def llm_is_science_postfilter(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> bool:
    """
    대화 전체를 입력으로 주고, 과학질문인지 후단 판별.
    애매하면 YES로 유도하는 프롬프트 적용.
    """
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_POST_FILTER_PROMPT},
                * (messages or []),
            ],
            temperature=0,
            timeout=20,
        )
        out = (res.choices[0].message.content or "").strip().upper()
        return out == "YES"
    except Exception:
        # 판별 실패 시 FN을 줄이기 위해 YES로 처리
        return True


# -----------------------------
# Core
# -----------------------------
def answer_question(
    client: OpenAI,
    llm_model: str,
    es: Elasticsearch,
    es_index: str,
    messages: List[Dict[str, str]],
    topk: int,
    multilingual: bool,
    sparse_candidates: int,
    rrf_k: int,
    debug: bool = False,
) -> (Dict[str, Any], Optional[Dict[str, Any]]):

    # 제출 포맷(=jsonl 라인)과 동일한 구조
    pred = {
        "standalone_query": "",
        "topk": [],
        "answer": "",
        "references": [],
    }

    dbg = None

    # 1) standalone query 생성 (멀티턴 보강)
    standalone = llm_make_standalone_query(client, llm_model, messages)
    if not standalone:
        # fallback: 마지막 user 발화
        for m in reversed(messages or []):
            if m.get("role") == "user":
                standalone = (m.get("content") or "").strip()
                break
    pred["standalone_query"] = standalone

    if not standalone:
        if debug:
            dbg = {"standalone_query": "", "is_science": None, "ko_query": "", "en_query": "", "fused_top": []}
        return pred, dbg

    # 2) 검색 (항상 수행)
    ko_hits = sparse_retrieve(es, es_index, standalone, sparse_candidates)
    ko_docids = [h.get("_source", {}).get("docid", "") for h in ko_hits]
    ko_docids = [d for d in ko_docids if d]

    ranklists = {"ko": ko_docids}
    en_query = ""

    if multilingual:
        en_query = translate_to_en(client, llm_model, standalone)
        if en_query:
            en_hits = sparse_retrieve(es, es_index, en_query, sparse_candidates)
            en_docids = [h.get("_source", {}).get("docid", "") for h in en_hits]
            en_docids = [d for d in en_docids if d]
            ranklists["en"] = en_docids

    fused = reciprocal_rank_fusion(ranklists, k=rrf_k)
    fused_top = fused[:topk]

    # 3) 후단 판별 (대화 전체 기반)
    is_science = llm_is_science_postfilter(client, llm_model, messages)

    if not is_science:
        # 비과학이면 topk 비움 (대회 MAP else 로직에서 1점)
        fused_top = []

    pred["topk"] = fused_top

    if debug:
        dbg = {
            "standalone_query": standalone,
            "is_science": is_science,
            "ko_query": standalone,
            "en_query": en_query,
            "ko_hits_count": len(ko_docids),
            "en_hits_count": len(ranklists.get("en", [])),
            "fused_top": fused_top,
        }

    return pred, dbg


# -----------------------------
# Main (jsonl-in-csv)
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--out_path", required=True)  # 확장자는 .csv로 저장해도 내용은 jsonl
    parser.add_argument("--debug_out_path", default="", help="Optional debug jsonl path (also can end with .csv)")

    parser.add_argument("--es_host", required=True)
    parser.add_argument("--es_username", required=True)
    parser.add_argument("--es_password", required=True)
    parser.add_argument("--ca_certs", required=True)
    parser.add_argument("--es_index", default="test")

    parser.add_argument("--llm_model", default="gpt-4o-mini")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--multilingual", action="store_true")

    parser.add_argument("--sparse_candidates", type=int, default=300)
    parser.add_argument("--rrf_k", type=int, default=60)

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()
    es = Elasticsearch(
        [args.es_host],
        basic_auth=(args.es_username, args.es_password),
        ca_certs=args.ca_certs,
    )

    with open(args.eval_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    debug_enabled = bool(args.debug_out_path.strip())
    debug_f = open(args.debug_out_path, "w", encoding="utf-8") if debug_enabled else None

    # 핵심: out_path는 ".csv"여도, 내용은 jsonl로 씀
    with open(args.out_path, "w", encoding="utf-8") as out_f:
        for s in samples:
            eval_id = s.get("eval_id")
            messages = s.get("msg", [])

            pred, dbg = answer_question(
                client=client,
                llm_model=args.llm_model,
                es=es,
                es_index=args.es_index,
                messages=messages,
                topk=args.topk,
                multilingual=args.multilingual,
                sparse_candidates=args.sparse_candidates,
                rrf_k=args.rrf_k,
                debug=debug_enabled,
            )

            line_obj = {
                "eval_id": eval_id,
                "standalone_query": pred["standalone_query"],
                "topk": pred["topk"],
                "answer": pred["answer"],
                "references": pred["references"],
            }
            out_f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

            if debug_enabled and dbg is not None:
                dbg_obj = {"eval_id": eval_id, **dbg}
                debug_f.write(json.dumps(dbg_obj, ensure_ascii=False) + "\n")

    if debug_f:
        debug_f.close()

    print(f"[DONE] Saved jsonl content to → {args.out_path}")
    if debug_enabled:
        print(f"[DONE] Saved debug jsonl to → {args.debug_out_path}")


if __name__ == "__main__":
    main()
