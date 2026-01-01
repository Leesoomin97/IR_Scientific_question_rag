# v1.py
import os
import json
import argparse
import traceback
from typing import List, Dict, Any

from elasticsearch import Elasticsearch, helpers
from openai import OpenAI


# -----------------------------
# Prompts (KEEP AS-IS)
# -----------------------------
PERSONA_FUNCTION_CALLING = """
[Role]
You are a scientific knowledge assistant in a Retrieval-Augmented Generation (RAG) system.

[Task Objective]
Determine whether the user's message STRICTLY requires scientific knowledge retrieval.

[Input Scope]
You are given a multi-turn conversation history.

[Behavior Rules]
- Call the search function ONLY IF:
  - The user explicitly asks for scientific facts, principles, mechanisms, definitions, or causes.
- Do NOT call search if the message is:
  - A greeting, emotion, opinion, encouragement, complaint, or casual conversation.
  - Ambiguous or not clearly a scientific knowledge request.
- If you are uncertain whether retrieval is needed, DO NOT call search.
- When calling search, generate a concise standalone scientific query.

[Output Constraints]
- Respond in Korean.
- Be concise.
""".strip()

PERSONA_QA = """
[Role]
You are a scientific knowledge assistant answering questions using retrieved reference documents.

[Task Objective]
Generate an accurate answer strictly based on the provided reference documents.

[Behavior Rules]
- Use ONLY the information in the reference documents.
- Do NOT add external knowledge.
- If the answer cannot be derived, leave the answer blank.

[Output Constraints]
- Answer in Korean.
- Be concise and factual.
""".strip()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "standalone_query": {
                        "type": "string"
                    }
                },
                "required": ["standalone_query"]
            }
        }
    }
]


# -----------------------------
# Elasticsearch helpers
# -----------------------------
def sparse_retrieve(
    es: Elasticsearch,
    index: str,
    query_str: str,
    size: int
) -> Dict[str, Any]:
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index=index, query=query, size=size)


# -----------------------------
# RAG core
# -----------------------------
def answer_question(
    client: OpenAI,
    llm_model: str,
    es: Elasticsearch,
    es_index: str,
    messages: List[Dict[str, str]],
    topk_size: int
) -> Dict[str, Any]:

    # 고정 출력 포맷
    response = {
        "standalone_query": "",
        "topk": [],
        "answer": "",
        "references": []
    }

    # 1) function calling 판단
    try:
        fc = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": PERSONA_FUNCTION_CALLING}] + messages,
            tools=TOOLS,
            temperature=0,
            seed=1,
            timeout=20
        )
    except Exception:
        traceback.print_exc()
        return response

    tool_calls = fc.choices[0].message.tool_calls

    # 2) 검색 호출된 경우
    if tool_calls:
        try:
            args = json.loads(tool_calls[0].function.arguments)
            standalone_query = args.get("standalone_query", "").strip()
        except Exception:
            return response

        if not standalone_query:
            return response

        response["standalone_query"] = standalone_query

        search_result = sparse_retrieve(
            es=es,
            index=es_index,
            query_str=standalone_query,
            size=topk_size
        )

        hits = search_result.get("hits", {}).get("hits", [])
        contexts = []

        for h in hits[:topk_size]:
            src = h.get("_source", {})
            docid = src.get("docid", "")
            content = src.get("content", "")
            score = h.get("_score", 0.0)

            if docid:
                response["topk"].append(docid)

            contexts.append(content)
            response["references"].append({
                "score": score,
                "content": content
            })

        # QA 단계 (reference-only)
        try:
            qa = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": PERSONA_QA},
                    *messages,
                    {"role": "assistant", "content": json.dumps(contexts, ensure_ascii=False)}
                ],
                temperature=0,
                seed=1,
                timeout=40
            )
            response["answer"] = qa.choices[0].message.content or ""
        except Exception:
            traceback.print_exc()

    # 3) 검색 안 하면 전부 빈칸 유지
    return response


# -----------------------------
# Load ALL eval samples
# -----------------------------
def load_all_eval_samples(eval_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)

    parser.add_argument("--es_host", type=str, default="https://localhost:9200")
    parser.add_argument("--es_username", type=str, default="elastic")
    parser.add_argument("--es_password", type=str, required=True)
    parser.add_argument("--ca_certs", type=str, required=True)
    parser.add_argument("--es_index", type=str, default="test")

    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--topk", type=int, default=3)

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()

    es = Elasticsearch(
        [args.es_host],
        basic_auth=(args.es_username, args.es_password),
        ca_certs=args.ca_certs
    )

    samples = load_all_eval_samples(args.eval_path)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(samples):
            eval_id = row["eval_id"]
            messages = row["msg"]

            resp = answer_question(
                client=client,
                llm_model=args.llm_model,
                es=es,
                es_index=args.es_index,
                messages=messages,
                topk_size=args.topk
            )

            output = {
                "eval_id": eval_id,
                "standalone_query": resp["standalone_query"],
                "topk": resp["topk"],
                "answer": resp["answer"],
                "references": resp["references"]
            }

            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Saved submission to {args.out_path}")


if __name__ == "__main__":
    main()
