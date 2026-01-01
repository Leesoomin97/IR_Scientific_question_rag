# 팀 레포: https://github.com/AIBootcamp16/scientific-knowledge-question-answering-ir-4
---
# Scientific Knowledge Question Answering  
과학 지식 질의 응답을 위한 RAG 기반 Information Retrieval 시스템

## 1. Project Overview

본 프로젝트는 **과학 상식 질문에 대해 신뢰 가능한 문서를 검색하고 이를 기반으로 답변을 생성하는 RAG(Retrieval Augmented Generation) 시스템**을 구축하는 것을 목표로 한다.

대회는 단순한 LLM 응답 생성이 아니라,  
- 질문 의도 판단  
- 검색 대상 문서 추출  
- 검색 결과의 정밀도  

를 종합적으로 평가하는 **Information Retrieval(IR) 중심 과제**이다.

특히 본 프로젝트에서는 **모델 성능보다 검색 파이프라인 설계와 Query Understanding이 성능을 좌우한다**는 문제 정의 하에 접근하였다.

---

## 2. Task Definition

시스템은 다음 시나리오를 가정한다.

1. 사용자 입력은 **단일 질문 또는 멀티턴 대화** 형태
2. 입력이 **과학 지식 질문인지 판별**
3. 과학 질문일 경우:
   - 검색엔진을 통해 관련 문서 검색
   - 검색된 문서를 기반으로 답변 생성
4. 비과학 질문일 경우:
   - 검색을 수행하지 않고 직접 응답

👉 핵심 과제는  
**주어진 대화 히스토리로부터 검색에 적합한 standalone query를 생성하고, 정답 문서를 포함한 후보군을 안정적으로 확보하는 것**이다.

---

## 3. Dataset

### 3.1 Indexing Documents
- 약 **4,200개 과학 상식 문서**
- 각 문서는 다음 필드로 구성됨:
  - `docid`: 문서 고유 ID
  - `src`: 출처 (MMLU / ARC 기반)
  - `content`: 실제 검색 대상 텍스트

문서는 **jsonl 포맷**으로 제공되며, 검색엔진 색인을 위한 순수 지식 문서로 사용된다.

---

### 3.2 Evaluation Data
- 총 **220개 질문**
  - 단일 턴 질문
  - 멀티턴 대화 (20개)
  - 비과학 일반 대화 (20개)
- 입력은 LLM 메시지 포맷(`role`, `content`)을 그대로 따름

멀티턴 환경을 고려해야 하므로 **Standalone Query 생성 품질이 매우 중요**하다.

---

## 4. System Architecture

```
User Message (Multi-turn)
        ↓
Intent Classification
        ↓
Standalone Query Generation
        ↓
Sparse Retrieval (BM25)
        +
Dense Retrieval (Embedding)
        ↓
Candidate Pool (Top-K)
        ↓
RRF Fusion
        ↓
Reranker
        ↓
Final Documents
```

---

## 5. Design Principles

### 5.1 Recall First
- 정답 문서가 초기 후보군에 포함되지 않으면 이후 단계는 무의미
- `num_candidates`를 충분히 확장하여 **Recall 확보를 최우선 목표**로 설정

### 5.2 Late Precision Recovery
- 초기 검색 단계에서는 노이즈 허용
- Reranker를 통해 Precision을 회복하는 구조 채택

### 5.3 Query Understanding over Model Choice
- Dense 모델 변경보다 **Standalone Query 품질 개선이 성능에 더 큰 영향**
- 검색 실패의 대부분은 Query 변환 단계에서 발생

---

## 6. Retrieval Strategy

### Sparse Retrieval
- Elasticsearch BM25
- 키워드 일치에 강하고 안정적인 성능

### Dense Retrieval
- 의미 기반 검색
- 데이터 규모가 작을 경우 노이즈 증가 가능성 존재

### Hybrid Retrieval
- Sparse + Dense 결과를 **RRF(Rank Reciprocal Fusion)**로 결합
- 단순 병합이 아닌 랭크 기반 결합 전략 사용

---

## 7. Reranking

- 초기 후보군: 최대 **Top-3000**
- Reranker를 통해 상위 문서 재정렬
- Recall이 확보된 경우에만 Precision 개선 효과 확인

---

## 8. Experiments & Findings

- Dense 단독 검색은 오히려 성능 저하 발생 가능
- Hybrid Search는 **조합 방식과 후보군 크기 설계가 핵심**
- RAG 성능 병목은 LLM이 아니라 **IR 구조와 Query 처리 단계**

---

## 9. Best Performing Configuration (v10)

본 프로젝트의 최고 성능은 **v10 Hybrid RAG 파이프라인**에서 달성되었다.  
v10은 단순히 새로운 모델을 추가한 버전이 아니라, **이전 실험(v1–v9)에서 반복적으로 관찰된 실패 원인을 구조적으로 제거한 최종 설계**에 해당한다.

---

### 9.1 Design Motivation

v1–v9 실험 과정에서 다음과 같은 문제점이 반복적으로 확인되었다.

- Dense retrieval 단독 또는 과도한 비중 사용 시 성능 불안정
- 초기 후보군 Recall이 충분히 확보되지 않을 경우 reranking 효과 미미
- 멀티턴 대화 환경에서 질문 의도와 검색 쿼리 간 불일치로 인한 검색 실패

이에 따라 v10에서는  
**Query Understanding → Recall 극대화 → Precision 회복**이라는 단계적 전략을 명확히 분리하여 설계하였다.

---

### 9.2 v10 Pipeline Overview

```
Multi-turn Messages
↓
LLM-based Routing & Standalone Query Generation
↓
Sparse Retrieval (BM25)
+
Dense Retrieval (Embedding-based)
↓
Candidate Expansion (Recall-oriented)
↓
RRF (Rank Reciprocal Fusion)
↓
Reranker (Top-N Precision Optimization)
↓
Final Top-K Documents
```

---

### 9.3 Key Components

#### 9.3.1 Standalone Query Generation (LLM-based)

- LLM을 **답변 생성기**가 아닌 **검색 최적화 도구**로 사용
- 멀티턴 대화 히스토리를 검색 엔진 친화적인 단일 쿼리로 재구성
- 애매한 경우에도 검색을 수행하도록 설계하여 검색 누락을 최소화

이를 통해 멀티턴 환경에서 빈번하게 발생하던 **의도 불일치 기반 검색 실패를 크게 완화**할 수 있었다.

---

#### 9.3.2 Hybrid Retrieval with Recall Priority

- **Sparse Retrieval (BM25)**  
  - 키워드 기반 검색으로 안정적인 1차 후보군 확보

- **Dense Retrieval**  
  - 의미적 유사성을 활용해 Recall을 보완
  - 단독 사용이 아닌 Sparse 검색을 보조하는 역할로 제한

Dense retrieval은 Recall 보완 수단으로만 사용되었으며,  
검색의 주된 신호는 Sparse retrieval이 담당하도록 설계되었다.

---

#### 9.3.3 Dense Weight Sensitivity Analysis (Critical Finding)

Sparse–Dense 결합 비율에 대한 실험을 반복적으로 수행한 결과,  
**Dense retrieval의 비중이 약 45%를 초과할 경우 성능이 오히려 하락하는 현상**이 관찰되었다.

이는 Dense 모델의 성능 자체 문제라기보다는,  
**본 과제 데이터의 특성과 질의 분포에 따른 구조적 한계**로 해석할 수 있다.

- 문서는 비교적 **짧고 정보 밀도가 높으며**
- 질문은 특정 개념·정의·사실을 직접적으로 묻는 경우가 많았다.

이러한 조건에서 Dense retrieval은  
*의미적으로는 유사하지만 정답과는 무관한 문서*를 상위 랭크에 포함시키는 경향을 보였고,  
Dense 비중이 증가할수록 Sparse retrieval이 제공하던 **정확한 키워드 기반 신호가 희석**되었다.

실제로 Dense weight를 **0.45 이상**으로 설정한 실험에서는  
RRF 이후 후보군 내 의미적 노이즈가 증가했고,  
이로 인해 reranking 단계에서도 성능 회복이 제한되었다.

---

#### 9.3.4 RRF-based Fusion

- Sparse / Dense 결과를 단순 병합하지 않고  
  **Rank Reciprocal Fusion (RRF)** 방식으로 결합
- 점수 스케일 차이 문제를 회피하고 랭크 기반 신호를 안정적으로 통합
- Hybrid Search의 불안정성을 완화하는 핵심 요소로 작용

v10에서는 Dense weight를 **0.45 이하로 제한**하여  
Dense가 성능을 저해하기 시작하는 구간을 명확히 회피하였다.

---

#### 9.3.5 Reranking for Late-stage Precision Recovery

- 초기 단계에서는 **Recall 확보를 최우선 목표**로 설정
- 이후 상위 후보 문서에 대해 Reranker를 적용하여 Precision 회복
- Recall이 충분히 확보된 경우에만 reranking이 유의미한 성능 향상을 보임을 확인

---

### 9.4 Why v10 Worked

v10의 성능 향상은 특정 모델의 변경 때문이 아니라,  
**각 단계의 역할을 명확히 분리하고 그 한계를 실험적으로 확인한 뒤 이를 설계에 반영한 결과**였다.

- Dense retrieval의 기여와 한계를 명확히 정의
- Recall과 Precision을 동시에 최적화하려 하지 않고 단계적으로 처리
- 실패 실험에서 얻은 관찰 결과를 구조적 설계로 연결

결과적으로 v10은  
**안정적인 Recall 확보와 일관된 Precision 회복을 동시에 달성한 구성**이었다.

---

### 9.5 Summary

| Aspect | v10 특징 |
|---|---|
| Query Handling | LLM 기반 Standalone Query 생성 |
| Retrieval | Sparse + Dense Hybrid |
| Fusion | RRF 기반 랭크 결합 |
| Optimization Strategy | Recall-first, Late Precision Recovery |
| Key Insight | Dense 비중 45% 초과 시 성능 저하 |
| Strength | 멀티턴 환경에서 안정적인 검색 성능 |

---

## 10. Repository Structure

```
├── test.py
└── README.md
```

---

## 11. Lessons Learned & Reflection

이번 대회를 통해 **RAG 기반 QA 시스템의 성능 병목은 LLM이 아니라 Information Retrieval 구조에 있다**는 점을 명확히 체감했다.

- Recall 확보는 모든 단계의 전제 조건
- Hybrid Search는 단순 조합이 아니라 구조적 설계 문제
- 멀티턴 환경에서는 **Standalone Query 생성이 핵심 병목**
- Precision은 초기 단계가 아니라 **후단(Reranker)에서 회복해야 효과적**

아쉬웠던 점은 **프롬프트 엔지니어링을 충분히 깊게 실험하지 못했다는 점**이다.

제출 횟수 제한으로 인해  
- 프롬프트 문장 단위 수정
- 지시 강도 조절
- 출력 형식 유도 방식 비교  

와 같은 **미세한 프롬프트 설계 실험을 반복적으로 검증하지 못했다**.

그럼에도 이 경험을 통해,  
**프롬프트 엔지니어링 역시 IR 시스템의 중요한 설계 요소**임을 분명히 인식하게 되었다.
