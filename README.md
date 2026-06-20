# RnDAgent

[한국어](README.md) | [English](README_EN.md)

**RnDAgent**는 국가 R&D 과제 선정 평가를 LLM 기반 다중 에이전트 토론 구조로 모사하고, NTIS 성과 데이터를 이용해 예측 결과를 검증하는 실험용 프레임워크입니다.  
본 저장소는 **서울대학교 산업공학과 「기술혁신이론 및 연구방법론」 수업의 팀프로젝트**로 수행된 연구의 코드와 실험 산출물을 정리한 것입니다.

연구의 핵심 질문은 다음과 같습니다.

> 전문가 평가위원회처럼 서로 다른 관점을 가진 LLM 에이전트들이 과제를 독립 평가하고 토론하면, 실제 R&D 성과를 어느 정도 예측할 수 있는가?

## 방법론 개요

<img width="5142" height="1934" alt="image" src="https://github.com/user-attachments/assets/27826230-462e-4311-8cf8-337f139945e2" />

프레임워크는 크게 네 단계로 구성됩니다.

1. **Data Input**  
   NTIS 과제정보와 성과정보를 입력으로 사용합니다. 과제정보에는 연구목표, 연구내용, 키워드, 기술분류, 연구비, 연구기간, 수행기관, 연구책임자 및 참여연구원 정보가 포함됩니다. 성과정보는 사후 검증을 위한 ground-truth label 구성에 사용됩니다.

2. **Multi-Agent System**  
   생명과학 기초연구 분야의 평가위원 3인을 LLM persona로 구성합니다. 각 에이전트는 창의성, 수행계획 충실성, 연구개발 역량을 모두 평가하되, 서로 다른 전문적 관점에서 판단합니다.

3. **Evaluation Output**  
   최종 moderator agent가 토론 결과를 종합하여 과제별 예상 성과 점수, confidence, 항목별 consensus score, 주요 판단 근거를 생성합니다.

4. **Validation**  
   예측 점수를 SCI(E) 논문 성과 기반의 실제 high-performance label과 비교하여 분류 성능과 ranking quality를 평가합니다.

## LangGraph 구조

<p align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/eaa74977-5d28-460c-ba0d-7909cae51a32" />
</p>

실행 파이프라인은 LangGraph로 구현되어 있습니다.

- `prefetch`: 내부 RAG 검색을 한 번 수행하고 모든 에이전트가 같은 근거를 공유합니다.
- `phase1`: 세 평가위원 에이전트가 독립적으로 1차 평가를 수행합니다.
- `coordinator`: 평가 차이와 쟁점을 요약하고 2차 토론 질문을 생성합니다.
- `phase2`: 각 에이전트가 다른 위원의 평가와 coordinator 질문을 반영해 반론/수정 의견을 제시합니다.
- `moderator`: 전체 토론을 종합하여 최종 성과 예측 verdict를 생성합니다.

## 저장소 구조

```text
.
├── main.py                              # 데이터 로딩, RAG, LangGraph 실행, 결과 저장 엔트리포인트
├── requirements.txt                     # Python 의존성
├── results_2019_persona_outcome_rag.jsonl
├── scripts/
│   ├── serve_vllm.sh                    # vLLM 서버 실행 스크립트
│   └── run_2019_vllm.sh                 # 2019년 종료 과제 전체 실행 스크립트
└── src/
    ├── agents.py                        # 평가위원, 토론자, facilitator agent 정의
    ├── config.py                        # 모델, 데이터 경로, 컬럼 설정
    ├── data_loader.py                   # 데이터 로딩 및 연도별 컬럼명 정규화
    ├── evaluate.py                      # 성과 label 비교 및 metric 계산
    ├── graph.py                         # LangGraph 기반 multi-agent debate pipeline
    ├── llm.py                           # vLLM / transformers backend wrapper
    ├── prompt_config.py                 # 평가 항목, persona, prompt 설정
    ├── rag.py                           # FAISS 기반 내부 RAG 검색
    └── web_rag.py                       # 선택적 외부 문헌 검색 보강
```

## 데이터 정제

본 프로젝트는 NTIS 과제정보와 성과정보를 결합해 평가용 데이터를 구성합니다. 원천 데이터는 용량과 재배포 조건 문제로 저장소에 포함하지 않는 것을 전제로 합니다.

데이터 정제 과정은 다음 흐름을 따릅니다.

1. **연도별 컬럼명 통합**  
   NTIS 데이터는 연도에 따라 컬럼명이 달라질 수 있으므로, `src/config.py`의 `COLUMN_ALIASES`를 사용해 구 컬럼명을 최신 정규 컬럼명으로 통합합니다.

2. **과제 입력 필드 구성**  
   `src/prompt_config.py`에서 창의성, 수행계획 충실성, 연구개발 역량 평가에 필요한 컬럼을 정의합니다. 각 과제는 `src/data_loader.py`의 `project_views()`를 통해 agent별 입력 view로 변환됩니다.

3. **성과 label 구성**  
   논문 성과는 과제기간 및 종료 후 4년 이내 SCI(E) 성과를 기준으로 집계합니다. 비교 가능한 strata 안에서 상위 20%에 해당하는 과제를 high-performance project로 정의합니다.

4. **누출 방지 RAG**  
   내부 RAG는 평가 대상 과제보다 이전에 종료된 과제만 검색합니다. 이를 통해 미래 성과 정보가 현재 평가에 직접 누출되지 않도록 합니다.

기본 실행 스크립트는 아래 파일을 기대합니다.

```text
dataset/projects_labeled_sci_performance_period_plus4.csv
```

별도 데이터 파일을 사용할 경우 `--dataset` 인자로 CSV, XLSX, Parquet 경로를 지정할 수 있습니다.

## 실행 방법

### 1. 환경 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

대규모 실행은 vLLM backend 사용을 권장합니다. 기본 모델은 `Qwen/Qwen3.5-9B`입니다.

### 2. vLLM 서버 실행

```bash
bash scripts/serve_vllm.sh
```

필요하면 환경변수로 모델, 포트, GPU 설정을 바꿀 수 있습니다.

```bash
export LLM_MODEL="Qwen/Qwen3.5-9B"
export VLLM_PORT=8000
export CUDA_VISIBLE_DEVICES=0
bash scripts/serve_vllm.sh
```

### 3. 2019년 종료 과제 실행

```bash
bash scripts/run_2019_vllm.sh
```

이 스크립트는 기본적으로 다음 설정으로 실행됩니다.

- 입력 데이터: `dataset/projects_labeled_sci_performance_period_plus4.csv`
- 출력 파일: `results_2019_persona_outcome_rag.jsonl`
- 평가 대상: 종료연도 2019년 과제
- LLM backend: vLLM
- 내부 RAG: 사용
- resume: 사용

### 4. 직접 실행 예시

일부 샘플만 실행하려면:

```bash
python main.py \
  --n 5 \
  --dataset dataset/projects_labeled_sci_performance_period_plus4.csv \
  --out results_sample.jsonl
```

전체 데이터를 실행하려면:

```bash
python main.py \
  --all \
  --resume \
  --dataset dataset/projects_labeled_sci_performance_period_plus4.csv \
  --target-end-year 2019 \
  --out results_2019_persona_outcome_rag.jsonl
```

RAG 없이 ablation 형태로 실행하려면:

```bash
python main.py \
  --all \
  --dataset dataset/projects_labeled_sci_performance_period_plus4.csv \
  --target-end-year 2019 \
  --no-rag \
  --out results_2019_no_rag.jsonl
```

## 결과물

실행 결과는 JSONL 형식으로 저장됩니다. 각 line은 하나의 과제 평가 결과이며, 주요 필드는 다음과 같습니다.

- `project_id`: 과제 고유번호
- `title`: 과제명
- `phase1`: 세 평가위원의 1차 독립 평가
- `coordinator_issues`: 평가 쟁점 및 2차 토론 질문
- `phase2_rebuttals`: 세 평가위원의 2차 반론/수정 의견
- `verdict`: moderator가 생성한 최종 성과 예측
- `validation`: 실제 성과 label과의 비교 결과

`main.py`는 결과 파일과 함께 metric 요약 파일도 생성합니다.

```text
results_2019_persona_outcome_rag.jsonl
results_2019_persona_outcome_rag.jsonl.metrics.json
```

논문 실험에서 2019년 종료 생명과학 기초연구 3,078개 과제를 대상으로 평가한 주요 결과는 다음과 같습니다.

| Metric | Value |
|---|---:|
| Accuracy | 0.696 |
| Balanced Accuracy | 0.535 |
| Precision (high-performance) | 0.271 |
| Recall (high-performance) | 0.256 |
| Specificity (low-performance) | 0.815 |
| F1 (high-performance) | 0.263 |
| ROC-AUC | 0.576 |

결과는 무작위 기준보다는 약간 높은 신호를 보였지만, 실제 고성과 과제를 안정적으로 식별하기에는 제한적이었습니다. 따라서 본 프레임워크는 전문가 평가를 대체하기보다는 설명 가능한 decision-support 및 screening 보조 도구로 해석하는 것이 적절합니다.

## 모델 및 설정

주요 설정은 `src/config.py`에서 관리합니다.

| 설정 | 기본값 | 설명 |
|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen3.5-9B` | 평가와 토론에 사용할 LLM |
| `LLM_BACKEND` | `vllm` | `vllm` 또는 `transformers` |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible vLLM endpoint |
| `EMBED_MODEL` | `BAAI/bge-m3` | 내부 RAG embedding model |
| `RAG_DEVICE` | `cpu` | RAG embedding 실행 장치 |
| `WEB_RAG_ENABLED` | `0` | 외부 문헌 검색 강제 사용 여부 |

## 연구팀

서울대학교 산업공학과  

| 이름 | 과정 | 소속 |
|---|---|---|
| 김지혁 | 석사과정 | 기술 인텔리전스 연구실 |
| 박동우 | 석사과정 | 삶향상기술연구실 |
| 안재홍 | 석사과정 | 기술 인텔리전스 연구실 |
| 최아정 | 석사과정 | 데이터과학 및 비즈니스 애널리틱스 연구실 |

## 주의사항

- NTIS 원천 데이터 및 정제 데이터는 저장소에 포함되어 있지 않습니다. 실행 전 `dataset/` 폴더에 필요한 데이터 파일을 배치해야 합니다.
- vLLM 기반 실행은 GPU 환경을 권장합니다.
- 생성형 모델의 평가는 prompt, persona, decoding 설정에 민감합니다. 논문 결과 재현 시 모델 버전과 실행 설정을 함께 고정하는 것이 좋습니다.
