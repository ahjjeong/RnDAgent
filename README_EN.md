# RnDAgent

[한국어](README.md) | [English](README_EN.md)

**RnDAgent** is an experimental framework that simulates public R&D project selection using an LLM-based multi-agent debate structure and validates the resulting predictions with NTIS performance data.  
This repository contains the code and experimental outputs from a team project for the **Technology Innovation Theory and Research Methodology** course in the **Department of Industrial Engineering at Seoul National University**.

The central research question is:

> To what extent can LLM agents with different expert perspectives predict actual R&D performance when they independently evaluate projects and deliberate like an expert review panel?

## Methodology Overview

<img width="5142" height="1934" alt="image" src="https://github.com/user-attachments/assets/27826230-462e-4311-8cf8-337f139945e2" />

The framework consists of four main stages.

1. **Data Input**  
   The system uses NTIS project information and performance data as input. Project information includes research objectives, research content, keywords, technology classifications, budget, research period, host institution, principal investigator, and participating researcher information. Performance data are used to construct ground-truth labels for ex post validation.

2. **Multi-Agent System**  
   Three LLM personas are configured as reviewers for basic life-science research projects. Each agent evaluates creativity, fidelity of the execution plan, and research capability, but does so from a distinct professional perspective.

3. **Evaluation Output**  
   A final moderator agent synthesizes the debate and produces a predicted performance score, confidence value, criterion-level consensus scores, and key reasons for each project.

4. **Validation**  
   The predicted scores are compared with actual high-performance labels based on SCI(E) publication outcomes, and the framework is evaluated using classification metrics and ranking quality.

## LangGraph Architecture

<p align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/eaa74977-5d28-460c-ba0d-7909cae51a32" />
</p>

The execution pipeline is implemented with LangGraph.

- `prefetch`: Runs internal RAG once so that all agents share the same evidence.
- `phase1`: Three reviewer agents independently conduct the first-round evaluation.
- `coordinator`: Summarizes disagreements and key issues, then generates questions for the second-round debate.
- `phase2`: Each agent responds to the coordinator's questions and revises or defends its position.
- `moderator`: Synthesizes the full debate and produces the final performance prediction verdict.

## Repository Structure

```text
.
├── main.py                              # Entry point for data loading, RAG, LangGraph execution, and result saving
├── requirements.txt                     # Python dependencies
├── results_2019_persona_outcome_rag.jsonl
├── scripts/
│   ├── serve_vllm.sh                    # Script for launching the vLLM server
│   └── run_2019_vllm.sh                 # Script for running all projects ending in 2019
└── src/
    ├── agents.py                        # Reviewer, debater, and facilitator agent definitions
    ├── config.py                        # Model, data path, and column settings
    ├── data_loader.py                   # Data loading and cross-year column normalization
    ├── evaluate.py                      # Outcome-label comparison and metric calculation
    ├── graph.py                         # LangGraph-based multi-agent debate pipeline
    ├── llm.py                           # vLLM / transformers backend wrapper
    ├── prompt_config.py                 # Evaluation items, personas, and prompt configuration
    ├── rag.py                           # FAISS-based internal RAG retrieval
    └── web_rag.py                       # Optional external literature-search augmentation
```

## Data Preprocessing

This project constructs an evaluation dataset by combining NTIS project information and performance data. The raw data are not included in this repository due to file size and redistribution constraints.

The preprocessing workflow is as follows.

1. **Cross-year column normalization**  
   NTIS column names may differ by year. `COLUMN_ALIASES` in `src/config.py` maps older column names to the canonical current names.

2. **Project input-field construction**  
   `src/prompt_config.py` defines the columns used to evaluate creativity, execution-plan fidelity, and research capability. Each project row is converted into agent-specific input views through `project_views()` in `src/data_loader.py`.

3. **Performance-label construction**  
   Publication outcomes are counted over the project period and up to four years after project completion. A project is labeled as high-performing if it falls within the top 20% of comparable projects in the same stratum.

4. **Leakage-controlled RAG**  
   Internal RAG retrieves only projects that ended before the target project. This prevents future performance information from leaking into the evaluation.

The default run script expects the following file:

```text
dataset/projects_labeled_sci_performance_period_plus4.csv
```

To use a different dataset, pass the path to a CSV, XLSX, or Parquet file with the `--dataset` argument.

## How to Run

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For large-scale runs, the vLLM backend is recommended. The default model is `Qwen/Qwen3.5-9B`.

### 2. Start the vLLM Server

```bash
bash scripts/serve_vllm.sh
```

You can customize the model, port, and GPU settings with environment variables.

```bash
export LLM_MODEL="Qwen/Qwen3.5-9B"
export VLLM_PORT=8000
export CUDA_VISIBLE_DEVICES=0
bash scripts/serve_vllm.sh
```

### 3. Run Projects Ending in 2019

```bash
bash scripts/run_2019_vllm.sh
```

By default, this script uses the following settings.

- Input data: `dataset/projects_labeled_sci_performance_period_plus4.csv`
- Output file: `results_2019_persona_outcome_rag.jsonl`
- Target projects: projects with end year 2019
- LLM backend: vLLM
- Internal RAG: enabled
- Resume mode: enabled

### 4. Manual Execution Examples

Run a small sample:

```bash
python main.py \
  --n 5 \
  --dataset dataset/projects_labeled_sci_performance_period_plus4.csv \
  --out results_sample.jsonl
```

Run the full target set:

```bash
python main.py \
  --all \
  --resume \
  --dataset dataset/projects_labeled_sci_performance_period_plus4.csv \
  --target-end-year 2019 \
  --out results_2019_persona_outcome_rag.jsonl
```

Run an ablation without RAG:

```bash
python main.py \
  --all \
  --dataset dataset/projects_labeled_sci_performance_period_plus4.csv \
  --target-end-year 2019 \
  --no-rag \
  --out results_2019_no_rag.jsonl
```

## Outputs

The execution results are saved in JSONL format. Each line corresponds to one project evaluation result and contains the following key fields.

- `project_id`: unique project identifier
- `title`: project title
- `phase1`: first-round independent evaluations by the three reviewers
- `coordinator_issues`: key disagreements and second-round debate questions
- `phase2_rebuttals`: second-round rebuttals or revised opinions from the three reviewers
- `verdict`: final performance prediction generated by the moderator
- `validation`: comparison with actual performance labels

`main.py` also creates a metric summary file alongside the result file.

```text
results_2019_persona_outcome_rag.jsonl
results_2019_persona_outcome_rag.jsonl.metrics.json
```

In the paper experiment, the framework was evaluated on 3,078 basic life-science research projects ending in 2019. The main results are:

| Metric | Value |
|---|---:|
| Accuracy | 0.696 |
| Balanced Accuracy | 0.535 |
| Precision (high-performance) | 0.271 |
| Recall (high-performance) | 0.256 |
| Specificity (low-performance) | 0.815 |
| F1 (high-performance) | 0.263 |
| ROC-AUC | 0.576 |

The results indicate a signal slightly above the random baseline, but the framework is not yet reliable enough to identify high-performing projects with high confidence. It is therefore best understood as an explainable decision-support and screening aid rather than a replacement for expert review.

## Model and Configuration

Main settings are managed in `src/config.py`.

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen3.5-9B` | LLM used for evaluation and debate |
| `LLM_BACKEND` | `vllm` | `vllm` or `transformers` |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible vLLM endpoint |
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding model for internal RAG |
| `RAG_DEVICE` | `cpu` | Device for RAG embedding inference |
| `WEB_RAG_ENABLED` | `0` | Whether to force external literature-search augmentation |

## Team

Department of Industrial Engineering, Seoul National University

| Name | Program | Affiliation |
|---|---|---|
| Jaehong Ahn | Master's Student | Technology Intelligence Lab |
| Ajeong Choi | Master's Student | Data Science & Business Analytics Lab |
| Jihyuk Kim | Master's Student | Technology Intelligence Lab |
| Dongwoo Park | Master's Student | Life Enhancing Technology Lab |

## Notes

- Raw and preprocessed NTIS datasets are not included in this repository. Place the required data files under `dataset/` before running the pipeline.
- GPU execution is recommended for vLLM-based runs.
- Generative-model evaluations are sensitive to prompts, personas, and decoding settings. For reproducibility, fix the model version and execution settings.
