#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-dataset/projects_labeled_sci_performance_period_plus4.csv}"
OUT="${OUT:-results_2019_persona_outcome_rag.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/.venv-vllm/bin/python}"

export CUDA_VISIBLE_DEVICES=""
export LLM_BACKEND="${LLM_BACKEND:-vllm}"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.5-9B}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export WEB_RAG_ENABLED="${WEB_RAG_ENABLED:-0}"
export WEB_RAG_ON_DEMAND="${WEB_RAG_ON_DEMAND:-0}"
export RAG_DEVICE="${RAG_DEVICE:-cpu}"
export VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-false}"

ARGS=(main.py --all --resume --dataset "$DATASET_PATH" --out "$OUT" --target-end-year 2019 --no-validation)

if [[ "${USE_RAG:-1}" == "1" ]]; then
  "$PYTHON_BIN" "${ARGS[@]}"
else
  "$PYTHON_BIN" "${ARGS[@]}" --no-rag
fi
