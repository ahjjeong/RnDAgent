#!/usr/bin/env bash
set -euo pipefail

# The model itself is served by vLLM on physical GPU 1.
# Keep this client process off CUDA so it does not touch other GPUs.
export CUDA_VISIBLE_DEVICES=""
export LLM_BACKEND="${LLM_BACKEND:-vllm}"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export WEB_RAG_ENABLED="${WEB_RAG_ENABLED:-0}"
# If enabled, an LLM gate decides whether internal prior-project RAG is enough.
export WEB_RAG_ON_DEMAND="${WEB_RAG_ON_DEMAND:-1}"
export RAG_DEVICE="${RAG_DEVICE:-cpu}"

OUT="${1:-results_all.jsonl}"

if [[ "${USE_RAG:-1}" == "1" ]]; then
  python main.py --all --resume --out "$OUT"
else
  python main.py --all --resume --no-rag --out "$OUT"
fi
