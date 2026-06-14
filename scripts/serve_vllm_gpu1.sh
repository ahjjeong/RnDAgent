#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

MODEL="${LLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
PORT="${VLLM_PORT:-8000}"

exec vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  --port "$PORT"
