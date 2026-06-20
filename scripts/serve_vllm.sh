#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${LLM_MODEL:-Qwen/Qwen3.5-9B}"
PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_BIN="${VLLM_BIN:-/workspace/.venv-vllm/bin/vllm}"

exec "$VLLM_BIN" serve "$MODEL"   --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"   --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}"   --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"   --language-model-only   --port "$PORT"
