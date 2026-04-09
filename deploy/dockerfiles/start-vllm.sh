#!/bin/bash

export NCCL_P2P_DISABLE=1

vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --dtype "${VLLM_DTYPE}" \
    --trust-remote-code \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE} \
    --data-parallel-size ${VLLM_DATA_PARALLEL_SIZE} \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}"
