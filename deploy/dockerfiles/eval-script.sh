#!/bin/bash

until curl -f "http://vllm-service:${VLLM_PORT}/health" > /dev/null 2>&1; do
    echo "waiting for vllm..."
    sleep 5
done

echo "start evaluation..."

GENERATION_CONFIG="{\"max_tokens\": ${EVAL_MAX_TOKENS}}"

evalscope eval \
 --model "${EVAL_MODEL_NAME}" \
 --api-url "${EVAL_API_URL}" \
 --api-key "${EVAL_API_KEY}" \
 --datasets "${EVAL_DATASETS}" \
 --eval-type "${EVAL_TYPE}" \
 --generation-config "${GENERATION_CONFIG}" \
 --limit "${EVAL_LIMIT}"
