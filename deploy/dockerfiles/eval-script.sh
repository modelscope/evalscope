#!/bin/bash

until curl -f "${EVAL_API_URL}/health" > /dev/null 2>&1; do
    exit_code=$?
    if [ $exit_code -eq 22 ]; then
        # curl 返回 22 表示 HTTP 4xx 错误，检查是否是 401
        http_status=$(curl -s -o /dev/null -w "%{http_code}" "${EVAL_API_URL}/health")
        if [ "$http_status" -eq 401 ]; then
            echo "Authentication failed (401), stopping health check"
            break
        fi
    fi
    echo "waiting for vllm..."
    sleep 5
done

echo "start evaluation..."

GENERATION_CONFIG="{\"max_tokens\": ${EVAL_MAX_TOKENS}}"

evalscope eval \
 --model "${EVAL_MODEL_NAME}" \
 --api-url "${EVAL_API_URL}/v1" \
 --api-key "${EVAL_API_KEY}" \
 --datasets "${EVAL_DATASETS}" \
 --eval-type "${EVAL_TYPE}" \
 --generation-config "${GENERATION_CONFIG}"
