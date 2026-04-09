#!/bin/bash

until curl -f "${EVAL_API_URL}/health" > /dev/null 2>&1; do
    exit_code=$?
    if [ $exit_code -eq 22 ]; then
        http_status=$(curl -s -o /dev/null -w "%{http_code}" "${EVAL_API_URL}/health")
        if [ "$http_status" -eq 401 ]; then
            echo "Authentication failed (401), stopping health check"
            break
        fi
    fi
    echo "waiting for vllm at ${EVAL_API_URL}..."
    sleep 5
done

CMD="evalscope eval \
 --model \"${EVAL_MODEL_NAME}\" \
 --api-url \"${EVAL_API_URL}/v1\" \
 --api-key \"${EVAL_API_KEY}\" \
 --datasets ${EVAL_DATASETS} \
 --eval-type \"${EVAL_TYPE}\" \
 --generation-config '${EVAL_GENERATION_CONFIG}' \
 --eval-batch-size ${EVAL_BATCH_SIZE} \
 --dataset-args '${EVAL_DATASET_ARGS}' \
 --ignore-errors"

[ -n "${EVAL_LIMIT}" ] && CMD="$CMD --limit ${EVAL_LIMIT}"
[ -n "${EVAL_USE_CACHE}" ] && CMD="$CMD --use-cache ${EVAL_USE_CACHE}"

echo "start evaluation with command ${CMD}..."

eval $CMD
