#!/bin/bash
cd /mnt/nas2/yunlin.myl/evalscope
export $(grep DASHSCOPE_API_KEY .env 2>/dev/null | xargs)
PYTHON=/mnt/nas2/anaconda3/envs/eval/bin/python

echo "=== Testing VQAv2 (3 samples) ==="
$PYTHON -m evalscope.run \
    --model qwen-vl-plus \
    --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key "$DASHSCOPE_API_KEY" \
    --datasets vqav2 \
    --limit 3 \
    --debug 2>&1 | tail -20

echo ""
echo "=== Testing MSR-VTT (3 samples) ==="
$PYTHON -m evalscope.run \
    --model qwen-vl-plus \
    --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key "$DASHSCOPE_API_KEY" \
    --datasets msr_vtt \
    --limit 3 \
    --debug 2>&1 | tail -20

echo ""
echo "=== Testing MSVD (3 samples) ==="
$PYTHON -m evalscope.run \
    --model qwen-vl-plus \
    --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key "$DASHSCOPE_API_KEY" \
    --datasets msvd \
    --limit 3 \
    --debug 2>&1 | tail -20
