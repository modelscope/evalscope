# Troubleshooting

Common error patterns and recovery steps.

## Connection Refused / Timeout

**Symptom**: `ConnectionRefusedError` or request hangs until timeout.

**Steps**:
1. Verify endpoint reachable: `curl -s http://localhost:8000/v1/models`
2. Check URL path — must include `/v1/chat/completions` (not just base URL)
3. For perf: use `--no-test-connection` to skip pre-flight check if the server is slow to start

## HTTP 4xx Error (401 / 404 / 422)

**Symptom**: `Non-retryable error (HTTP 4xx)` — will not retry automatically.

**Steps**:
1. **401**: Check `--api-key` value
2. **404**: Model name doesn't match server — verify with `curl .../v1/models`
3. **422**: Request format mismatch — check `--api` type matches the server (openai vs dashscope)

## Evaluation Interrupted Mid-Run

**Symptom**: Process killed, partial predictions exist in `outputs/<timestamp>/predictions/`.

**Recovery**:
```bash
evalscope eval --model qwen-plus --datasets mmlu gsm8k \
  --api-url http://localhost:8000/v1/chat/completions \
  --use-cache outputs/20250601_120000
```

`--use-cache` reuses existing predictions (matched by sample_id) and only runs missing items. Add `--rerun-review` to force re-scoring while keeping prediction cache.

## CUDA Out of Memory

**Symptom**: `torch.cuda.OutOfMemoryError` during local checkpoint evaluation.

**Steps**:
1. Reduce batch size: `--eval-batch-size 1`
2. Use a smaller model or quantized variant
3. Switch to API mode: deploy with vLLM then evaluate via `--api-url`

## Benchmark Not Found

**Symptom**: `KeyError: '<name>' not found in registry` or `No module named`.

**Steps**:
1. List available: `evalscope benchmark-info --list`
2. Check spelling — names are case-sensitive and use underscores (e.g. `gsm8k`, `mm_bench`)
3. Some benchmarks require extra install: `pip install 'evalscope[vlmeval]'` for multimodal

## Perf Results Anomalous

**Symptom**: TTFT shows 0, all requests fail, or throughput is unexpectedly low.

**Steps**:
1. TTFT=0 usually means `--no-stream` was used — TTFT only measures streaming first token
2. All failures: check `--debug` output for response bodies
3. Low throughput: verify `--parallel` is > 1, check if server is the bottleneck
4. For precise measurements: use `--tokenizer-path` for accurate token counting
