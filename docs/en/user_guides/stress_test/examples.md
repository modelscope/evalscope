# Examples

## Open-loop arrivals

```bash
evalscope perf --model qwen-plus --base-url https://example.com/v1 \
  --api-key "$API_KEY" --workload prompt --prompt 'Hello' \
  --mode open_loop --request-rate 20 --requests 1000 \
  --max-outstanding 256 --overflow-policy record_drop --arrival poisson
```

Dropped arrivals are recorded as observations; the scheduler never waits for a free slot and silently changes open-loop traffic into closed-loop traffic.

## Local JSONL prompts

```bash
evalscope perf --model qwen-plus --base-url https://example.com/v1 \
  --workload line_by_line --workload-path ./requests.jsonl --data-source local \
  --concurrency 4 --requests 100
```

## Multiple load points

Use repeatable `--load` JSON or construct multiple typed load objects in `BenchmarkSuite.loads`.

## Local targets

Set `--target-kind local_transformers` or `--target-kind local_vllm`. EvalScope owns startup, readiness checks, logs, and shutdown for the local process.
