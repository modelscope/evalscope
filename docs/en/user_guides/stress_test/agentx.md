# AgentX Serving Benchmark

AgentX measures how an inference service handles long coding-agent sessions. It reports serving performance, not model quality or task accuracy.

## Install

AgentX requires Python 3.11--3.13 and is intentionally separate from the default Perf extra:

```bash
pip install 'evalscope[agentx]'
```

## Service prerequisites

The default 256K workload needs a streaming OpenAI Chat Completions endpoint with at least 256K tokens of context. A 32K service cannot run the default traces. For vLLM, set `--max-model-len 262144`, then confirm `max_model_len` in `/v1/models`.

## Quick start

Run a short smoke test first. It uses one active session and is not comparable with published results:

```bash
evalscope perf \
  --scenario '{"name":"agentx","mode":"smoke"}' \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 1
```

## Standard run

The default run uses the verified ModelScope 256K mirror, a fixed seed, and a 30-minute duration:

```bash
evalscope perf \
  --scenario agentx \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 8
```

`parallel` is the number of active AgentX sessions. A session can issue more than one request, so actual request concurrency may be higher.

Use `--data-source huggingface` only when you need an upstream-compatible result:

```bash
evalscope perf --scenario agentx --data-source huggingface \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

## Advanced configuration

Use JSON only to select a different workload variant or record deployment metadata. The `full` variant requires a service that supports its larger context window:

```bash
evalscope perf \
  --scenario '{"name":"agentx","variant":"full","num_gpus":8,"engine":"vllm"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

## Results

Each concurrency point writes an EvalScope summary in `agentx_summary.json` and a combined `agentx_sweep_summary.json`. Raw benchmark files are kept in the adjacent `aiperf/` directory for troubleshooting.

Smoke, cancelled, failed, shortened, and hash-mismatched runs have `submission_valid=false`. Use a standard run with the required dataset and duration when you need a comparable result.

## Compatibility

This integration pins `aiperf==0.12.0` and its `inferencex-agentx-mvp` scenario. It is not equivalent to the newer InferenceX AgentX v1.0 leaderboard methodology, which has different profiling and warmup rules. Do not use AgentX metrics as model or agent-task quality scores.
