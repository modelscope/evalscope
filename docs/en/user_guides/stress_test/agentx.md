# AgentX MVP Serving Benchmark

AgentX is a serving-system workload, not an agent-quality evaluation. It replays long-context coding-agent session trees, including shared prefixes, think time, and subagent dependencies through AIPerf.

## Install

AgentX requires Python 3.11--3.13 and is intentionally separate from the default Perf extra:

```bash
pip install 'evalscope[agentx]'
```

## Run

The shorthand uses the verified ModelScope 256K mirror, a fixed seed (`20260707`), and AIPerf's AgentX MVP default duration of 1800 seconds:

```bash
evalscope perf \
  --scenario agentx \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_HF_TOKENIZER \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 8 16
```

Use Hugging Face explicitly when an AIPerf upstream submission-valid result is required:

```bash
evalscope perf --scenario agentx --data-source huggingface \
  --model YOUR_MODEL --tokenizer-path YOUR_HF_TOKENIZER \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

The JSON form exposes AgentX-specific metadata without duplicating normal Perf connection settings:

```bash
evalscope perf \
  --scenario '{"name":"agentx","variant":"full","max_context_length":1000000,"num_gpus":8,"engine":"vllm"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_HF_TOKENIZER \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

`parallel` means active agent session trees, not a cap on individual HTTP requests. Subagents can make actual in-flight request concurrency higher.

## Smoke runs and artifacts

Use `mode=smoke` for a short, intentionally non-comparable run. It defaults to four traces and 60 seconds:

```bash
evalscope perf \
  --scenario '{"name":"agentx","mode":"smoke"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_HF_TOKENIZER \
  --url http://localhost:8000/v1/chat/completions --parallel 1
```

Each point stores raw, unchanged AIPerf files under `agentx_<variant>/parallel_<N>/aiperf/`, its EvalScope summary in `agentx_summary.json`, and a root `agentx_sweep_summary.json`.

`submission_valid` in the EvalScope summary is false for smoke, cancelled, failed, shortened, or hash-mismatched runs. For the byte-verified ModelScope mirror, EvalScope can revalidate a completed canonical run where AIPerf's only invalid reason is the local-mirror `unsafe_override`; the original AIPerf validity fields are always retained separately.

## Compatibility

This integration pins `aiperf==0.12.0` and its `inferencex-agentx-mvp` scenario. It is not equivalent to the newer InferenceX AgentX v1.0 leaderboard methodology, which has different profiling and warmup rules. Do not use AgentX metrics as model or agent-task quality scores.
