# AgentX Serving Benchmark

AgentX measures how an inference service handles long coding-agent sessions. It reports serving performance, not model quality or task accuracy.

`agentx` retains the legacy MVP methodology. The [AgentX Harness scenarios](#agentx-harness) use a separate runner and do not change existing commands or defaults.

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

## Request timeout

Set `request_timeout_seconds` in the scenario JSON to override AIPerf's per-request timeout. The value must be greater than zero; omit it to keep AIPerf's default.

```bash
evalscope perf \
  --scenario '{"name":"agentx","mode":"smoke","request_timeout_seconds":300}' \
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

## Waiting for in-flight requests

Set `benchmark_grace_period` in the scenario JSON to control how many additional seconds AIPerf waits for outstanding responses after the sending duration. For example, replace `--scenario agentx` with:

```bash
--scenario '{"name":"agentx","benchmark_grace_period":600}'
```

The value must be finite and nonnegative; zero is forwarded explicitly. Omit it to retain AIPerf's default (30 seconds in 0.12.0). This does not change `--duration` or the per-request `request_timeout_seconds`. Responses completed during this period are included in the metrics, and the phase elapsed time can exceed the sending duration. Use the same setting when comparing engines and check the phase logs for cancelled requests.

## Advanced configuration

Use JSON to select a different workload variant, configure request timing, or record deployment metadata. The `full` variant requires a service that supports its larger context window:

```bash
evalscope perf \
  --scenario '{"name":"agentx","variant":"full","num_gpus":8,"engine":"vllm"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

## Results

Each concurrency point writes an EvalScope summary in `agentx_summary.json` and a combined `agentx_sweep_summary.json`. Raw benchmark files are kept in the adjacent `aiperf/` directory for troubleshooting.

Smoke, cancelled, failed, shortened, and hash-mismatched runs have `submission_valid=false`. Use a standard run with the required dataset and duration when you need a comparable result.

## MVP compatibility

The legacy `agentx` integration pins `aiperf==0.12.0` and its `inferencex-agentx-mvp` scenario. It is not equivalent to the newer InferenceX AgentX v1.0 leaderboard methodology, which has different profiling and warmup rules. Do not use AgentX metrics as model or agent-task quality scores.

## AgentX Harness

The new scenarios pin `agentx-harness==1.0.13` and both invoke its native `agentx` scenario:

| Scenario | Corpus | Default benchmark duration | Default seed |
| --- | --- | --- | --- |
| `agentx_harness` | Full Weka corpus | 3600 seconds | 42 |
| `agentx_harness_256k` | 256K Weka subset | 3600 seconds | 42 |

Use POSIX and Python 3.11--3.13. **Never install `evalscope[agentx]` and `evalscope[agentx_harness]` in the same environment**: both runners own the `aiperf` Python package. Mixed environments are rejected; rebuild a clean environment rather than uninstalling one overlapping package in place.

```bash
python3 -m venv .venv/agentx_harness
.venv/agentx_harness/bin/python -m pip install -e '.[agentx_harness]'

# Smoke check: four traces, 60-second sending duration, not a valid submission.
.venv/agentx_harness/bin/evalscope perf \
  --scenario '{"name":"agentx_harness_256k","mode":"smoke"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 1

# Benchmark: use agentx_harness instead for the full corpus.
.venv/agentx_harness/bin/evalscope perf \
  --scenario agentx_harness_256k \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

The editable install above is for a repository checkout. For a released package, install `evalscope[agentx_harness]` into a fresh environment. Keep an existing MVP environment separate.

The 256K scenario needs a server with at least a 256K context window; the full corpus may require a larger window. Both require streaming OpenAI Chat Completions and server support for `ignore_eos=true`. `--parallel` counts active session trees, not individual requests. The harness owns warmup and drain, so elapsed wall time can exceed `--duration`. Benchmark duration must be finite and at least 900 seconds. Smoke defaults to 60 seconds and four traces; `trace_limit` is allowed only in smoke mode.

JSON accepts the same timing/tokenizer/deployment options described above. Remote tokenizer code remains **opt-in** via `tokenizer_trust_remote_code`; pin `tokenizer_revision` when enabling it. A `variant` contradicting the selected name is rejected. `--number`, open-loop/rate control, multi-turn, SLA tuning, Perf warmup, and custom dataset arguments are not supported.

Python uses the same typed `Arguments` interface:

```python
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

results = run_perf_benchmark(Arguments(
    scenario={'name': 'agentx_harness_256k', 'mode': 'smoke'},
    model='YOUR_MODEL',
    tokenizer_path='YOUR_TOKENIZER_PATH_OR_ID',
    url='http://localhost:8000/v1/chat/completions',
    parallel=[1],
))
```

The harness defaults to the pinned upstream public dataset. `--data-source modelscope` or `--dataset-path` selects a verified mirror/local corpus; native public aliases are checked against their actual current repository head and expected checksum. Local JSONL directories go directly to the harness loader, without MVP's per-trace JSON conversion. Benchmark hash mismatches are rejected; unverified data is permitted only for smoke.

Each concurrency point writes `<scenario>/parallel_<n>/agentx_summary.json` (schema `2.0`), raw artifacts under `harness/`, and `harness.log`; each scenario directory also contains `agentx_sweep_summary.json`. Summaries include the pinned harness version, native engine version, dataset provenance, native invalid reasons, and metrics. Failed/cancelled runs retain diagnostics and stop the sweep. Reusing an existing native artifact directory is rejected to prevent stale results.

Native submission validity is authoritative: mirrors and smoke use `unsafe_override` and are never promoted to valid submissions. Do not pool harness results with MVP numbers. This integration supports CLI/Python; dashboard-specific result tables and archives are not provided.

## Custom tokenizers

For tokenizers that require custom Python code, explicitly set `tokenizer_trust_remote_code` to `true` in the scenario JSON. EvalScope forwards this option to AIPerf as `--tokenizer-trust-remote-code`. It defaults to `false`; enable it only for tokenizer code you have reviewed and trust. Set `tokenizer_revision` to a HuggingFace tag or commit hash to pin the tokenizer and its remote code for reproducible runs.

```bash
evalscope perf \
  --scenario '{"name":"agentx","mode":"smoke","tokenizer_trust_remote_code":true,"tokenizer_revision":"<commit-or-tag>"}' \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 1
```
