---
name: evalscope
description: >-
  LLM evaluation & inference performance testing via the evalscope CLI.
  Translates natural language requests into evalscope commands for:
  (1) Model accuracy evaluation — runs 160+ benchmarks against local
  checkpoints or API endpoints (OpenAI-compatible, Anthropic, LiteLLM);
  (2) Performance stress testing — TTFT, TPOT, throughput, latency under
  configurable concurrency; (3) RAG evaluation — RAGAS quality metrics,
  MTEB embedding benchmarks, CLIP retrieval; (4) Benchmark discovery —
  list/filter/inspect benchmarks by tag. Trigger on: evaluate / benchmark /
  score a model, throughput / latency / QPS / stress test, find benchmarks,
  view results, 评测模型, 压测, 跑 benchmark, 性能测试, 查看评测结果,
  有哪些评测集, RAG 评测, embedding 评测. Do NOT trigger for: model
  training / finetuning / deployment / serving requests.
---

# EvalScope

Read only the relevant reference file for the matched workflow — don't preload all of them.

| Workflow | When | Reference |
|----------|------|-----------|
| Eval (accuracy) | evaluate / benchmark / score | [eval-reference.md](eval-reference.md) |
| Perf (stress test) | throughput / latency / QPS / perf | [perf-reference.md](perf-reference.md) |
| RAG Evaluation | RAG / embedding / retrieval quality | [rag-reference.md](rag-reference.md) |
| Visualization | view results / compare / dashboard | (below) |
| Benchmark Discovery | list / find / what benchmarks | (below) |
| Troubleshooting | errors / failures / debug | [troubleshooting.md](troubleshooting.md) |

## Prerequisites

```bash
evalscope --version            # verify installation
pip install evalscope           # basic
pip install 'evalscope[all]'   # all backends (perf, rag, service, aigc)
pip install 'evalscope[perf]'  # perf only
pip install 'evalscope[rag]'   # RAG only (RAGAS, MTEB, CLIP)
pip install 'evalscope[service]' # Web dashboard
```

## Decision Tree

- User wants **accuracy evaluation** (evaluate / benchmark / score / 评测)
  - Local checkpoint path or HuggingFace/ModelScope ID → `--model PATH` (auto `llm_ckpt`)
  - API endpoint → `--model NAME --api-url URL` (auto `openai_api`)
  - Anthropic → `--eval-type anthropic_api`
  - LiteLLM multi-provider → `--eval-type litellm --model provider/name`
  - OpenAI Responses API → `--eval-type openai_responses_api`
  - Pipeline test → `--model mock --eval-type mock_llm`
  - Image generation → `--eval-type text2image`
  - TTS → `--eval-type text2speech`
  - Image editing → `--eval-type image_editing`
- User wants **performance test** (throughput / latency / QPS / 压测)
  - → `evalscope perf` workflow
  - API types: `openai` (default), `local`, `local_vllm`, `dashscope`, `embedding`, `rerank`, `custom`
- User wants **RAG evaluation** (RAG / embedding quality / retrieval)
  - → `evalscope eval --eval-backend RAGEval` with tool config
- User wants **visualization** (view / compare / dashboard)
  - → `evalscope service`
- User wants **benchmark info** (list / find / what benchmarks / 有哪些评测集)
  - → `evalscope benchmark-info`

## Workflow 1: Eval (Accuracy)

Core command pattern:
```bash
# Local checkpoint
evalscope eval --model Qwen/Qwen2.5-0.5B-Instruct --datasets gsm8k --limit 10

# API endpoint (auto-detects openai_api when --api-url is set)
evalscope eval --model qwen-plus --datasets gsm8k arc \
  --api-url http://localhost:8000/v1/chat/completions --api-key sk-xxx --limit 10

# Anthropic
evalscope eval --model claude-3-5-sonnet --eval-type anthropic_api --datasets mmlu --api-key sk-ant-xxx
```

Key parameters: `--datasets`, `--limit`, `--generation-config`, `--dataset-args`, `--eval-backend`, `--judge-strategy`. For full parameter list → [eval-reference.md](eval-reference.md).

Output: `outputs/<timestamp>/reports/*.json` (scores), `report.html` (summary).

## Workflow 2: Perf (Stress Test)

Core command pattern:
```bash
# Basic throughput test
evalscope perf --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions --api openai \
  --dataset openqa --parallel 5 --number 200 --stream

# Concurrency gradient (--parallel and --number must pair)
evalscope perf --model qwen-plus --url http://localhost:8000/v1/chat/completions \
  --api openai --parallel 1 5 10 20 --number 50 250 500 1000 --stream

# Embedding model
evalscope perf --model text-embedding-v3 --url http://localhost:8000/v1/embeddings \
  --api embedding --parallel 10 --number 500

# Rerank model
evalscope perf --model bge-reranker --url http://localhost:8000/v1/rerank \
  --api rerank --parallel 5 --number 200
```

Key parameters: `--parallel`, `--number`, `--dataset`, `--max-tokens`, `--sla-auto-tune`. For full parameter list → [perf-reference.md](perf-reference.md).

Output: console table (TTFT/TPOT/throughput p50-p99) + HTML report.

## Workflow 3: RAG Evaluation

Uses `--eval-backend RAGEval` with a Python dict/YAML config. Three tools: RAGAS, MTEB, clip_benchmark.

```python
from evalscope import run_task
run_task({
    'eval_backend': 'RAGEval',
    'eval_config': {
        'tool': 'MTEB',  # or 'RAGAS' or 'clip_benchmark'
        ...
    }
})
```

For config schemas and examples → [rag-reference.md](rag-reference.md).

## Workflow 4: Visualization

```bash
evalscope service --host 0.0.0.0 --port 9000 --outputs ./outputs
```

Options: `--host` (default 0.0.0.0), `--port` (default 9000), `--outputs PATH` (scan dir), `--debug`.

Requires: `pip install 'evalscope[service]'`.

## Workflow 5: Benchmark Discovery

```bash
evalscope benchmark-info --list                    # all benchmarks
evalscope benchmark-info --list --tag Math Coding  # filter by tags (OR, case-insensitive)
evalscope benchmark-info gsm8k                     # text summary
evalscope benchmark-info gsm8k --format json       # structured JSON
evalscope benchmark-info gsm8k --format markdown   # full docs
```

## Workflow 6: Sandbox Evaluation

For code-execution benchmarks (HumanEval, MBPP, etc.) with Docker isolation:

```bash
evalscope eval --model qwen-plus --datasets humaneval \
  --api-url http://localhost:8000/v1/chat/completions \
  --sandbox '{"enabled": true, "type": "docker"}'
```

Requires Docker daemon running. See `evalscope eval --help` for `--sandbox` schema.

## Quick Lookup Table

For up-to-date results: `evalscope benchmark-info --list --tag <TAG>`

| User Need | Tags | Typical Benchmarks |
|-----------|------|--------------------|
| Math / reasoning | Math, Reasoning | gsm8k, math_500, aime24, competition_math |
| Coding | Coding | humaneval, mbpp, live_code_bench |
| General knowledge | Knowledge, MCQ | mmlu, ceval, cmmlu, mmlu_pro |
| Chinese | Chinese | ceval, cmmlu, chinese_simpleqa |
| Multimodal / vision | MultiModal | mmmu, mm_bench, math_vista |
| Instruction following | InstructionFollowing | ifeval, multi_if |
| Function calling | FunctionCalling | bfcl_v3, bfcl_v4 |
| Long context | LongContext | needle_haystack, longbench_v2 |
| Agent | Agent | tau_bench |

Common suites:
- **General LLM**: `mmlu gsm8k bbh humaneval ifeval`
- **Chinese**: `ceval cmmlu chinese_simpleqa`
- **Multimodal**: `mmmu mm_bench math_vista mm_star`

## General Notes

- Always `--limit 5` for first-run validation
- Default output: `./outputs/<timestamp>/`
- Long runs: background + `tail -f outputs/<timestamp>/logs/eval_log.log`
- Resume interrupted runs: `--use-cache outputs/<previous_timestamp>`
- Full parameter help: `evalscope eval --help` / `evalscope perf --help`
- On errors → [troubleshooting.md](troubleshooting.md)
