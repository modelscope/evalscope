---
name: evalscope
description: >-
  Translates natural language requests into evalscope CLI commands.
  Core capabilities: (1) Model accuracy evaluation (eval) — runs 156+
  benchmarks (Math, Coding, Chinese, Multimodal, Agent, etc.) against
  local checkpoints or OpenAI-compatible / Anthropic API endpoints;
  (2) Performance stress testing (perf) — measures TTFT, TPOT,
  throughput, and latency under configurable concurrency gradients or
  SLA auto-tuning; (3) Benchmark discovery — lists and filters
  benchmarks by capability tag, retrieves full metadata and sample
  examples; (4) Result visualization — launches a Gradio web UI to
  compare and explore evaluation outputs. Trigger this skill whenever
  the user mentions: evaluate / benchmark / score a model, throughput /
  latency / QPS / stress test, find benchmarks by tag or capability, or
  view / compare evaluation results.
---

# EvalScope

EvalScope is an LLM evaluation framework supporting 156+ benchmarks, performance stress testing, and result visualization. This skill converts natural language evaluation requests into `evalscope` CLI commands.

No source code access is required. All operations are performed through the `evalscope` CLI.

## Prerequisites

Before running any evalscope command, check that it is installed:

```bash
evalscope --version
```

If not installed, install it:

```bash
# Basic installation
pip install evalscope

# Install all major backends (opencompass, vlmeval, rag, perf, app, aigc, service)
# Note: does NOT include sandbox (requires Docker) or dev/docs extras
pip install 'evalscope[all]'

# Selective extras
pip install 'evalscope[perf]'        # Performance benchmarking only
pip install 'evalscope[app]'         # Visualization UI only
pip install 'evalscope[rag]'         # RAG evaluation backend
pip install 'evalscope[sandbox]'     # Sandbox code execution (requires Docker)
```

## Decision Tree

Follow this tree to determine the correct workflow:

```
User Request
|
+-- "evaluate / benchmark / test accuracy / score" --> Eval Workflow
+-- "throughput / latency / QPS / stress test / perf" --> Perf Workflow
+-- "view results / visualize / compare" --> Visualization Workflow
+-- "what benchmarks / list benchmarks / tell me about X" --> Benchmark Discovery Workflow
+-- "filter by tag / math benchmarks / coding benchmarks" --> Benchmark Discovery (--tag)
|
+-- Model source (for eval):
|   +-- Local model path or HuggingFace/ModelScope ID --> checkpoint mode
|   +-- API endpoint (URL + key) --> openai_api mode
|   +-- Just testing the pipeline --> mock_llm mode
|
+-- What to evaluate:
|   +-- User named specific datasets --> use directly with --datasets
|   +-- User described a capability (e.g. "math", "coding") --> use Quick Lookup Table below
|   +-- User wants to browse --> run: evalscope benchmark-info --list
|
+-- Additional configuration:
    +-- Limit samples --> --limit N
    +-- Output directory --> --work-dir PATH
    +-- Need full parameter list --> evalscope eval --help / evalscope perf --help
    +-- Complex configuration --> see eval-reference.md or perf-reference.md
```

## Workflow 1: Model Evaluation (eval)

Use `evalscope eval` to evaluate model capabilities on benchmarks.

### Step 1: Confirm Model Source

Ask the user how the model is served:

**Local checkpoint** (downloaded or local path):
```bash
evalscope eval --model Qwen/Qwen2.5-0.5B-Instruct --datasets gsm8k --limit 10
```

**API service** (OpenAI-compatible endpoint):
```bash
evalscope eval \
  --model qwen-plus \
  --datasets gsm8k arc \
  --api-url http://localhost:8000/v1/chat/completions \
  --api-key sk-xxx \
  --limit 10
```
When `--api-url` is provided, eval-type is automatically set to `openai_api`.

**Anthropic API**:
```bash
evalscope eval \
  --model claude-3-5-sonnet \
  --eval-type anthropic_api \
  --datasets mmlu \
  --api-key sk-ant-xxx
```

**Mock mode** (test pipeline without real model):
```bash
evalscope eval --model mock --eval-type mock_llm --datasets gsm8k --limit 5
```

### Step 2: Select Benchmarks

If the user specified dataset names, use them directly with `--datasets`.
If the user described a capability, use the Quick Lookup Table in the Benchmark Discovery section.
For detailed benchmark info:
```bash
evalscope benchmark-info <name> --format markdown
```

### Step 3: Configure Parameters

Defaults work for most cases. Suggest `--limit 10` for first-run validation.

Key optional parameters:
- `--generation-config '{"temperature": 0.7, "max_tokens": 2048}'` - generation settings
- `--dataset-args '{"gsm8k": {"few_shot_num": 4}}'` - per-dataset config
- `--eval-backend Native|OpenCompass|VLMEvalKit|RAGEval` - evaluation backend
- `--eval-batch-size N` - batch size (default 1)
- `--work-dir PATH` - output directory (default `./outputs`)
- `--seed 42` - random seed

For the full parameter reference, see [eval-reference.md](eval-reference.md).

### Step 4: Build and Execute

1. Show the full command to the user for confirmation
2. Execute the command
   - For quick tests (`--limit` <= 20): run in foreground
   - For full evaluations: run in background (may take hours)
3. Output goes to `./outputs/<timestamp>/`

### Step 5: Interpret Results

After evaluation completes:
1. Check reports: `ls outputs/<timestamp>/reports/`
2. Read JSON reports: each file contains `score`, `metrics`, `dataset_name`, `model_name`
3. Summarize: extract the `score` and `metrics[].score` fields to present results
4. For HTML reports: check `outputs/<timestamp>/reports/report.html`

Output directory structure:
```
outputs/<timestamp>/
  configs/task_config.yaml    # Full run configuration
  logs/eval_log.log           # Evaluation log
  predictions/<model>/*.jsonl # Model predictions
  reviews/<model>/*.jsonl     # Review/judge results
  reports/<model>/*.json      # Score reports
  reports/report.html         # Summary HTML report
```

## Workflow 2: Performance Benchmark (perf)

Use `evalscope perf` to stress-test model API endpoints.

### Step 1: Confirm Endpoint

Required: `--model` (model name) and `--url` (API endpoint).

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai
```

### Step 2: Choose Test Scenario

**Simple throughput test**:
```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 1 \
  --number 100 \
  --stream
```

**Concurrency gradient test** (multiple parallel values):
```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 1 5 10 20 \
  --number 100 500 1000 2000 \
  --stream
```
Note: `--parallel` and `--number` must have the same number of values (paired).

**Precise input/output length test**:
```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset random \
  --min-prompt-length 1000 \
  --max-prompt-length 1000 \
  --max-tokens 512 \
  --min-tokens 512 \
  --parallel 5 \
  --number 100
```

**SLA auto-tuning** (find max concurrency within SLA constraints):
```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"name": "latency", "operator": "<=", "value": 5.0}]'
```

**Local model test** (auto-starts local inference server):
```bash
evalscope perf \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --api local \
  --dataset openqa \
  --parallel 1 \
  --number 10
```

### Step 3: Key Parameters

- `--api`: API protocol - `openai` (default), `local`, `local_vllm`, `dashscope`, `custom`
- `--stream` / `--no-stream`: Enable/disable streaming (default: stream)
- `--max-tokens N`: Max output tokens (default: 2048)
- `--temperature F`: Sampling temperature (default: 0.0)
- `--dataset`: Data source - `openqa`, `random`, `speed_benchmark`, `random_vl`, `line_by_line`, etc.
- `--tokenizer-path PATH`: For accurate token counting

For the full parameter reference, see [perf-reference.md](perf-reference.md).

### Step 4: Execute and Read Results

Execute the command. Key output metrics:

| Metric | Description |
|--------|-------------|
| TTFT | Time to First Token (seconds) |
| TPOT | Time Per Output Token (seconds) |
| Latency | Average request latency (seconds) |
| Request Throughput | Requests per second |
| Output Token Throughput | Output tokens per second |
| Total Token Throughput | Total tokens per second |

Results include percentile statistics: p50, p75, p90, p95, p99.

An HTML report is auto-generated in `outputs/`. Read the console output for the report path.

## Workflow 3: Visualization

### Launch Gradio Web UI

```bash
evalscope app --outputs ./outputs --lang zh --server-port 7860
```

Options:
- `--outputs PATH` - directory containing evaluation results (default: `./outputs`)
- `--lang zh|en` - interface language
- `--server-port N` - port number
- `--server-name ADDR` - bind address (default: `0.0.0.0`)
- `--share` - generate a public Gradio link

Requires: `pip install 'evalscope[app]'`

### View HTML Reports Directly

Perf and eval both generate HTML reports in the output directory:
- Eval: `outputs/<timestamp>/reports/report.html`
- Perf: check console output for the exact report path

## Workflow 4: Benchmark Discovery (benchmark-info)

### List All Benchmarks

```bash
evalscope benchmark-info --list
```

Output: a table of all benchmarks with columns: Name, Pretty Name, Category, Tags, Metrics count, Subsets count.

### Filter Benchmarks by Tag

When the user asks about a specific capability area (e.g. "what math benchmarks are there?"), use `--tag` to filter:

```bash
# Single tag
evalscope benchmark-info --list --tag Math

# Multiple tags (OR logic, case-insensitive)
evalscope benchmark-info --list --tag Coding Agent

# Other useful tag filters
evalscope benchmark-info --list --tag Chinese
evalscope benchmark-info --list --tag MultiModal
evalscope benchmark-info --list --tag LongContext
evalscope benchmark-info --list --tag Medical
```

Available tags correspond to the Tags column in the Quick Lookup Table below.

### Get Benchmark Details (Text, Default)

```bash
evalscope benchmark-info gsm8k
```

Prints a detailed text summary including: name, dataset ID, category, tags, output types, few-shot count, aggregation method, train/eval splits, subsets, paper URL, metrics, description, prompt template, system prompt, configurable parameters (with types, defaults, choices), and data statistics (total samples, prompt length mean/min/max).

### Get Benchmark Details (JSON)

```bash
evalscope benchmark-info gsm8k --format json
```

Returns the complete benchmark metadata as structured JSON. Key fields:

| Field | Description |
|-------|-------------|
| `meta.pretty_name` | Display name |
| `meta.dataset_id` | Dataset source ID (ModelScope/HuggingFace) |
| `meta.tags` | Capability tags (Math, Reasoning, Coding, etc.) |
| `meta.metrics` | Evaluation metrics with config (e.g. acc, f1) |
| `meta.few_shot_num` | Default few-shot count |
| `meta.eval_split` | Dataset split used for evaluation |
| `meta.subset_list` | Available subsets |
| `meta.category` | Category: `llm`, `vlm`, `agent`, `aigc` |
| `meta.description` | Benchmark description (markdown) |
| `meta.prompt_template` | Prompt template used for evaluation |
| `meta.system_prompt` | System prompt (if any) |
| `meta.extra_params` | Configurable parameters with types and defaults |
| `meta.paper_url` | Link to the benchmark paper |
| `statistics.total_samples` | Total number of evaluation samples |
| `statistics.subset_stats` | Per-subset sample counts and prompt length stats |
| `statistics.prompt_length` | Prompt length statistics (mean, min, max, std) |
| `sample_example` | A sample data example from the benchmark |
| `readme.en` | Full English markdown documentation |
| `readme.zh` | Full Chinese markdown documentation |

### Get Benchmark Details (Markdown)

```bash
evalscope benchmark-info gsm8k --format markdown
```

Returns a full markdown document with: overview, task description, key features, evaluation notes, properties table (metrics, default shots, splits), data statistics, and a sample example.

If `--format markdown` encounters an error, fall back to `--format json` and read the `readme.en` (English) or `readme.zh` (Chinese) field from the JSON output.

### Query Multiple Benchmarks

```bash
evalscope benchmark-info gsm8k mmlu ceval --format json
```

### Quick Lookup Table

Use this table for fast benchmark recommendations without running CLI commands:

| User Need | Tags | Typical Benchmarks |
|-----------|------|--------------------|
| Math / reasoning | Math, Reasoning | gsm8k, math_500, aime24, competition_math |
| Coding / programming | Coding | humaneval, mbpp, live_code_bench |
| General knowledge | Knowledge, MCQ | mmlu, ceval, cmmlu, mmlu_pro |
| Chinese language | Chinese | ceval, cmmlu, chinese_simpleqa |
| Multimodal / vision | MultiModal | mmmu, mm_bench, math_vista, mm_star |
| Instruction following | InstructionFollowing | ifeval, multi_if |
| Function calling / tools | FunctionCalling | bfcl_v3, bfcl_v4, tool_bench |
| Long context | LongContext | needle_haystack, longbench_v2, cl_bench |
| Arena / model ranking | Arena | arena_hard, alpaca_eval |
| Commonsense | Commonsense | hellaswag, winogrande |
| Hallucination | Hallucination | truthfulqa |
| Medical | Medical | medqa, medmcqa |
| Agent | Agent | tau_bench |
| Audio / speech | Audio, SpeechRecognition | asr benchmarks |

### Common Evaluation Suites

Pre-built combinations for common evaluation needs:

- **General LLM**: `mmlu gsm8k bbh humaneval ifeval`
- **Chinese capability**: `ceval cmmlu chinese_simpleqa`
- **Math reasoning**: `gsm8k math_500 aime24 competition_math`
- **Multimodal (VLM)**: `mmmu mm_bench math_vista mm_star`
- **Code generation**: `humaneval mbpp live_code_bench`

## General Notes

- Always run `evalscope --version` before first use to verify installation
- Default output directory: `./outputs/<timestamp>/`
- For first-run validation, always suggest `--limit 5` or `--limit 10`
- For API-based eval, confirm the service endpoint is reachable before starting
- For local checkpoint eval, verify GPU availability and sufficient memory
- Long-running evaluations: run in background, monitor with `tail -f outputs/<timestamp>/logs/eval_log.log`
- On failure: read the log file for error details
- To discover all available parameters: `evalscope eval --help` or `evalscope perf --help`
- The `--no-timestamp` flag prevents creating timestamped subdirectories (useful for CI)
- Use `--use-cache PATH` to reuse cached results from a previous run
- Use `--debug` for verbose output during troubleshooting
