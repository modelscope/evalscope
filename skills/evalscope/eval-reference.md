# Eval Parameter Reference

Complete parameter reference for `evalscope eval`. All parameters can also be discovered via `evalscope eval --help`.

## Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | (required) | Model ID on ModelScope/HuggingFace, or local model directory path |
| `--model-id` | str | None | Custom model name for display in reports |
| `--model-args` | str | None | Model init args as JSON or `key=value,key=value` format |
| `--model-task` | str | `text_generation` | Task type: `text_generation` or `image_generation` |
| `--chat-template` | str | None | Custom Jinja2 chat template for prompt formatting |

## Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--datasets` | str (multiple) | (required) | Space-separated list of dataset names (e.g. `gsm8k arc mmlu`) |
| `--dataset-args` | JSON str | `{}` | Per-dataset configuration as JSON |
| `--dataset-dir` | str | None | Local dataset cache directory |
| `--dataset-hub` | str | None | Dataset source: `modelscope` or `huggingface` |

### Dataset Args Format

`--dataset-args` accepts a JSON string with per-dataset configuration:

```bash
--dataset-args '{
  "gsm8k": {"few_shot_num": 4, "few_shot_random": false},
  "arc": {"subset_list": ["ARC-Challenge"]},
  "ifeval": {"filters": {"remove_until": "</think>"}}
}'
```

Common per-dataset options:
- `few_shot_num`: Number of few-shot examples (overrides benchmark default)
- `few_shot_random`: Whether to randomly select few-shot examples (default: false)
- `subset_list`: List of subsets to evaluate (default: all subsets)
- `filters`: Output filters (e.g. remove thinking tokens)
- `dataset_id`: Override the default dataset source path

## Evaluation Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval-type` | str | auto | Evaluation type (see below) |
| `--eval-backend` | str | `Native` | Backend: `Native`, `OpenCompass`, `VLMEvalKit`, `RAGEval` |
| `--eval-config` | str | None | Config file path for non-Native backends |
| `--eval-batch-size` | int | 1 | Batch size for evaluation |
| `--limit` | float | None | Max samples per subset (int for count, float < 1 for fraction) |
| `--repeats` | int | 1 | Repeat dataset N times (for k-metrics calculation) |

### Eval Types

| Value | When to Use |
|-------|-------------|
| `llm_ckpt` (checkpoint) | Local model weights (auto-detected when --model is a path) |
| `openai_api` | OpenAI-compatible API endpoint (auto-detected when --api-url is set) |
| `anthropic_api` | Anthropic Claude API |
| `mock_llm` | Pipeline testing without a real model |
| `text2image` | Image generation models |

### Eval Backends

| Backend | Use Case | Extra Install |
|---------|----------|---------------|
| `Native` | Default, built-in evaluation | None |
| `OpenCompass` | Use OpenCompass evaluation suite | `pip install 'evalscope[opencompass]'` |
| `VLMEvalKit` | Multimodal model evaluation | `pip install 'evalscope[vlmeval]'` |
| `RAGEval` | RAG pipeline evaluation (RAGAS, MTEB) | `pip install 'evalscope[rag]'` |

## Generation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--generation-config` | str | None | Generation params as JSON or `key=value` format |

Accepts JSON or comma-separated key=value pairs:

```bash
# JSON format
--generation-config '{"temperature": 0.7, "max_tokens": 2048, "top_p": 0.9}'

# Key=value format
--generation-config "temperature=0.7,max_tokens=2048,top_p=0.9"
```

Common generation parameters: `temperature`, `max_tokens`, `top_p`, `top_k`, `do_sample`, `repetition_penalty`.

## API Connection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--api-url` | str | None | API endpoint URL (setting this auto-selects `openai_api` eval type) |
| `--api-key` | str | `EMPTY` | API authentication key |
| `--timeout` | float | None | Request timeout in seconds |
| `--stream` | flag | None | Enable streaming mode |

## Judge / LLM Review

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--judge-strategy` | str | `auto` | Judge strategy: `auto`, `rule`, `llm`, `llm_recall` |
| `--judge-model-args` | JSON str | `{}` | Judge model configuration (model, api_url, api_key, etc.) |
| `--judge-worker-num` | int | 1 | Number of parallel judge workers |
| `--analysis-report` | flag | false | Generate analysis report using judge model |

### Judge Strategy

| Strategy | Description |
|----------|-------------|
| `auto` | Automatically choose rule or LLM based on benchmark |
| `rule` | Rule-based evaluation only (exact match, regex, etc.) |
| `llm` | Use an LLM as judge for all evaluations |
| `llm_recall` | Use LLM judge only for items that failed rule-based evaluation |

Example with LLM judge:
```bash
evalscope eval \
  --model qwen-plus \
  --datasets arena_hard \
  --api-url http://localhost:8000/v1/chat/completions \
  --judge-strategy llm \
  --judge-model-args '{"model": "gpt-4", "api_url": "https://api.openai.com/v1/chat/completions", "api_key": "sk-xxx"}'
```

## Cache and Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--work-dir` | str | `./outputs` | Root output directory |
| `--use-cache` | str | None | Path to reuse cached results from a previous run |
| `--rerun-review` | flag | false | Re-run the review/judge step when using cache |
| `--no-timestamp` | flag | false | Don't add timestamp subdirectory (overwrite previous results) |
| `--enable-progress-tracker` | flag | false | Write progress.json for tracking |

## Debug and Runtime

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--debug` | flag | false | Verbose debug output |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--ignore-errors` | flag | false | Continue evaluation even if some items fail |

## Sandbox

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-sandbox` | flag | false | Run evaluation in sandboxed environment |
| `--sandbox-type` | str | `docker` | Sandbox type: `docker` or `volcengine` |
| `--sandbox-manager-config` | JSON str | `{}` | Sandbox configuration |

## Scenario Templates

### Local Model Quick Evaluation

```bash
evalscope eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --datasets gsm8k arc \
  --limit 10 \
  --seed 42
```

### API Model Evaluation

```bash
evalscope eval \
  --model qwen-plus \
  --datasets mmlu gsm8k ifeval \
  --api-url http://localhost:8000/v1/chat/completions \
  --api-key sk-xxx \
  --limit 50
```

### Multimodal Model Evaluation

```bash
# Using VLMEvalKit backend
evalscope eval \
  --model qwen-vl-plus \
  --eval-backend VLMEvalKit \
  --eval-config vlm_config.yaml \
  --datasets mmmu mm_bench

# Using native backend with VL benchmarks
evalscope eval \
  --model qwen-vl-plus \
  --datasets mmmu math_vista \
  --api-url http://localhost:8000/v1/chat/completions \
  --api-key sk-xxx
```

### Full Evaluation with Custom Dataset Args

```bash
evalscope eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --datasets gsm8k arc ceval mmlu \
  --dataset-args '{
    "gsm8k": {"few_shot_num": 4, "few_shot_random": false},
    "arc": {"subset_list": ["ARC-Challenge"]},
    "ceval": {"few_shot_num": 5}
  }' \
  --generation-config '{"temperature": 0.0, "max_tokens": 2048}' \
  --work-dir ./my_eval_outputs \
  --seed 42
```

### Evaluation with LLM Judge

```bash
evalscope eval \
  --model qwen-plus \
  --datasets arena_hard \
  --api-url http://localhost:8000/v1/chat/completions \
  --api-key sk-xxx \
  --judge-strategy llm \
  --judge-model-args '{"model": "gpt-4o", "api_url": "https://api.openai.com/v1/chat/completions", "api_key": "sk-xxx"}' \
  --judge-worker-num 4
```

### Resume from Cache

```bash
evalscope eval \
  --model qwen-plus \
  --datasets mmlu gsm8k \
  --api-url http://localhost:8000/v1/chat/completions \
  --use-cache ./outputs/20250101_120000
```
