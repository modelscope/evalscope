# Perf Parameter Reference

Complete parameter reference for `evalscope perf`. All parameters can also be discovered via `evalscope perf --help`.

## Model and API

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | (required) | Model name or path |
| `--api` | str | `openai` | API protocol: `openai`, `local`, `local_vllm`, `dashscope`, `embedding`, `rerank`, `custom` |
| `--url` | str | `http://127.0.0.1:8877/v1/chat/completions` | API endpoint URL |
| `--port` | int | 8877 | Port for local inference server |
| `--api-key` | str | None | API authentication key |
| `--tokenizer-path` | str | None | Tokenizer path for accurate token counting |
| `--attn-implementation` | str | None | Attention implementation (local inference only) |

### API Types

| Value | Use Case |
|-------|----------|
| `openai` | OpenAI-compatible API endpoints (default) |
| `local` | Auto-start local inference server using the model |
| `local_vllm` | Auto-start local vLLM inference server |
| `dashscope` | Alibaba Cloud DashScope API |
| `embedding` | OpenAI-compatible embedding API |
| `rerank` | OpenAI-compatible reranking API |
| `custom` | Custom API implementation |

## Connection Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--headers` | key=value pairs | None | Extra HTTP headers (e.g. `--headers key1=val1 key2=val2`) |
| `--connect-timeout` | int | None | Network connection timeout (seconds) |
| `--read-timeout` | int | None | Network read timeout (seconds) |
| `--total-timeout` | int | 21600 | Total request timeout in seconds (default: 6 hours) |
| `--no-test-connection` | flag | false | Skip connection test before starting benchmark |

## Concurrency and Load

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--parallel` | int (multiple) | 1 | Number of concurrent requests. Supports multiple values for gradient tests |
| `-n, --number` | int (multiple) | 1000 | Total number of requests. Must match `--parallel` count |
| `--rate` | float | -1 | Requests per second limit (-1 = unlimited) |
| `--sleep-interval` | int | 5 | Sleep seconds between consecutive perf runs |

**Important**: When providing multiple values for `--parallel` and `--number`, they are paired positionally:
```bash
# Run 3 tests: (parallel=1, n=100), (parallel=5, n=500), (parallel=10, n=1000)
--parallel 1 5 10 --number 100 500 1000
```

## SLA Auto-Tuning

Automatically find the maximum concurrency or rate that satisfies SLA constraints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sla-auto-tune` | flag | false | Enable SLA auto-tuning |
| `--sla-variable` | str | `parallel` | Variable to tune: `parallel` or `rate` |
| `--sla-params` | JSON str | None | SLA constraints as JSON array |
| `--sla-num-runs` | int | 3 | Runs to average per configuration |
| `--sla-upper-bound` | int | 65536 | Upper bound for binary search |
| `--sla-lower-bound` | int | 1 | Lower bound for binary search |
| `--sla-number-multiplier` | float | None | Multiplier: `number = round(variable * N)`. Default: 2 |

### SLA Params Format

`--sla-params` accepts a JSON array of constraint objects:

```json
[
  {"name": "latency", "operator": "<=", "value": 5.0},
  {"name": "ttft", "operator": "<=", "value": 0.5},
  {"name": "output_token_throughput", "operator": ">=", "value": 100}
]
```

Available constraint names: `latency`, `ttft`, `tpot`, `request_throughput`, `output_token_throughput`, `total_token_throughput`.

Operators: `<=`, `>=`, `<`, `>`, `==`.

## Dataset / Prompt Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | str | `openqa` | Dataset type (see below) |
| `--dataset-path` | str | None | Path to custom dataset file |
| `--prompt` | str | None | Use a fixed prompt text for all requests |
| `--query-template` | str | None | Jinja2 template for request formatting |
| `--apply-chat-template` | flag | auto | Apply chat template to prompts (auto-detected from URL) |

### Dataset Types

| Type | Description |
|------|-------------|
| `openqa` | Open-ended QA prompts (default, good for general testing) |
| `random` | Random token sequences (for precise input length control) |
| `speed_benchmark` | Speed benchmark with fixed prompts |
| `random_vl` | Random vision-language data (for multimodal models) |
| `line_by_line` | Read prompts from a file, one per line |
| `flickr8k` | Flickr8K vision-language dataset |
| `longalpaca` | Long-context prompts |
| `embedding_dataset` | For embedding model benchmarks |
| `rerank_dataset` | For reranking model benchmarks |
| `custom` | Custom dataset plugin |

### Prompt Length Control (for `random` dataset)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--min-prompt-length` | int | 0 | Minimum input prompt length. Unit is **tokens** when `--tokenizer-path` is provided, otherwise **characters** (`len(prompt)`) |
| `--max-prompt-length` | int | `sys.maxsize` (no cap) | Maximum input prompt length. Unit is **tokens** when `--tokenizer-path` is provided, otherwise **characters** (`len(prompt)`) |
| `--prefix-length` | int | 0 | Fixed prefix length for random prompts |

## Vision-Language Settings (for `random_vl` dataset)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--image-width` | int | 224 | Image width in pixels |
| `--image-height` | int | 224 | Image height in pixels |
| `--image-format` | str | `RGB` | Image format |
| `--image-num` | int | 1 | Number of images per request |
| `--image-patch-size` | int | 28 | Patch size for image tokenizer (local token calculation) |

## Response Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max-tokens` | int | 2048 | Maximum output tokens |
| `--min-tokens` | int | None | Minimum output tokens (forces model to generate at least N tokens) |
| `--temperature` | float | 0.0 | Sampling temperature |
| `--top-p` | float | None | Top-p (nucleus) sampling |
| `--top-k` | int | None | Top-k sampling |
| `--stream` / `--no-stream` | flag | stream | Enable/disable streaming (default: enabled) |
| `--frequency-penalty` | float | None | Frequency penalty |
| `--repetition-penalty` | float | None | Repetition penalty |
| `--n-choices` | int | None | Number of response choices |
| `--seed` | int | None | Random seed |
| `--stop` | str (multiple) | None | Stop sequences |
| `--stop-token-ids` | str (multiple) | None | Stop token IDs |
| `--logprobs` | flag | None | Return log probabilities |
| `--extra-args` | JSON str | `{}` | Additional API-specific arguments |

## Output and Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--outputs-dir` | str | `outputs` | Output directory |
| `--no-timestamp` | flag | false | Don't add timestamp to output directory |
| `--name` | str | None | Custom name for the run (used in DB and visualizer) |
| `--debug` | flag | false | Enable debug logging |
| `--log-every-n-query` | int | 10 | Log progress every N queries |
| `--enable-progress-tracker` | flag | false | Write progress.json for tracking |

## Visualization Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--visualizer` | str | None | Visualizer: `wandb`, `swanlab`, `clearml` |
| `--wandb-api-key` | str | None | W&B API key |
| `--swanlab-api-key` | str | None | SwanLab API key |

## Performance Tuning Knobs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--db-commit-interval` | int | 1000 | Rows buffered before SQLite commit |
| `--queue-size-multiplier` | int | 5 | Request queue maxsize = parallel * multiplier |
| `--in-flight-task-multiplier` | int | 2 | Max scheduled tasks = parallel * multiplier |

## Scenario Templates

### Basic API Throughput Test

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 1 \
  --number 100 \
  --max-tokens 512 \
  --stream
```

### Concurrency Gradient Test

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 1 2 4 8 16 32 \
  --number 50 100 200 400 800 1600 \
  --stream
```

### Precise Input/Output Token Length Test

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset random \
  --min-prompt-length 2000 \
  --max-prompt-length 2000 \
  --max-tokens 512 \
  --min-tokens 512 \
  --parallel 5 \
  --number 100
```

### SLA Auto-Tuning (Find Max Concurrency)

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"name": "latency", "operator": "<=", "value": 5.0}]' \
  --sla-num-runs 3
```

### SLA Auto-Tuning (Find Max Rate)

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --sla-auto-tune \
  --sla-variable rate \
  --sla-params '[{"name": "ttft", "operator": "<=", "value": 0.5}, {"name": "latency", "operator": "<=", "value": 3.0}]'
```

### Vision-Language Model Test

```bash
evalscope perf \
  --model qwen-vl-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset random_vl \
  --image-width 512 \
  --image-height 512 \
  --image-num 1 \
  --parallel 2 \
  --number 50 \
  --max-tokens 256 \
  --stream
```

### Local Model Inference Test

```bash
evalscope perf \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --api local \
  --dataset openqa \
  --parallel 1 \
  --number 20 \
  --max-tokens 512
```

### DashScope API Test

```bash
evalscope perf \
  --model qwen-plus \
  --api dashscope \
  --api-key sk-xxx \
  --dataset openqa \
  --parallel 5 \
  --number 100 \
  --stream
```

### Non-Streaming Latency Test

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 1 \
  --number 50 \
  --no-stream \
  --max-tokens 256
```

### Custom Dataset from File

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset line_by_line \
  --dataset-path /path/to/prompts.txt \
  --parallel 5 \
  --number 100
```

### With Visualization Logging

```bash
evalscope perf \
  --model qwen-plus \
  --url http://localhost:8000/v1/chat/completions \
  --api openai \
  --dataset openqa \
  --parallel 5 \
  --number 200 \
  --visualizer wandb \
  --wandb-api-key xxx \
  --name "qwen-plus-perf-test"
```
