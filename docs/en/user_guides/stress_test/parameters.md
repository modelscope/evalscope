# Parameter

Execute `evalscope perf --help` to get a full parameter description.

## Basic Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--model` | `str` | Name or path of the test model | - |
| `--url` | `str` | API address, supporting `/chat/completion` and `/completion` endpoints | - |
| `--name` | `str` | Name for wandb/swanlab database result and result database | `{model_name}_{current_time}` |
| `--api` | `str` | Service API type<br>• `openai`: OpenAI-compatible API (requires `--url`)<br>• `local`: Start local transformers inference<br>• `local_vllm`: Start local vLLM inference service<br>• Custom: See [Custom API Guide](./custom.md#custom-api-requests) | - |
| `--port` | `int` | Port for local inference service<br>Only applicable to `local` and `local_vllm` | `8877` |
| `--attn-implementation` | `str` | Attention implementation method<br>Only effective when `api=local` | `None`<br>(Optional: `flash_attention_2`, `eager`, `sdpa`) |
| `--api-key` | `str` | API key | `None` |
| `--debug` | `bool` | Whether to output debug information | `False` |

## Network Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--total-timeout` | `int` | Total timeout for each request (seconds) | `21600` (6 hours) |
| `--connect-timeout` | `int` | Network connection timeout (seconds) | `None` |
| `--read-timeout` | `int` | Network read timeout (seconds) | `None` |
| `--headers` | `str` | Additional HTTP headers<br>Format: `key1=value1 key2=value2`<br>Will be used for each query | - |
| `--no-test-connection` | `bool` | Do not send connection test, start stress test directly | `False` |

## Request Control

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--parallel` | `list[int]` | Number of concurrent requests<br>Can input multiple values separated by spaces | `1` |
| `--number` | `list[int]` | Total number of requests to be sent<br>Can input multiple values (must correspond one-to-one with `parallel`) | `1000` |
| `--rate` | `float` | Number of requests generated per second<br>• `-1`: All requests generated at time 0 with no interval<br>• Other values: Use Poisson process to generate request intervals | `-1` |
| `--log-every-n-query` | `int` | Log every N queries | `10` |
| `--stream` | `bool` | Whether to use SSE stream output<br>Must be enabled to measure TTFT (Time to First Token) metric | `True` |
| `--sleep-interval` | `int` | Sleep time between each performance test (seconds)<br>Helps avoid overloading the server | `5` |

```{tip}
In the implementation of this tool, request generation and sending are separate:
- The `--rate` parameter controls the number of requests generated per second, which are placed in a request queue
- The `--parallel` parameter controls the number of workers sending requests, with each worker retrieving requests from the queue and sending them, only proceeding to the next request after receiving a response to the previous one
```
## SLA Settings

| Parameter | Type | Description | Default |
|------|------|------|--------|
| `--sla-auto-tune` | `bool` | Whether to enable SLA auto-tuning mode | `False` |
| `--sla-variable` | `str` | Variable for auto-tuning<br>Options: `parallel` (concurrency), `rate` (request rate) | `parallel` |
| `--sla-params` | `str` | SLA constraint conditions<br>JSON string<br>Supported metrics: `avg_latency`, `p99_latency`, `avg_ttft`, `p99_ttft`, `avg_tpot`, `p99_tpot`, `rps`, `tps`<br>Supported operators: `<=`, `<`, `min` (for latency metrics); `>=`, `>`, `max` (for throughput metrics)<br>Example: `'[{"p99_latency": "<=2"}]'` | `None` |
| `--sla-upper-bound` | `int` | Maximum concurrency/rate limit for auto-tuning | `65536` |
| `--sla-lower-bound` | `int` | Minimum concurrency/rate limit for auto-tuning | `1` |
| `--sla-num-runs` | `int` | Number of runs per concurrency level (average taken) | `3` |

```{seealso}
For details on using the SLA auto-tuning feature, see the [Auto-tuning Guide](./sla_auto_tune.md).
```

## Prompt Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--max-prompt-length` | `int` | Maximum input prompt length<br>Prompts exceeding this length will be discarded | `131072` |
| `--min-prompt-length` | `int` | Minimum input prompt length<br>Prompts shorter than this will be discarded | `0` |
| `--prefix-length` | `int` | Length of the prompt prefix<br>Only effective for `random` dataset | `0` |
| `--prompt` | `str` | Specify request prompt<br>String or local file (specify via `@/path/to/file`)<br>Higher priority than `dataset`<br>Example: `@./prompt.txt` | - |
| `--query-template` | `str` | Specify query template<br>JSON string or local file (specify via `@/path/to/file`)<br>Example: `@./query_template.json` | - |
| `--apply-chat-template` | `bool` | Whether to apply chat template | `None` (automatically determined based on URL suffix) |
| `--image-width` | `int` | Image width for random VL dataset | `224` |
| `--image-height` | `int` | Image height for random VL dataset | `224` |
| `--image-format` | `str` | Image format for random VL dataset | `RGB` |
| `--image-num` | `int` | Number of images for random VL dataset | `1` |
| `--image-patch-size` | `int` | Patch size for the image<br>Only used for local image token calculation | `28` |

## Dataset Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--dataset` | `str` | Dataset mode, see table below for details | - |
| `--dataset-path` | `str` | Dataset file path<br>Used in conjunction with dataset | - |

### Dataset Mode Description

| Mode | Description | Supports dataset-path |
|------|-------------|----------------------|
| `openqa` | Automatically downloads [OpenQA](https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/summary) from ModelScope<br>Prompts are relatively short (usually <100 tokens)<br>Uses `question` field from jsonl file when `dataset_path` is specified | ✓ |
| `longalpaca` | Automatically downloads [LongAlpaca-12k](https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/dataPeview) from ModelScope<br>Prompts are much longer (generally >6000 tokens)<br>Uses `instruction` field from jsonl file when `dataset_path` is specified | ✓ |
| `line_by_line` | Each line in txt file is used as a separate prompt<br>**Requires `dataset_path`** | ✓ (Required) |
| `flickr8k` | Automatically downloads [Flick8k](https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/dataPeview) from ModelScope<br>Builds image-text inputs; large dataset suitable for evaluating multimodal models | ✗ |
| `kontext_bench` | Automatically downloads [Kontext-Bench](https://modelscope.cn/datasets/black-forest-labs/kontext-bench/dataPeview) from ModelScope<br>Builds image-text inputs; approximately 1,000 samples, suitable for quick evaluation of multimodal models | ✗ |
| `random` | Randomly generates prompts based on `prefix-length`, `max-prompt-length`, and `min-prompt-length`<br>**Requires `tokenizer-path`**<br>[Usage example](./examples.md#using-the-random-dataset) | ✗ |
| `random_vl` | Randomly generates both image and text inputs<br>Based on `random`, with additional image-related parameters<br>[Usage example](./examples.md#using-the-random-multimodal-dataset) | ✗ |
| `custom` | Custom dataset parser<br>See [Custom Dataset Guide](custom.md#custom-dataset) | ✓ |

## Model Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--tokenizer-path` | `str` | Tokenizer weights path<br>Used to calculate the number of tokens in input and output<br>Usually located in the same directory as model weights | `None` |
| `--frequency-penalty` | `float` | frequency_penalty value | - |
| `--logprobs` | `bool` | Whether to return logarithmic probabilities | - |
| `--max-tokens` | `int` | Maximum number of tokens that can be generated | - |
| `--min-tokens` | `int` | Minimum number of tokens to generate<br>Note: Not all model services support this parameter<br>For `vLLM>=0.8.1`, you need to additionally set<br>`--extra-args '{"ignore_eos": true}'` | - |
| `--n-choices` | `int` | Number of completion choices to generate | - |
| `--seed` | `int` | Random seed | `None` |
| `--stop` | `str` | Tokens that stop the generation | - |
| `--stop-token-ids` | `list[int]` | IDs of tokens that stop the generation | - |
| `--temperature` | `float` | Sampling temperature | `0` |
| `--top-p` | `float` | Top-p sampling | - |
| `--top-k` | `int` | Top-k sampling | - |
| `--extra-args` | `str` | Additional parameters to be passed in the request body<br>JSON string format<br>Example: `'{"ignore_eos": true}'` | - |

## Data Storage

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--visualizer` | `str` | Visualizer to use<br>Options: `wandb`, `swanlab`, `clearml`<br>If set, metrics will be saved to the specified visualizer | `None` |
| `--wandb-api-key` | `str` | wandb API key for logging metrics to wandb<br>**Deprecated**, please use `--visualizer wandb` instead | - |
| `--swanlab-api-key` | `str` | swanlab API key for logging metrics to swanlab<br>**Deprecated**, please use `--visualizer swanlab` instead | - |
| `--outputs-dir` | `str` | Output file path | `./outputs` |

## Other Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--db-commit-interval` | `int` | Number of rows buffered before writing results to SQLite database | `1000` |
| `--queue-size-multiplier` | `int` | Maximum size of the request queue<br>Calculated as: `parallel * multiplier` | `5` |
| `--in-flight-task-multiplier` | `int` | Maximum number of in-flight tasks<br>Calculated as: `parallel * multiplier` | `2` |