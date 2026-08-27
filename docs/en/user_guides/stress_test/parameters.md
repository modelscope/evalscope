# Parameter

Execute `evalscope perf --help` to get a full parameter description.

## Basic Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--model` | `str` | Name or path of the test model | - |
| `--url` | `str` | API address, supporting `/chat/completions`, `/completions`, and `/responses` endpoints | - |
| `--name` | `str` | Name for wandb/swanlab database result and result database | `{model_name}_{current_time}` |
| `--api` | `str` | Service API type<br>• `openai`: OpenAI-compatible Chat Completions API (requires `--url`)<br>• `openai_responses`: OpenAI official Responses API<br>• `openai_embedding`: OpenAI-compatible Embedding API<br>• `openai_rerank`: OpenAI/Cohere-compatible Rerank API<br>• `local`: Start local transformers inference<br>• `local_vllm`: Start local vLLM inference service<br>• Custom: See [Custom API Guide](./custom.md#custom-api-requests) | - |
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
| `--rate` | `float` | Request scheduling rate (requests/second)<br>• `-1`: No rate pacing; in the default closed-loop mode, requests are scheduled as fast as possible, but the number of in-flight HTTP requests is still capped by `--parallel`, so requests are not all sent to the server at once<br>• `> 0`: Requests are scheduled following a Poisson arrival model — the inter-arrival interval follows an exponential distribution with mean `1/rate`, resulting in an **average** of `rate` scheduled requests per second | `-1` |
| `--log-every-n-query` | `int` | Log every N queries | `100` |
| `--stream` | `bool` | Whether to use SSE stream output<br>Must be enabled to measure TTFT (Time to First Token) metric | `True` |
| `--sleep-interval` | `int` | Sleep time between each performance test (seconds)<br>Helps avoid overloading the server | `5` |
| `--open-loop` | `bool` | Enable open-loop mode: dispatch requests following a Poisson arrival schedule without semaphore backpressure.<br>Requests are fired at the rate set by `--rate` regardless of whether the server has finished processing previous requests.<br>• `--rate` becomes the sweep variable (accepts multiple values), replacing `--parallel` to drive multi-run iterations<br>• `--number` must have the same length as `--rate`; each pair `(rate, number)` corresponds to one independent run<br>• `--parallel` is ignored in this mode (internally set to -1 / INF)<br>See [Usage Example](./examples.md#open-loop-mode) | `False` |
| `--warmup-num` | `float` | Number or ratio of warmup requests:<br>• `0`: disabled (default)<br>• `>= 1`: absolute count, e.g. `--warmup-num 10` sends 10 warmup requests<br>• `0 < value < 1`: ratio mode, e.g. `--warmup-num 0.1` = 10% of `--number`<br>Warmup requests are sent with the same concurrency/rate as the benchmark but **excluded from performance metrics**<br>Useful for eliminating cold-start effects (KV-cache filling, JIT compilation, etc.)<br>In closed-loop mode, set it to `--parallel` or more, otherwise the first few requests inflate `p99`<br>See [Usage Example](./examples.md#warmup) | `0` |
| `--duration` | `float` | Wall-clock budget for one benchmark run (seconds)<br>Soft-exit semantics: once the deadline elapses **no new requests are dispatched**, but **already in-flight requests are allowed to finish** before exit<br>In multi-turn mode "in-flight" means **already-claimed traces run every remaining turn** (trace-level soft exit, aligned with upstream trie)<br>When combined with `--number`, **whichever cap is hit first** ends the run | `None` |

```{tip}
**Closed-loop (default)** vs **Open-loop** (`--open-loop`) — parameter behaviour comparison:

| | Closed-loop (default) | Open-loop (`--open-loop`) |
|---|---|---|
| **`--rate`** | Controls request scheduling rate (`-1` = no pacing, but still bounded by the `--parallel` concurrency cap; `R` = Poisson-arrival mean) | Controls dispatch rate; **must be > 0**; accepts multiple values (e.g. `5 10 20`), each driving one independent run |
| **`--number`** | Total requests per run; must match `--parallel` in length | Total requests per run; must match `--rate` in **length** |
| **`--parallel`** | Max in-flight requests; each worker waits for a response before sending the next (**backpressure**) | **Ignored**; concurrency is unbounded (INF); requests are fired on schedule without waiting for responses |
| **Use case** | Measure latency and throughput under controlled concurrency | Simulate realistic traffic (arrivals independent of service time); sweep throughput-latency curve across multiple rates |
```

## SLA Settings

| Parameter | Type | Description | Default |
|------|------|------|--------|
| `--sla-auto-tune` | `bool` | Whether to enable SLA auto-tuning mode | `False` |
| `--sla-variable` | `str` | Variable for auto-tuning<br>Options: `parallel` (concurrency), `rate` (request rate) | `parallel` |
| `--sla-params` | `str` | SLA constraint conditions<br>JSON string<br>Supported metrics: `avg_latency`, `p99_latency`, `avg_ttft`, `p99_ttft`, `avg_tpot`, `p99_tpot`, `rps`, `tps`<br>Supported operators: `<=`, `<`, `min` (for latency metrics); `>=`, `>`, `max` (for throughput metrics)<br>Example: `'[{"p99_latency": "<=2"}]'` | `None` |
| `--sla-upper-bound` | `int` | Upper bound of the tuned SLA variable search range | `65536` |
| `--sla-lower-bound` | `int` | Lower bound of the tuned SLA variable search range | `1` |
| `--sla-fixed-parallel` | `int` | Fixed parallel workers used when `--sla-variable=rate`; defaults to `--sla-upper-bound` for backward compatibility | `None` |
| `--sla-num-runs` | `int` | Number of runs per concurrency level (average taken) | `3` |
| `--sla-number-multiplier` | `float` | Multiplier of total requests relative to the tuned variable (concurrency or rate), i.e. `number = round(variable × N)`; defaults to `2` when not set | `None` |

```{seealso}
For details on using the SLA auto-tuning feature, see the [Auto-tuning Guide](./sla_auto_tune.md).
```

## Dataset Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|--------|
| `--dataset` | `str` | Dataset mode, see [Dataset Modes](#dataset-modes) below | - |
| `--dataset-path` | `str` | Dataset file or directory path<br>Points to a file: read directly; points to a directory: looks for the corresponding data file inside (for offline use with pre-downloaded dataset cache) | - |
| `--data-source` | `str` | Data source for dataset loading: `modelscope`, `huggingface`, or `local`<br>Defaults to `modelscope` when not specified; automatically treated as `local` when `--dataset-path` is a local directory | `modelscope` |
| `--dataset-args` | `str` | Per-dataset arguments (JSON string), validated against the schema of the selected `--dataset` (unknown keys raise an error) | - |

The keys carried by `--dataset-args` are documented in the sections where they apply:

| Key | Purpose | Section |
|-----|---------|---------|
| `target_input_len` / `input_len_mode` | Truncate real data to a fixed input length | [Length Control](#length-control) |
| `prefix_file` / `prefix_role` | Long-context prefix injection for very long fixed-length inputs | [Long-context Prefix Injection](#long-context-prefix-injection) |
| `speed` / `model_override` / `model_mapping` / `match_output_length` | Replay behaviour for production traffic | [Production Traffic Replay](#production-traffic-replay) |
| Token-length arguments | Multi-turn datasets | [Multi-turn Conversation](#multi-turn-conversation) |

```{note}
`--multi-turn-args` is deprecated; use `--dataset-args` instead (the key names are unchanged). The old flag still works and is automatically merged into `--dataset-args` (on a key conflict, `--dataset-args` takes precedence).
```

### Dataset Modes

**Text / Chat**

| Mode | Description | Supports dataset-path |
|------|-------------|----------------------|
| `openqa` | Automatically downloads [OpenQA](https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/summary) from ModelScope<br>Prompts are relatively short (usually <100 tokens)<br>Uses `question` field from jsonl file when `dataset_path` is specified | ✓ |
| `longalpaca` | Automatically downloads [LongAlpaca-12k](https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/dataPeview) from ModelScope<br>Prompts are much longer (generally >6000 tokens)<br>Uses `instruction` field from jsonl file when `dataset_path` is specified | ✓ |
| `line_by_line` | Each line in txt file is used as a separate prompt<br>**Requires `dataset_path`** | ✓ (Required) |
| `random` | Randomly generates prompts based on `prefix-length`, `max-prompt-length`, and `min-prompt-length`<br>**Requires `tokenizer-path`**<br>[Usage example](./examples.md#random-dataset) | ✗ |
| `custom` | Custom dataset parser<br>See [Custom Dataset Guide](custom.md#custom-dataset) | ✓ |

**Multimodal**

| Mode | Description | Supports dataset-path |
|------|-------------|----------------------|
| `flickr8k` | Automatically downloads [Flick8k](https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/dataPeview) from ModelScope<br>Builds image-text inputs; large dataset suitable for evaluating multimodal models<br>Supports `--dataset-path` pointing to a local dataset directory (offline) | ✓ (directory) |
| `kontext_bench` | Automatically downloads [Kontext-Bench](https://modelscope.cn/datasets/black-forest-labs/kontext-bench/dataPeview) from ModelScope<br>Builds image-text inputs; approximately 1,000 samples, suitable for quick evaluation of multimodal models<br>Supports `--dataset-path` pointing to a local dataset directory (offline) | ✓ (directory) |
| `random_vl` | Randomly generates both image and text inputs<br>Based on `random`, with additional image-related parameters<br>[Usage example](./examples.md#random-multimodal-dataset) | ✗ |

**Embedding**

| Mode | Description | Supports dataset-path |
|------|-------------|----------------------|
| `embedding` | Load text data from file to evaluate Embedding model<br>Supports Line-by-line (TXT) or JSONL format (with `text` field) | ✓ (Required) |
| `random_embedding` | Randomly generate queries based on `max-prompt-length` and `min-prompt-length` to evaluate Embedding model<br>**Must specify `tokenizer-path`** | ✗ |
| `embedding_batch` | Batch send text data to evaluate Embedding model<br>Load data from file<br>Supports `--extra-args '{"batch_size": 8}'` to set batch size | ✓ (Required) |
| `random_embedding_batch` | Batch send randomly generated query data to evaluate Embedding model<br>**Must specify `tokenizer-path`**<br>Supports `--extra-args '{"batch_size": 8}'` to set batch size | ✗ |

**Rerank**

| Mode | Description | Supports dataset-path |
|------|-------------|----------------------|
| `rerank` | Load Query-Document pairs from file to evaluate Rerank model<br>Supports JSONL format (with `query` and `documents` fields) | ✓ (Required) |
| `random_rerank` | Randomly generate query data to evaluate Rerank model<br>**Must specify `tokenizer-path`**<br>Supports `--extra-args '{"num_documents": 10, "document_length_ratio": 5}'` to set number of documents and length ratio | ✗ |

**Multi-turn Conversation**

Must be used with `--multi-turn`; see [Multi-turn Conversation](#multi-turn-conversation) for the parameters and the [Multi-turn Benchmark Guide](./multi_turn.md) for details.

| Mode | Description | Supports dataset-path |
|------|-------------|----------------------|
| `random_multi_turn` | Synthetic multi-turn conversations; each turn randomly generates a token sequence<br>**Requires `--tokenizer-path` and `--max-turns`**<br>[Usage example](./multi_turn.md#random_multi_turn) | ✗ |
| `share_gpt_zh_multi_turn` | Automatically downloads the Chinese [ShareGPT](https://www.modelscope.cn/datasets/swift/sharegpt) dataset (~70k conversations) from ModelScope, preserving full multi-turn conversations<br>[Usage example](./multi_turn.md#share_gpt_multi_turn) | ✓ |
| `share_gpt_en_multi_turn` | Automatically downloads the English [ShareGPT](https://www.modelscope.cn/datasets/swift/sharegpt) dataset (~70k conversations) from ModelScope, preserving full multi-turn conversations | ✓ |
| `custom_multi_turn` | Uses a local JSONL file as a custom multi-turn dataset<br>Each line must be a JSON array of OpenAI message dicts; ideal for benchmarking with your own conversation data<br>**Requires `--dataset-path`**<br>[Usage example](./multi_turn.md#custom_multi_turn) | ✓ (Required) |

**Production Traffic Replay**

Must be used with `--open-loop`; see [Production Traffic Replay](#production-traffic-replay) for the trace file format and replay arguments.

| Mode | Description | Supports dataset-path |
|------|-------------|----------------------|
| `workload_trace` | Replays a recorded production-traffic JSONL verbatim — original timestamps, request bodies, and headers — for benchmarking against real-world load (bursty arrivals, heterogeneous requests, multi-model routing)<br>Each request carries its own `model`, preserving multi-model routing<br>**Requires `--open-loop` and `--dataset-path`**; `--model`/`--number` are optional (the trace carries its own model and count) | ✓ (Required) |

## Input Construction

Controls the content and length of the input sent to the model. Below, `--xxx` entries are command-line flags while bare keys are passed through the `--dataset-args` JSON.

### Length Control

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--max-prompt-length` | `int` | Maximum input prompt length<br>Prompts exceeding this length will be discarded | `131072` |
| `--min-prompt-length` | `int` | Minimum input prompt length<br>Prompts shorter than this will be discarded | `0` |

To benchmark a **fixed input length** with real data (instead of `random`), use the following `--dataset-args` keys. Supported on `openqa`, `longalpaca`, `line_by_line` (plain-text lines only), and ShareGPT (`share_gpt_zh` / `share_gpt_en`). **Requires `--tokenizer-path`**.

| Key | Description | Default |
|-----|-------------|--------|
| `target_input_len` | Target input length in tokens. When set, every prompt is truncated to this length | disabled |
| `input_len_mode` | What to do with prompts **shorter than** the target: `cap` (keep as-is; that prompt may be shorter than the target); `drop` (discard it, so every emitted prompt is exactly the target). `drop` cannot be combined with `prefix_file` (see [Long-context Prefix Injection](#long-context-prefix-injection)) | `cap` |

```bash
# Truncate every input to 2048 tokens
evalscope perf \
  --model qwen2.5 --url http://127.0.0.1:8000/v1/completions \
  --dataset share_gpt_zh --tokenizer-path /path/to/tokenizer \
  --dataset-args '{"target_input_len": 2048}'
```

Lengths are counted as bare content tokens, without chat-template overhead. Multi-turn datasets like ShareGPT fit/filter only the **last user turn** by default (same as single-turn); the whole-conversation budget (sum of all message contents, over-long dropped, short padded by the prefix) applies only when a `prefix_file` is configured (see [Long-context Prefix Injection](#long-context-prefix-injection)). JSON lines in `line_by_line` (messages array / full request body) do not go through length control, so combining them with these keys raises an error.

How it differs from `--max/min-prompt-length`:
- `--max/min-prompt-length` only **filters** and never changes content — samples outside the range are dropped, so you get real samples of varying lengths;
- `target_input_len` **rewrites content** — it truncates every prompt to the given length, ideal for controlled fixed-length benchmarking.

> To get "every prompt exactly N tokens", you must use `target_input_len`; setting `--min-prompt-length` equal to `--max-prompt-length` cannot achieve it (real data almost never has prompts of exactly N tokens, so they get filtered out). The `random` dataset is the exception — it is generated on the fly, so min=max already yields a fixed length and this arg is not needed.

### Long-context Prefix Injection

Real instruction datasets are mostly 4K-8K tokens, so setting `target_input_len` to 128K filters everything out in `drop` mode and leaves varying lengths in `cap` mode. `prefix_file` lets you point at a long text (e.g. a book or document corpus); the framework slices it to exactly `target_input_len − prompt length` tokens and prepends it to each short prompt, so **every request's total input hits the target length** while keeping the low-entropy character of real human language (ideal for measuring prefix-cache hit rates and MTP acceptance rates). Supported on the same datasets as [Length Control](#length-control).

| Key | Description | Default |
|-----|-------------|--------|
| `prefix_file` | Path to the long prefix text file (UTF-8 plain text). **Requires `target_input_len`** and cannot be combined with `input_len_mode="drop"` | disabled |
| `prefix_role` | Injection role for the prefix: `system` (injected as a leading system message, matching real RAG traffic; inference engines usually have dedicated prefix-cache handling for system prompts); `user` (prepended directly to the user message content) | `system` |

```bash
# Align every request to exactly 131072 tokens using a long text prefix injected as the system role
evalscope perf \
  --model qwen2.5 --url http://127.0.0.1:8000/v1/chat/completions \
  --dataset openqa --tokenizer-path /path/to/tokenizer \
  --dataset-args '{"target_input_len": 131072, "prefix_file": "/path/to/long_text.txt", "prefix_role": "system"}'
```

Behavior notes:
- **Budget split**: the prompt is kept as-is (over-length prompts are truncated per `input_len_mode`); the prefix is sliced to exactly `target_input_len − tokens of all message contents`, so the total equals the target. Multi-turn history counts towards the budget (see the length convention in [Length Control](#length-control)).
- **Incompatible with `drop`**: `drop` only keeps prompts that already fill `target_input_len`, leaving a zero prefix budget, so the injection could never take effect — the combination is rejected at config time. Use `cap` plus prefix filling for fixed lengths.
- **Short prefix**: when the prefix file has fewer tokens than the remaining budget, it is repeated (tiled) to cover it and then truncated precisely, with a warning.
- **Fallback**: when `apply_chat_template` is off (e.g. the `/v1/completions` endpoint), a system message cannot be injected, so the prefix falls back to plain-text concatenation with a warning.
- **Join boundary**: with `prefix_role="user"` and in the plain-text fallback the prefix sits directly next to the prompt. The prefix and prompt are counted independently, so when the tokenizer merges or splits characters across the join the measured total can differ from the target by about ±1 token. `prefix_role="system"` (chat-template mode) is separated by message markers and is always exact.
- **Cache friendly**: all requests share the same prefix head (lengths differ slightly per prompt), which naturally suits prefix-cache hit testing.

```{note}
The `prefix_file` in this section injects a **real text** prefix for real datasets, whereas `--prefix-length` injects a **random token** prefix and only applies to the `random` dataset (see [Random Data Generation](#random-data-generation) below). They serve different purposes and should not be confused.
```

### Random Data Generation

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--prefix-length` | `int` | Length of the random token prefix<br>Only effective for the `random` dataset; all requests share the same prefix, which can be used to induce prefix-cache hits | `0` |
| `--image-width` | `int` | Image width for random VL dataset | `224` |
| `--image-height` | `int` | Image height for random VL dataset | `224` |
| `--image-format` | `str` | Image format for random VL dataset | `RGB` |
| `--image-num` | `int` | Number of images for random VL dataset | `1` |
| `--image-patch-size` | `int` | Patch size for the image<br>Only used for local image token calculation | `28` |

The length of `random` prompts is set by `--min-prompt-length` / `--max-prompt-length` (equal values yield a fixed length); `target_input_len` is not needed.

### Prompt and Template

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--prompt` | `str` | Specify request prompt<br>String or local file (specify via `@/path/to/file`)<br>Higher priority than `dataset`<br>Example: `@./prompt.txt` | - |
| `--query-template` | `str` | Specify query template<br>JSON string or local file (specify via `@/path/to/file`)<br>Example: `@./query_template.json` | - |
| `--apply-chat-template` | `bool` | Whether to apply chat template | `None` (automatically determined based on URL suffix) |
| `--tokenize-prompt` | `bool` | Tokenize the prompt client-side into a token-ID list and send it directly via `/v1/completions`, bypassing server-side re-tokenization | `False` |

## Multi-turn Conversation

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--multi-turn` | `bool` | Enable multi-turn conversation benchmark mode; `--number` is the total number of turns to send and `--parallel` is the number of concurrent turn-level requests | `False` |
| `--min-turns` | `int` | Minimum number of user turns per conversation; used by `random_multi_turn` and `swe_smith` | `1` |
| `--max-turns` | `int` | Maximum number of user turns per conversation; required for `random_multi_turn`; optional for ShareGPT / `custom_multi_turn` (truncates long conversations); for `swe_smith` it's the upper bound for per-conversation turn sampling, falling back to `--min-turns` when unset | `None` |
| `--num-workers` | `int` | Worker processes for CPU-bound dataset/request generation.<br>`0` = auto-detect from CPU affinity; `1` = serial (no multiprocessing); `>1` = explicit worker count.<br>Used by `random` (long-prompt parallel generation) and `swe_smith` (live construction). Supersedes the deprecated `multi_turn_args.num_workers`. | `0` |

Token-length arguments for multi-turn datasets such as `swe_smith` are passed via `--dataset-args`.

```{seealso}
Available multi-turn datasets are listed in [Dataset Modes](#dataset-modes); for full usage see the [Multi-turn Benchmark Guide](./multi_turn.md).
```

## Production Traffic Replay

Use `--dataset workload_trace` to replay recorded production traffic verbatim following its **original arrival timing**, closely matching real-world load. **Requires `--open-loop`**; no `--rate` is needed (arrival times come from the trace timestamps). See the full example at [Production Traffic Replay](./examples.md#production-traffic-replay).

The trace file is JSONL, one request record per line:

```json
{"body": {"model": "qwen-plus", "messages": [{"role": "user", "content": "hi"}]}, "timestamp": 1700000000.0}
{"body": {"model": "qwen-max", "messages": [{"role": "user", "content": "hello"}]}, "timestamp": 1700000001.5, "headers": {"X-Tag": "exp"}, "request_id": "req-42", "completion_tokens": 256}
```

| Field | Required | Description |
|-------|----------|-------------|
| `body` | ✓ | Complete request body (dict or JSON string), sent as-is |
| `timestamp` | ✓ | Arrival time (number or ISO-8601 string); only relative deltas matter; must be monotonically non-decreasing |
| `headers` | | Per-request HTTP headers (merged with CLI headers, CLI wins; hop-by-hop headers are stripped) |
| `request_id` | | Propagated to results for correlation with the original request |
| `completion_tokens` | | Used together with `match_output_length` |

Replay behaviour is tuned via `--dataset-args`:

| Key | Type | Description | Default |
|-----|------|-------------|--------|
| `speed` | float | Replay speed multiplier (2.0 = 2× faster, 0.5 = 2× slower) | `1.0` |
| `model_override` | str | Replace the `model` of every request with this value | disabled |
| `model_mapping` | dict | Remap `model` by name (a match takes priority; unmatched keeps the original) | disabled |
| `match_output_length` | bool | Set `max_tokens` from the recorded `completion_tokens` and enable `ignore_eos` (requires vLLM or compatible; `ignore_eos` is auto-skipped for constrained-decoding requests) | `false` |

```{note}
`--model` **does not rewrite** the trace body for `workload_trace` — each request keeps its own `model`, preserving multi-model routing. To rewrite models, use `model_override` / `model_mapping`.
```

## Model and Generation

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--tokenizer-path` | `str` | Tokenizer weights path<br>Used to calculate the number of tokens in input and output<br>Usually located in the same directory as model weights<br>When benchmarking a chat endpoint, the tokenizer must ship a chat template (see the note below) | `None` |
| `--frequency-penalty` | `float` | frequency_penalty value | - |
| `--logprobs` | `bool` | Whether to return logarithmic probabilities | - |
| `--max-tokens` | `int` or `int int` | Maximum number of tokens that can be generated<br>• A single integer: fixed value, e.g. `--max-tokens 2048`<br>• Two integers: `min max`, sampled uniformly at random per request, e.g. `--max-tokens 512 2048` | `2048` |
| `--min-tokens` | `int` | Minimum number of tokens to generate<br>Note: Not all model services support this parameter<br>For `vLLM>=0.8.1`, you need to additionally set<br>`--extra-args '{"ignore_eos": true}'` | - |
| `--n-choices` | `int` | Number of completion choices to generate | - |
| `--seed` | `int` | Random seed | `None` |
| `--stop` | `str` | Tokens that stop the generation | - |
| `--stop-token-ids` | `list[int]` | IDs of tokens that stop the generation | - |
| `--temperature` | `float` | Sampling temperature | `0` |
| `--top-p` | `float` | Top-p sampling | - |
| `--top-k` | `int` | Top-k sampling | - |
| `--extra-args` | `str` | Additional parameters to be passed in the request body<br>JSON string format<br>Example: `'{"ignore_eos": true}'` | - |

### Tokenizer and chat template

When benchmarking a `chat/completions` endpoint, `--apply-chat-template` is on by default and the client applies the chat template before counting tokens, so client-side lengths line up with the `usage.prompt_tokens` reported by the service; the tokenizer given to `--tokenizer-path` must therefore ship a Jinja chat template. DeepSeek-V3.2 / V4 provide `encoding` scripts instead, and base / pretrain checkpoints have no template either — those fail with an error that lists the available options.

Two things to watch out for: without `--tokenizer-path`, `--min-prompt-length` / `--max-prompt-length` filter by characters instead of tokens; and borrowing another model's tokenizer does not fail, but a mismatched vocabulary silently distorts token counts.

## Output

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--visualizer` | `str` | Visualizer to use<br>Options: `wandb`, `swanlab`, `clearml`<br>If set, metrics will be saved to the specified visualizer | `None` |
| `--enable-progress-tracker` | `bool` | Whether to enable progress tracking, writing hierarchical stress-test progress to `progress.json` in real time, queryable via the service API | `False` |
| `--wandb-api-key` | `str` | wandb API key for logging metrics to wandb<br>**Deprecated**, please use `--visualizer wandb` instead | - |
| `--swanlab-api-key` | `str` | swanlab API key for logging metrics to swanlab<br>**Deprecated**, please use `--visualizer swanlab` instead | - |
| `--outputs-dir` | `str` | Output file path | `./outputs` |
| `--no-timestamp` | `bool` | Exclude timestamp from output directory name | `False` |

## Other Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--db-commit-interval` | `int` | Number of rows buffered before writing results to SQLite database | `1000` |
| `--queue-size-multiplier` | `int` | Maximum size of the request queue<br>Calculated as: `parallel * multiplier` | `5` |
| `--in-flight-task-multiplier` | `int` | Maximum number of in-flight tasks<br>Calculated as: `parallel * multiplier` | `2` |
