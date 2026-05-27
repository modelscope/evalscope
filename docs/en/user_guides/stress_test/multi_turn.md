# Multi-turn Conversation Benchmark

The multi-turn conversation benchmark allows you to test a model service in realistic multi-turn interaction scenarios. Unlike a standard benchmark, multi-turn mode appends the model's **actual replies** to the conversation context so that every subsequent request carries the full history — faithfully simulating a user's continuous conversation with the model, and enabling measurement of latency, throughput, and KV-cache utilization as context grows.

## Features

- **Real context accumulation**: After each successful turn, the model's actual output is appended to the conversation history; the next turn sends the complete history rather than just the current user message.
- **Approx KV cache hit rate estimation**: Based on client-side token counts, estimates the proportion of history tokens relative to the total input tokens in each request — i.e., the theoretical upper bound of tokens that could benefit from server-side prefix caching. Whether caching actually occurs depends on whether the server has prefix caching enabled and has retained the relevant cache.
- **Multiple dataset support**: Provides random synthetic (`random_multi_turn`), real conversations (`share_gpt_zh_multi_turn` / `share_gpt_en_multi_turn`), custom local data (`custom_multi_turn`), real Agent trajectories (`swe_smith`), and production agentic trace replay (`trie_agentic_coding` / `trie_code_qa` / `trie_office_work`) datasets.
- **Consistent parameter semantics**: `--number` is the total number of conversations and `--parallel` is the number of concurrent conversations, keeping the same semantics as standard benchmark mode.

## Parameters

### Multi-turn Specific Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--multi-turn` | `bool` | Enable multi-turn conversation benchmark mode | `False` |
| `--min-turns` | `int` | Minimum number of user turns per conversation; used by `random_multi_turn` only | `1` |
| `--max-turns` | `int` | Maximum number of user turns per conversation; **required** for `random_multi_turn`; optional for ShareGPT / `custom_multi_turn` datasets to truncate long conversations; for `swe_smith` live construction, the per-conversation turn count is sampled from `[min_turns, max_turns]` | `None` |
| `--dataset-offset` | `int` | Skip the first N conversations in the dataset; useful for sharded testing or avoiding cache hits | `0` |

### `multi_turn_args` (swe_smith-specific parameters)

The `swe_smith` dataset's live construction mode supports fine-grained control of conversation structure and token-length targets via a `MultiTurnArgs` object.

The number of turns per conversation is sampled from `[--min-turns, --max-turns]`; the amount of content filled per turn is controlled by the token-length parameters below.

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `first_turn_length` | `int` | Target prompt token count for turn 1; trajectory messages are sliced until this length is reached | `65000` |
| `subsequent_turn_length` | `int` | Target token increment per subsequent turn; controls the delta size added each round | `500` |
| `chars_per_token` | `float` | Characters-per-token estimate used for pre-filtering trajectories when no tokenizer is available | `3.0` |
| `num_workers` | `int` | Number of parallel workers for live conversation building (>1 uses multiprocessing.Pool) | `4` |

### Semantics of Existing Parameters in Multi-turn Mode

| Parameter | Meaning in multi-turn mode |
|-----------|---------------------------|
| `--number` | Total number of **conversations** to run; all workers stop once this many conversations have been completed |
| `--parallel` | Number of concurrently active conversations (each worker owns one conversation) |

## Workflow

1. **Load conversation pool**: At startup, conversations are read sequentially from the dataset file and pre-loaded into memory, up to a maximum of `--number` conversations (to avoid excessive memory usage with large datasets).

2. **Start workers**: `--parallel` concurrent coroutines (workers) are launched, each running independently and sharing a single global conversation counter.

3. **Conversation assignment (first-come, first-served, sequential cycling)**: Conversations are assigned to workers in order. Whoever finishes the current conversation first picks up the next one. Once all conversations are exhausted, cycling restarts from the beginning.

   Example with `--parallel 3` and 5 conversations:

   ```
   Startup: 3 workers begin simultaneously

   Worker 0 → takes conversation 1
   Worker 1 → takes conversation 2
   Worker 2 → takes conversation 3

   ↓ Each sends requests (other workers can continue executing while waiting for replies)

   Suppose Worker 1 finishes all turns of conversation 2 first:
   Worker 1 → takes conversation 4

   Suppose Worker 0 finishes conversation 1 second:
   Worker 0 → takes conversation 5

   If --number is not yet reached after conversation 5:
   Worker 0 → cycles back to conversation 1

   ... and so on — whoever finishes first picks up the next one
   ```

4. **Single-conversation execution (turn by turn)**: Each worker maintains its own independent conversation history, not shared with other workers:

   ```
   Turn 1: send [user_msg_1]                                         → receive model_reply_1
   Turn 2: send [user_msg_1, model_reply_1, user_msg_2]              → receive model_reply_2
   Turn 3: send [user_msg_1, model_reply_1, user_msg_2, model_reply_2, user_msg_3] → ...
   ```

   After each turn, the user message is appended to the history before sending the request. The model's **actual reply** is then appended for use in the next turn. Dataset `assistant` content is never sent to the model — only real model outputs build the context.

5. **Budget control and stopping**: The global conversation counter is incremented synchronously before each conversation starts, ensuring the total number of completed conversations across all workers does not exceed `--number`. Once the limit is reached, all workers stop and no new conversations are started.

6. **Failure handling**: If a turn request fails, the current conversation is immediately abandoned and the worker starts a new conversation from scratch — no failed context is carried into subsequent requests.

7. **Metrics aggregation**: After all workers finish, all latency, throughput, and multi-turn-specific metrics (average context turns, KV cache hit rate) are aggregated and output.

> **Note**: When the request success rate is below 100%, interrupted conversations do not contribute subsequent turns to the context, which may result in lower reported KV cache hit rates.

## Datasets

evalscope provides the following multi-turn datasets:

### random_multi_turn

Generates synthetic token sequences based on the `random` dataset. Each conversation contains `[min_turns, max_turns]` user turns. No external data file is required, making it ideal for quick benchmarking and performance comparisons.

**Required**: `--tokenizer-path`, `--max-turns`

**Optional**: `--min-turns` (default `1`), `--min-prompt-length`, `--max-prompt-length` (control the token length range of each user message)

Each conversation produced by the dataset has the following structure:

```json
[
  {"role": "user", "content": "...turn 1 random token sequence..."},
  {"role": "user", "content": "...turn 2 random token sequence..."}
]
```

> **Note**: `--tokenize-prompt` is not supported in multi-turn mode and will be silently ignored. Multi-turn conversations are always sent as message dicts to the `/v1/chat/completions` endpoint.

**Usage example**: Quickly evaluate service performance at a specified prompt length distribution and conversation depth without a real dataset.

```bash
evalscope perf \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_MODEL \
  --url OPENAI_API_COMPAT_URL \
  --api openai \
  --dataset random_multi_turn \
  --min-prompt-length 256 \
  --max-prompt-length 512 \
  --max-tokens 256 \
  --multi-turn \
  --min-turns 2 \
  --max-turns 5 \
  --number 20 \
  --parallel 10
```

Example output:

```text
            Performance Overview
┏━━━━━━┳━━━━━━┳━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃Conc. ┃ Rate ┃ Num ┃  RPS ┃   Gen/s ┃ Success ┃
┡━━━━━━╇━━━━━━╇━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│   10 │    - │  20 │ 4.32 │ 1103.48 │  100.0% │
└──────┴──────┴─────┴──────┴─────────┴─────────┘


                    Per-Request Metrics
┏━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃Conc. ┃ Rate ┃ Metric        ┃    avg ┃   p50 ┃   p99 ┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│   10 │    - │ Latency (s)   │  2.289 │ 2.180 │ 3.541 │
│      │      │ TTFT (ms)     │   41.0 │  38.0 │  72.0 │
│      │      │ TPOT (ms)     │    9.0 │   8.8 │  11.0 │
│      │      │ Input Tokens  │  315.4 │ 280.0 │ 650.0 │
│      │      │ Output Tokens │   92.0 │  85.0 │ 128.0 │
│      │      │ Turns/Req     │   1.60 │     - │     - │
│      │      │ Cache Hit (%) │  58.1% │     - │     - │
└──────┴──────┴───────────────┴────────┴───────┴───────┘
```

**Interpreting the metrics**:

- `Turns/Req: 1.60`: Each request carried an average of 1.60 turns of context during the test, consistent with the `--min-turns 2 --max-turns 5` random sampling distribution.
- `Cache Hit (%): 58.1%`: About 58% of input tokens came from conversation history.

### share_gpt_multi_turn

Includes two datasets: `share_gpt_zh_multi_turn` (Chinese) and `share_gpt_en_multi_turn` (English). Uses real conversation data from [swift/sharegpt](https://www.modelscope.cn/datasets/swift/sharegpt) (~70k Chinese / English conversations), preserving the full user + assistant alternation, making it suitable for evaluating models against realistic conversation distributions.

- **Auto download**: When `--dataset-path` is not specified, the dataset is automatically downloaded from ModelScope.
- **Local data support**: Provide a local JSONL file via `--dataset-path` (one `conversation` object per line).
- **Optional truncation**: Use `--max-turns` to limit the maximum number of user turns used from each conversation.

Local JSONL dataset format (one conversation per line):

```json
{"conversation": [{"human": "Hello", "assistant": "Hi! How can I help you?"}, {"human": "Write me a poem", "assistant": "Sure, ..."}]}
```

Runtime context structure (when sending turn 2):

```json
[
  {"role": "user",      "content": "Hello"},
  {"role": "assistant", "content": "<model's actual reply to turn 1>"},
  {"role": "user",      "content": "Write me a poem"}
]
```

> **Note**: The reference assistant replies in the dataset are included for structural completeness only and are never sent directly to the model. At runtime, workers always append the model's **actual output** to the context to ensure accurate history.

**Usage example**: Evaluate service performance using a realistic user conversation distribution, better reflecting production environment behavior.

```bash
evalscope perf \
  --model YOUR_MODEL \
  --url OPENAI_API_COMPAT_URL \
  --api openai \
  --dataset share_gpt_zh_multi_turn \
  --max-tokens 512 \
  --multi-turn \
  --max-turns 3 \
  --number 300 \
  --parallel 10
```

If the dataset is already downloaded locally, use `--dataset-path` to avoid re-downloading:

```bash
evalscope perf \
  --model YOUR_MODEL \
  --url OPENAI_API_COMPAT_URL \
  --api openai \
  --dataset share_gpt_zh_multi_turn \
  --dataset-path /path/to/common_zh_70k.jsonl \
  --max-tokens 512 \
  --multi-turn \
  --max-turns 3 \
  --number 300 \
  --parallel 10
```

Example output:

```text
             Performance Overview
┏━━━━━━┳━━━━━━┳━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃Conc. ┃ Rate ┃ Num ┃  RPS ┃  Gen/s ┃ Success ┃
┡━━━━━━╇━━━━━━╇━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│   10 │    - │ 300 │ 3.87 │ 985.21 │  100.0% │
└──────┴──────┴─────┴──────┴────────┴─────────┘


                    Per-Request Metrics
┏━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃Conc. ┃ Rate ┃ Metric        ┃    avg ┃   p50 ┃   p99 ┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│   10 │    - │ Latency (s)   │  2.571 │ 2.340 │ 4.103 │
│      │      │ TTFT (ms)     │   55.0 │  50.0 │  98.0 │
│      │      │ TPOT (ms)     │   10.0 │   9.5 │  13.0 │
│      │      │ Input Tokens  │  428.7 │ 390.0 │ 912.0 │
│      │      │ Output Tokens │  186.3 │ 160.0 │ 384.0 │
│      │      │ Turns/Req     │   1.98 │     - │     - │
│      │      │ Cache Hit (%) │  53.7% │     - │     - │
└──────┴──────┴───────────────┴────────┴───────┴───────┘
```

**Interpreting the metrics**:

- `Turns/Req: 1.98`: Limited by `--max-turns 3`, each request carried approximately 2 turns of context on average, as expected (turn 1 has no history, turns 2 and 3 carry 1 and 2 turns of history respectively, averaging ~1.98).
- `Cache Hit (%): 53.7%`: Real conversations have longer context; history tokens account for ~54% of input.

### custom_multi_turn

Uses a local JSONL file as a custom multi-turn conversation dataset. Each line stores a complete conversation directly in **OpenAI messages format** — no format conversion required. Ideal for benchmarking with your own existing conversation data.

- **`--dataset-path` is required** and must point to a local JSONL file.
- **Optional truncation**: Use `--max-turns` to limit the maximum number of user turns per conversation.

**JSONL dataset format** (one conversation per line, as an OpenAI messages array):

```json
[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help you?"}, {"role": "user", "content": "Write me a poem"}]
[{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}, {"role": "user", "content": "Tell me more about it."}]
```

Each line must satisfy:
- Must be a JSON array.
- Every element must have `role` and `content` fields.
- `role` must be either `user` or `assistant`.
- Must contain at least one `user` message.

Runtime context structure (when sending turn 2):

```json
[
  {"role": "user",      "content": "Hello"},
  {"role": "assistant", "content": "<model's actual reply to turn 1>"},
  {"role": "user",      "content": "Write me a poem"}
]
```

> **Note**: The `assistant` messages in the dataset are used only to identify conversation structure and are **never** sent directly to the model. At runtime, workers always append the model's actual output to the context to ensure accurate history.

**Usage example**: You have conversation data already in OpenAI messages format and want to benchmark directly without any format conversion.

First, prepare the JSONL data file (one conversation per line):

```json
[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help you?"}, {"role": "user", "content": "Write me a poem"}, {"role": "assistant", "content": "Sure, ..."}, {"role": "user", "content": "Write another one"}]
[{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}, {"role": "user", "content": "What are some famous landmarks there?"}]
```

Then run the benchmark:

```bash
evalscope perf \
  --model YOUR_MODEL \
  --url OPENAI_API_COMPAT_URL \
  --api openai \
  --dataset custom_multi_turn \
  --dataset-path /path/to/my_conversations.jsonl \
  --max-tokens 512 \
  --multi-turn \
  --max-turns 3 \
  --number 100 \
  --parallel 10
```

### swe_smith

Uses real Agent code-repair trajectory data from [SWE-bench/SWE-smith-trajectories](https://www.modelscope.cn/datasets/SWE-bench/SWE-smith-trajectories), designed specifically for **long-context + multi-turn Agent scenario** benchmarking. Each trajectory consists of tool calls, code snippets, patch results, etc. A single prompt typically exceeds tens of thousands of tokens, making it ideal for evaluating prefill throughput and KV cache hit rates under large contexts.

Two data source modes are supported:

- **Pre-built JSON mode** (recommended): Specify `--dataset-path` to load a pre-generated `agentic_dataset.json`. No tokenizer is required and startup is fast.
- **Live construction mode** (no `--dataset-path`): Pulls raw trajectories from ModelScope at runtime and dynamically builds conversations. `--tokenizer-path` is **required** for accurate token counting.

Common features of both modes:
- **Offset support**: Skip the first N conversations via `--dataset-offset`, useful for sharded testing or avoiding KV cache hot-spots.

> **Note**: The turn count for each conversation is sampled from `[--min-turns, --max-turns]`; the amount of content per turn is determined by `first_turn_length` / `subsequent_turn_length`.

**Building the Dataset**

It is recommended to pre-build `agentic_dataset.json` using `examples/perf/build_swe_smith_dataset.py` before running a benchmark — build once, reuse many times, avoiding repeated downloads and on-the-fly construction.

**Key parameters**:

| Parameter | Description | Default |
|-----------|-------------|--------|
| `--model-path` | Tokenizer path for accurate token counting (ModelScope model ID or local path) | `Qwen/Qwen2.5-7B-Instruct` |
| `--first-turn-length` | Target prompt token count for turn 1 | `65000` |
| `--subsequent-turn-length` | Target token increment per subsequent turn | `500` |
| `--min-turns` | Minimum number of turns per conversation | `1` |
| `--max-turns` | Maximum number of turns per conversation; the actual turn count is sampled from `[min_turns, max_turns]` (consistent with live construction behaviour); defaults to `--min-turns` if not set | `None` |
| `--number` | Number of conversations to generate | `128` |
| `--output-path` | Output file path | `agentic_dataset.json` |
| `--seed` | Random seed for reproducibility | `42` |
| `--num-workers` | Number of parallel workers | CPU count |

```bash
python examples/perf/build_swe_smith_dataset.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --first-turn-length 8192 \
  --subsequent-turn-length 1024 \
  --min-turns 3 \
  --max-turns 8 \
  --number 128 \
  --output-path agentic_dataset.json \
  --seed 42 \
  --num-workers 8
```

**Usage Example: Pre-built JSON Mode (Recommended)**

After generating `agentic_dataset.json`, load it via `--dataset-path` — no tokenizer needed and startup is faster:

```bash
evalscope perf \
  --model YOUR_MODEL \
  --url OPENAI_API_COMPAT_URL \
  --api openai \
  --dataset swe_smith \
  --dataset-path /path/to/agentic_dataset.json \
  --max-tokens 512 \
  --multi-turn \
  --dataset-offset 100 \
  --number 200 \
  --parallel 20
```

> **Note**: `--dataset-offset` skips the first N conversations in the dataset, making it suitable for multi-machine sharded benchmarking or avoiding KV cache hot-spots.

**Usage Example: Live Construction Mode**

Automatically pulls SWE-smith-trajectories from ModelScope and constructs conversations at runtime. `--tokenizer-path` is required:

```bash
evalscope perf \
  --model YOUR_MODEL \
  --url OPENAI_API_COMPAT_URL \
  --api openai \
  --dataset swe_smith \
  --tokenizer-path YOUR_MODEL \
  --max-tokens 512 \
  --min-tokens 512 \
  --multi-turn \
  --multi-turn-args '{
      "first_turn_length": 8192,
      "subsequent_turn_length": 1024
  }' \
  --min-turns 3 \
  --max-turns 8 \
  --seed 42 \
  --number 10 20 \
  --parallel 5 10 \
  --extra-args '{"ignore_eos": true}'
```

### trie Trace Replay

Replays the token-length sequences of real production agentic workloads to measure inference-service performance under multi-turn, long-tail, tool-call-paced load. The traces come from [applied-compute/trie](https://github.com/applied-compute/trie) (Apache-2.0); evalscope re-hosts the three workloads on [ModelScope dataset `evalscope/trie-workloads`](https://www.modelscope.cn/datasets/evalscope/trie-workloads) and downloads them on demand.

Each trace contains only token-length metadata (per-turn prompt length, output length, tool-call wait time, etc.) without real conversation text. At benchmark time the client automatically synthesizes prompts to the recorded lengths, preserving the original load characteristics.

**Three built-in workloads**:

| Dataset alias | Source jsonl | Typical num_turns | When to use |
|---|---|---|---|
| `trie_agentic_coding` | `agentic_coding_8k.jsonl` | 8-15 | Coding-agent apps (IDE completion, refactor agents) |
| `trie_code_qa` | `code_qa_8k.jsonl` | 5-12 | Code-Q&A agents (repo analysis, doc generation) |
| `trie_office_work` | `office_work_8k.jsonl` | 18-41 | Office-automation agents (longest, heaviest; best for stress-ceiling tests) |

Each trace starts with ~8 k initial-prompt tokens; total input volume per trace is ~25 k-50 k tokens. Each jsonl has ~8000 traces, more than enough for long-running statistically meaningful tests.

Bring your own trace? Feed any jsonl in the same format via `--dataset-path`:

```bash
evalscope perf ... --dataset trie_office_work --dataset-path /path/to/your_traces.jsonl
```

**Required**: `--tokenizer-path` (for synthesizing prompts to exact target lengths)

**Usage example**:

```bash
evalscope perf \
  --model qwen-plus \
  --url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  --api-key $DASHSCOPE_API_KEY \
  --api openai \
  --dataset trie_office_work \
  --multi-turn \
  --parallel 4 \
  --number 10 \
  --duration 600 \
  --tokenizer-path Qwen/Qwen2.5-7B-Instruct \
  --extra-args '{"ignore_eos": true}' \
  --stream
```

> **Note**: trie trace replay relies on `ignore_eos` to force the model to generate to the exact recorded length. vLLM and SGLang support this parameter; DashScope / OpenAI API and most cloud services do not — the model stops at natural EOS, so actual output will be shorter than the recorded length. This does not affect latency / cache hit / decode TPS correctness.

## Multi-turn Output Metrics

In addition to the Performance Overview and Per-Request Metrics tables, multi-turn mode outputs the following additional tables.

### Per-Trace Metrics

Per-conversation (trace) metrics, aggregated as mean / p50 / p90 / p99 / max across all completed conversations:

| Column | Meaning | Where to look when it's off |
|---|---|---|
| **Latency (s)** | Wall-clock from conversation's first-turn start to last-turn completion | Slow conversation → check TTFAT split (is prefill slow or decode slow?) |
| **First-Turn TTFT (s)** | TTFT of the first turn; reflects cold-prefill performance | High → server's prefill stage is slow |
| **TTFAT (s)** | Wall-clock from conversation start to the first token of the final reply; end-to-end "how long until the final answer starts streaming" | Any slow turn in the conversation inflates this |
| **Decode TPS** | Arithmetic mean over the conversation's turns of `(completion-1)/(latency-ttft)` | Reflects steady-state decode speed |
| **Cache Hit Rate (%)** | `sum(cached_tokens) / sum(prompt_tokens)`; includes turn 1 (contributes 0 to numerator but counts in denominator) | Low → server prefix cache isn't enabled or evicts too aggressively |
| **Eligible Cache Hit Rate (%)** | Denominator only counts the theoretically cacheable prefix, excluding turn 1 and the current turn's new content | Both this and Cache Hit Rate low → cache off; this high but Cache Hit low → cache enabled but capacity-bound |

Also outputs `trace_summary.json` to the results directory.

### Workload Throughput

Time-based token throughput rates, output in all modes (single-turn and multi-turn):

```
┌──────┬──────┬─────────────────────┬─────────┬──────────┬──────────────────┐
│Conc. │ Rate │ Metric (tok/s)      │ Overall │ Last 30s │ Steady (drop 20%)│
├──────┼──────┼─────────────────────┼─────────┼──────────┼──────────────────┤
│   10 │    - │ Total Prompt tok/s  │  ...    │   ...    │           ...    │
│      │      │ New Prompt tok/s    │  ...    │   ...    │           ...    │
│      │      │ Cached Prompt tok/s │  ...    │   ...    │           ...    │
│      │      │ Completion tok/s    │  ...    │   ...    │           ...    │
└──────┴──────┴─────────────────────┴─────────┴──────────┴──────────────────┘
```

| Column | Meaning |
|---|---|
| **Overall** | total tokens / wall_time over the whole run |
| **Last 30s** | 30-second tail-window tok/s; falls back to Overall when the run is shorter than 30s |
| **Steady (drop 20%)** | tok/s after dropping the first 20% of wall_time; removes the cold-start phase |

| Row | Meaning |
|---|---|
| Total Prompt | New Prompt + Cached Prompt |
| New Prompt | `prompt_tokens - cached_tokens`; the part the server actually runs attention on |
| Cached Prompt | the part the server skipped via prefix cache hit |
| Completion | output-token rate |

Steady-state is the fairest number for "how fast can this endpoint sustainably run"; Last 30s is good for tail jitter. Also outputs `workload_throughput.json` and `workload_timeline.json` (raw cumulative-token timeline, pandas-friendly) to the results directory.

### First-Turn vs Subsequent-Turn TTFT

The `TTFT (ms)` in the Per-Request Metrics table averages every turn. In multi-turn scenarios, the first turn is a cold prefill while subsequent turns benefit heavily from prefix cache, so the two TTFT types can differ by 2-10x. The Per-Request Metrics table shows:

- `First-Turn TTFT (ms)`
- `Subsequent-Turn TTFT (ms)`

These appear only in multi-turn runs; single-turn benchmarks omit them.

## `--duration` Soft Exit

In multi-turn mode, `--duration` uses soft-exit semantics: once the deadline elapses, **no new conversations are claimed**, but **already in-flight conversations run every remaining turn** before exit. This keeps every conversation complete, so partial conversations don't skew statistics.

Three cap combinations:

| Command | Meaning |
|---|---|
| `--number 50` | run 50 conversations, no time limit |
| `--duration 300` | run for 300 seconds, however many conversations fit |
| `--number 50 --duration 300` | both caps; **whichever is reached first** ends the run (recommended) |

Side effect: actual wall_time may overshoot `--duration` slightly (the overshoot equals the time needed for in-flight conversations to finish).

## FAQs

### Chat-template token overhead

When using the `/v1/chat/completions` endpoint, the chat template adds 10-50 extra tokens per turn (role markers, special tokens, etc.), causing Cache Hit Rate to be 2-3pp lower than raw completions. This is expected behaviour.
