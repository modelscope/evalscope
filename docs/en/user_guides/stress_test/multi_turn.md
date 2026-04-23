# Multi-turn Conversation Benchmark

The multi-turn conversation benchmark allows you to test a model service in realistic multi-turn interaction scenarios. Unlike a standard benchmark, multi-turn mode appends the model's **actual replies** to the conversation context so that every subsequent request carries the full history — faithfully simulating a user's continuous conversation with the model, and enabling measurement of latency, throughput, and KV-cache utilization as context grows.

## Features

- **Real context accumulation**: After each successful turn, the model's actual output is appended to the conversation history; the next turn sends the complete history rather than just the current user message.
- **Approx KV cache hit rate estimation**: Based on client-side token counts, estimates the proportion of history tokens relative to the total input tokens in each request — i.e., the theoretical upper bound of tokens that could benefit from server-side prefix caching. Whether caching actually occurs depends on whether the server has prefix caching enabled and has retained the relevant cache.
- **Multiple dataset support**: Provides four datasets — random synthetic (`random_multi_turn`), real conversations (`share_gpt_zh_multi_turn` / `share_gpt_en_multi_turn`), and custom local data (`custom_multi_turn`).
- **Consistent parameter semantics**: `--number` is the total number of turns and `--parallel` is the number of concurrent turns, keeping the same semantics as standard benchmark mode.

## Parameters

### Multi-turn Specific Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--multi-turn` | `bool` | Enable multi-turn conversation benchmark mode | `False` |
| `--min-turns` | `int` | Minimum number of user turns per conversation; used by `random_multi_turn` only | `1` |
| `--max-turns` | `int` | Maximum number of user turns per conversation; **required** for `random_multi_turn`; optional for ShareGPT datasets to truncate long conversations | `None` |

### Semantics of Existing Parameters in Multi-turn Mode

| Parameter | Meaning in multi-turn mode |
|-----------|---------------------------|
| `--number` | Total number of **turns** to send (not conversations); all workers stop once this many HTTP requests have been made |
| `--parallel` | Number of concurrently in-flight turn-level requests |

## Datasets

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

### share_gpt_zh_multi_turn / share_gpt_en_multi_turn

Uses real conversation data from [swift/sharegpt](https://www.modelscope.cn/datasets/swift/sharegpt) (~70k Chinese / English conversations), preserving the full user + assistant alternation, making it suitable for evaluating models against realistic conversation distributions.

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

**Runtime context structure** (when sending turn 2):

```json
[
  {"role": "user",      "content": "Hello"},
  {"role": "assistant", "content": "<model's actual reply to turn 1>"},
  {"role": "user",      "content": "Write me a poem"}
]
```

> **Note**: The `assistant` messages in the dataset are used only to identify conversation structure and are **never** sent directly to the model. At runtime, workers always append the model's actual output to the context to ensure accurate history.

## Workflow

1. **Load conversation pool**: At startup, conversations are read sequentially from the dataset file and pre-loaded into memory, up to a maximum of `--number` conversations (to avoid excessive memory usage with large datasets).

2. **Start workers**: `--parallel` concurrent coroutines (workers) are launched, each running independently and sharing a single global turn counter.

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

5. **Budget control and stopping**: The global turn counter is incremented synchronously before each request is sent, ensuring the total number of requests across all workers does not exceed `--number`. Once the limit is reached, all workers stop and no new conversations are started.

6. **Failure handling**: If a turn request fails, the current conversation is immediately abandoned and the worker starts a new conversation from scratch — no failed context is carried into subsequent requests.

7. **Metrics aggregation**: After all workers finish, all latency, throughput, and multi-turn-specific metrics (average context turns, KV cache hit rate) are aggregated and output.

> **Note**: When the request success rate is below 100%, interrupted conversations do not contribute subsequent turns to the context, which may result in lower reported KV cache hit rates.

## Output Metrics

In addition to all standard benchmark metrics, multi-turn mode outputs two extra metrics:

| Metric | Description | How to interpret |
|--------|-------------|-----------------|
| `Average input turns per request` | Average number of user turns in the context at the time each request is sent | Reflects the average growth of context during the test; a higher value means deeper conversations and longer prompts |
| `Average approx KV cache hit rate (%)` | Estimated proportion of history tokens relative to total input tokens (theoretical upper bound for prefix caching benefit) | A higher ratio means more of the input is historical context; if the server has prefix caching enabled, a higher ratio translates to greater latency savings. Calculation: `(prev_prompt_tokens + prev_completion_tokens) / current_prompt_tokens × 100%`; the first turn (no history) is excluded from the calculation |

All other metrics (Avg/P99 Latency, TTFT, TPOT, RPS, TPS) have the same meaning as in [standard benchmark mode](./parameters.md).

## Usage Examples

### 1. Using random_multi_turn (Synthetic Multi-turn Conversations)

Use case: Quickly evaluate service performance at a specified prompt length distribution and conversation depth without a real dataset.

```bash
evalscope perf \
  --model Qwen2.5-7B-Instruct \
  --tokenizer-path Qwen/Qwen2.5-7B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
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
                                    Detailed Performance Metrics
┏━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃      ┃      ┃      ┃     Avg ┃     P99 ┃     Avg ┃     P99 ┃     Avg ┃    P99 ┃    Gen. ┃ Success┃
┃Conc. ┃ Rate ┃  RPS ┃ Lat.(s) ┃ Lat.(s) ┃ TTFT(s) ┃ TTFT(s) ┃ TPOT(s) ┃ TPOT(… ┃  toks/s ┃    Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│   10 │  INF │ 4.32 │   2.289 │   3.541 │   0.041 │   0.072 │   0.009 │  0.011 │ 1103.48 │  100.0%│
└──────┴──────┴──────┴─────────┴─────────┴─────────┴─────────┴─────────┴────────┴─────────┴────────┘

                                          Request Metrics
┏━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃      ┃          ┃             ┃     P99 In ┃     Avg Out ┃    P99 Out ┃         Avg ┃      Approx┃
┃Conc. ┃ Num Reqs ┃ Avg In Toks ┃       Toks ┃        Toks ┃       Toks ┃   Turns/Req ┃   Cache Hit┃
│   10 │       20 │       315.4 │      650.0 │        92.0 │      128.0 │        1.60 │       58.1%│
└──────┴──────────┴─────────────┴────────────┴─────────────┴────────────┴─────────────┴────────────┘
```

**Interpreting the metrics**:

- `Avg Turns/Req: 1.60`: Each request carried an average of 1.60 turns of context during the test, consistent with the `--min-turns 2 --max-turns 5` random sampling distribution.
- `Approx Cache Hit: 58.1%`: About 58% of input tokens came from conversation history.

### 2. Using share_gpt_zh_multi_turn (Real Chinese Conversations)

Use case: Evaluate service performance using a realistic user conversation distribution, better reflecting production environment behavior.

```bash
evalscope perf \
  --model Qwen2.5-7B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
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
  --model Qwen2.5-7B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
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
                                    Detailed Performance Metrics
┏━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃      ┃      ┃      ┃     Avg ┃     P99 ┃     Avg ┃     P99 ┃     Avg ┃    P99 ┃    Gen. ┃ Success┃
┃Conc. ┃ Rate ┃  RPS ┃ Lat.(s) ┃ Lat.(s) ┃ TTFT(s) ┃ TTFT(s) ┃ TPOT(s) ┃ TPOT(… ┃  toks/s ┃    Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│   10 │  INF │ 3.87 │   2.571 │   4.103 │   0.055 │   0.098 │   0.010 │  0.013 │  985.21 │  100.0%│
└──────┴──────┴──────┴─────────┴─────────┴─────────┴─────────┴─────────┴────────┴─────────┴────────┘

                                          Request Metrics
┏━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃      ┃          ┃             ┃     P99 In ┃     Avg Out ┃    P99 Out ┃         Avg ┃      Approx┃
┃Conc. ┃ Num Reqs ┃ Avg In Toks ┃       Toks ┃        Toks ┃       Toks ┃   Turns/Req ┃   Cache Hit┃
│   10 │      300 │       428.7 │      912.0 │       186.3 │      384.0 │        1.98 │       53.7%│
└──────┴──────────┴─────────────┴────────────┴─────────────┴────────────┴─────────────┴────────────┘
```

**Interpreting the metrics**:

- `Avg Turns/Req: 1.98`: Limited by `--max-turns 3`, each request carried approximately 2 turns of context on average, as expected (turn 1 has no history, turns 2 and 3 carry 1 and 2 turns of history respectively, averaging ~1.98).
- `Approx Cache Hit: 53.7%`: Real conversations have longer context; history tokens account for ~54% of input.

### 3. Using custom_multi_turn (Custom Local Conversations)

Use case: You have conversation data already in OpenAI messages format and want to benchmark directly without any format conversion.

First, prepare the JSONL data file (one conversation per line):

```json
[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help you?"}, {"role": "user", "content": "Write me a poem"}, {"role": "assistant", "content": "Sure, ..."}, {"role": "user", "content": "Write another one"}]
[{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}, {"role": "user", "content": "What are some famous landmarks there?"}]
```

Then run the benchmark:

```bash
evalscope perf \
  --model Qwen2.5-7B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
  --api openai \
  --dataset custom_multi_turn \
  --dataset-path /path/to/my_conversations.jsonl \
  --max-tokens 512 \
  --multi-turn \
  --max-turns 3 \
  --number 100 \
  --parallel 10
```