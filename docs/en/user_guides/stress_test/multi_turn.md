# Multi-turn Conversation Benchmark

The multi-turn conversation benchmark allows you to test a model service in realistic multi-turn interaction scenarios. Unlike a standard benchmark, multi-turn mode appends the model's **actual replies** to the conversation context so that every subsequent request carries the full history — faithfully simulating a user's continuous conversation with the model, and enabling measurement of latency, throughput, and KV-cache utilization as context grows.

## Features

- **Real context accumulation**: After each successful turn, the model's actual output is appended to the conversation history; the next turn sends the complete history rather than just the current user message.
- **Approx KV cache hit rate estimation**: Based on client-side token counts, estimates the proportion of history tokens relative to the total input tokens in each request — i.e., the theoretical upper bound of tokens that could benefit from server-side prefix caching. Whether caching actually occurs depends on whether the server has prefix caching enabled and has retained the relevant cache.
- **Multiple dataset support**: Provides three datasets — random synthetic (`random_multi_turn`) and real conversations (`share_gpt_zh_multi_turn` / `share_gpt_en_multi_turn`).
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

## Workflow

1. **Load conversations**: Pre-load all conversations from the dataset plugin into an in-memory pool.
2. **Start workers**: Launch `--parallel` async coroutines, each processing conversations independently.
3. **Cycle through conversations**: Each worker pulls conversations from the pool in a round-robin fashion (restarting from the beginning once all conversations have been processed).
4. **Send turn by turn**: For each user turn, append the current user message to the context and send the request; after receiving the reply, append the model's actual output to the context for use in the next turn.
5. **Budget control**: Once the global turn counter reaches `--number`, all workers stop.
6. **Failure handling**: If a turn request fails, the current conversation is abandoned and the worker immediately starts a new conversation.
7. **Metrics aggregation**: Aggregate all latency, throughput, and multi-turn-specific metrics and output the results.

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
- `Approx Cache Hit: 58.1%`: About 58% of input tokens came from conversation history. If the server has prefix caching enabled, these tokens can be served from cache; if the hit rate is close to 0, check whether prefix caching is enabled on the server.

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
