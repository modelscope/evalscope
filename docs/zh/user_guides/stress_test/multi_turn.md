# 多轮对话压测

多轮对话压测功能允许用户在真实的多轮交互场景下测试模型服务。与普通压测不同，多轮模式会将模型的**真实回复**追加到上下文，后续每轮请求都携带完整的历史对话，从而真实模拟用户与模型的连续对话过程，并能评估服务在上下文不断增长时的延迟、吞吐量及 KV 缓存命中率表现。

## 功能特性

- **真实上下文累积**：每轮请求成功后，将模型实际输出追加到对话历史，下一轮请求携带完整历史，而非仅发送当前用户消息。
- **KV 缓存可命中率估算**：基于客户端 token 计数，估算每轮请求中历史对话 token 占总输入 token 的比例，即理论上可被服务端 prefix caching 利用的上限；实际是否命中取决于服务端是否开启并维持了对应缓存。
- **多数据集支持**：提供随机合成（`random_multi_turn`）和真实对话（`share_gpt_zh_multi_turn` / `share_gpt_en_multi_turn`）三种数据集。
- **参数语义一致**：`--number` 为总 turn 数，`--parallel` 为并发 turn 数，与普通压测模式语义保持一致。

## 参数说明

### 多轮专属参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--multi-turn` | `bool` | 启用多轮对话压测模式 | `False` |
| `--min-turns` | `int` | 每个对话最少用户轮数，仅 `random_multi_turn` 使用 | `1` |
| `--max-turns` | `int` | 每个对话最多用户轮数；`random_multi_turn` **必须设置**；ShareGPT 数据集可选，用于截断过长对话 | `None` |

### 已有参数在多轮模式下的语义

| 参数 | 多轮模式下的含义 |
|------|----------------|
| `--number` | 总发送 **turn 数**（而非对话数）；所有 worker 合计发送的 HTTP 请求数达到此值后停止 |
| `--parallel` | 同时在途的 turn 级并发请求数 |

## 数据集说明

### random_multi_turn

基于 `random` 数据集随机生成 token 序列，每个对话包含 `[min_turns, max_turns]` 轮用户消息，无需外部数据文件，适合快速基准测试和性能对比。

**必需参数**：`--tokenizer-path`、`--max-turns`

**可选参数**：`--min-turns`（默认 1）、`--min-prompt-length`、`--max-prompt-length`（控制每轮用户消息的 token 长度范围）

数据集产出的每条对话结构如下：

```json
[
  {"role": "user", "content": "...turn 1 随机 token 序列..."},
  {"role": "user", "content": "...turn 2 随机 token 序列..."}
]
```

> **注意**：`--tokenize-prompt` 在多轮模式下不支持，会被自动忽略。多轮对话始终通过 `/v1/chat/completions` 端点以消息列表形式发送。

### share_gpt_zh_multi_turn / share_gpt_en_multi_turn

使用来自 [swift/sharegpt](https://www.modelscope.cn/datasets/swift/sharegpt) 的真实对话数据（约 70k 条中文 / 英文对话），保留完整的 user + assistant 轮次交替结构，适合评测模型在真实对话分布下的表现。

- **数据自动下载**：未指定 `--dataset-path` 时，自动从 ModelScope 下载数据集
- **支持本地数据**：通过 `--dataset-path` 指定本地 JSONL 文件（每行一条 `conversation` 对象）
- **可选截断**：通过 `--max-turns` 限制每条对话最多使用的用户轮数

本地 JSONL 数据集格式（每行一条对话）：

```json
{"conversation": [{"human": "你好", "assistant": "你好！有什么可以帮助你？"}, {"human": "帮我写一首诗", "assistant": "好的，..."}]}
```

运行时上下文结构（第 2 轮发送时）：

```json
[
  {"role": "user",      "content": "你好"},
  {"role": "assistant", "content": "<模型第 1 轮实际回复>"},
  {"role": "user",      "content": "帮我写一首诗"}
]
```

> **说明**：数据集中的参考助手回复仅用于结构完整性，不会被直接发送给模型。运行时 worker 始终将模型的**实际输出**追加到上下文，保证历史准确。

## 工作流程


1. **加载对话**：从数据集插件预加载全部对话到内存池。
2. **启动 workers**：启动 `--parallel` 个异步协程，各自独立处理对话。
3. **循环取对话**：每个 worker 从对话池中循环取出对话（当所有对话被处理完后从头重新循环）。
4. **逐 turn 发送**：对每个用户轮次，将当前用户消息追加到上下文后发送请求；等待回复后，将模型真实输出追加到上下文，供下一轮使用。
5. **预算控制**：全局 turn 计数器达到 `--number` 时，所有 worker 停止。
6. **失败处理**：某轮请求失败时，当前对话被放弃，worker 继续取下一条对话开始新的对话。
7. **指标汇总**：汇总全部延迟、吞吐量及多轮专属指标后输出结果。

> **注意**：请求成功率低于 100% 时，被中断的对话不会贡献后续轮次的上下文，KV 缓存命中率等指标可能偏低。

## 输出指标说明

多轮模式在普通压测的全部指标基础上，额外输出以下两个专属指标：

| 指标 | 说明 | 如何解读 |
|------|------|----------|
| `Average input turns per request` | 每次请求时，上下文中平均包含的用户轮数 | 反映测试过程中上下文平均增长程度；值越大说明对话越深入，prompt 越长 |
| `Average approx KV cache hit rate (%)` | 估算当前请求中历史对话 token 占总输入 token 的比例（即理论上可被 prefix caching 利用的上限） | 比值越高说明历史上下文占比越大，若服务端开启了 prefix caching 则收益越显著；计算方式：`(前一轮 prompt_tokens + 前一轮 completion_tokens) / 当前轮 prompt_tokens × 100%`；第 1 轮（无历史）不计入统计 |

其余指标（Avg/P99 Latency、TTFT、TPOT、RPS、TPS）含义与[普通压测](./parameters.md)一致。

## 使用示例

### 1. 使用 random_multi_turn（合成多轮对话）

适用场景：快速评测服务在指定 prompt 长度分布和多轮深度下的性能，无需真实数据集。

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

输出示例：

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

**指标解读**：

- `Average input turns per request: 2.87`：测试期间每次请求平均携带 2.87 轮上下文，符合 `--min-turns 2 --max-turns 5` 的随机采样分布。
- `Average approx KV cache hit rate (%): 41.23`：平均约 41% 的输入 token 来自 KV 缓存（历史对话），服务端 prefix caching 有一定收益；若命中率接近 0，建议检查服务是否开启 prefix caching。

### 2. 使用 share_gpt_zh_multi_turn（真实中文对话）

适用场景：使用真实用户对话分布评测服务，反映更贴近生产环境的性能。

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
