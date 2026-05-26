# Agentic Trace 重放压测

重放生产环境真实 agentic 工作负载的 token 长度序列，用于测试推理服务在多轮、长尾、含 tool-call 等待的 agent 场景下的性能。数据来自 [applied-compute/trie](https://github.com/applied-compute/trie)（Apache-2.0），evalscope 把三个 workload 重新发布在 [ModelScope dataset `evalscope/trie-workloads`](https://www.modelscope.cn/datasets/evalscope/trie-workloads) 上，使用时自动下载。

## 为什么 workload jsonl 不带真实对话内容

workload jsonl 每一行只有 token 长度元数据，没有真实文本：

```json
{
  "input_prompt_length": 8749,
  "num_turns": 24,
  "assistant_response_length": [85, 46, 220, ...],
  "tool_call_output_length": [664, 664, 32, ...],
  "tool_call_latency": [37.197, 1.56, 0.814, ...],
  "final_assistant_response_length": 1658
}
```

测试时客户端按这些目标长度**合成假 prompt**（用随机 token 凑到精确字节长度）。这样做的原因：

- **隐私 / IP**：trie 数据来源于真实生产用户 trace，文本会泄露用户对话和应用业务逻辑
- **跨模型可比**：测试目标是「服务器处理 8k input + N 个几千 token tool output 多轮对话」的系统性能，不是「服务器对某段具体业务文本的语义响应质量」，用合成 prompt 把语义噪声标准化掉
- **License 干净**：长度数字不涉及版权 / 训练数据问题，能 Apache-2.0 公开

合成方式与 `random_dataset` 相同（从词表 uniform sample → decode → re-encode 修复到精确长度），区别只在 **trie 的长度序列来自真实生产 trace 而非用户指定的分布**。

## 快速开始

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

跑完后输出会出现两张多轮专属表（除了已有的 Detailed Performance / Request Metrics）：

- **Per-trace Summary**：每条 trace 的 6 个指标（Latency / First-Turn TTFT / TTFAT / Decode TPS / Cache Hit / Eligible Cache Hit）做 mean/min/p50/p90/p95/p99/max 汇总
- **Workload Throughput**：四类 token/s（Total / New / Cached prompt + Completion）在三个时间窗下（Overall / Last 30s / Steady drop-20%）的分布

同时输出目录会多三个文件：`trace_summary.json` / `workload_throughput.json` / `workload_timeline.json`（最后一个是 raw cumulative-token 时间序列，pandas 友好）。

## 三个内置 workload

| 数据集别名 | 来源 jsonl | 典型 num_turns | 适用场景 |
|---|---|---|---|
| `trie_agentic_coding` | `agentic_coding_8k.jsonl` | 8-15 | coding agent 类应用（IDE 代码补全、refactor agent） |
| `trie_code_qa` | `code_qa_8k.jsonl` | 5-12 | 代码问答 agent（仓库分析、文档生成） |
| `trie_office_work` | `office_work_8k.jsonl` | 18-41 | 办公自动化 agent（最长、最重，适合压力上限测试） |

每条 trace 默认初始 prompt 约 8k token，整体 input 量级 25k-50k token / trace。所有 trace 加在一起每个文件约 8000 条，足够长时间跑出统计意义。

想用自己的 trace？把同样格式的 jsonl 喂给 `--dataset-path` 即可：

```bash
evalscope perf ... --dataset trie_office_work --dataset-path /path/to/your_traces.jsonl
```

## 输出指标怎么读

### Per-trace Summary

| 列 | 含义 | 这个数字异常时该看哪里 |
|---|---|---|
| **Latency (s)** | trace 第一 turn `start_time` → 最后一 turn `completed_time` 的墙钟 | 单 trace 跑得慢 → 看 TTFAT 拆分（是 prefill 慢还是 decode 慢） |
| **First-Turn TTFT (s)** | 第一 turn 的 TTFT；反映 8k 冷 prefill 性能 | 高 → 服务端 prefill 阶段慢，KV cache 不存在时第一 token 就要算完 8k attention |
| **TTFAT (s)** | trace 起点 → 末 turn 首 token 的墙钟，端到端「等多久能看到最终回复」 | trace 中间任何 turn 慢都会拉高 TTFAT |
| **Decode TPS** | trace 内每 turn `(completion-1)/(latency-ttft)` 的算术平均 | 反映 decode 阶段稳态速度；与服务端单 token 生成开销直接相关 |
| **Cache Hit Rate (%)** | `sum(cached_tokens) / sum(prompt_tokens)`，含 turn 1（贡献 0 但计入分母） | 低 → 服务端 prefix cache 没开启或 evict 太激进；首 turn 贡献 0 是设计如此（无前文可 cache） |
| **Eligible Cache Hit Rate (%)** | 分母只算「理论上应能 cache 的 prefix」（`prev_prompt + prev_completion`），剔除首 turn 与当前 turn 新增 tool 输出 | 与 Cache Hit Rate 接近且都低 → 服务端没开 cache；前者高、后者低 → cache 开了但容量不够 |

汇总列（mean / min / p50 / p90 / p95 / p99 / max）跨所有完成的 trace 算分位数。

### Workload Throughput

```
┌─────────────────────┬───────────┬────────────┬─────────────────────┐
│ Metric (tok/s)      │   Overall │   Last 30s │   Steady (drop 20%) │
├─────────────────────┼───────────┼────────────┼─────────────────────┤
│ Total Prompt tok/s  │   ...     │   ...      │            ...      │
│ New Prompt tok/s    │   ...     │   ...      │            ...      │
│ Cached Prompt tok/s │   ...     │   ...      │            ...      │
│ Completion tok/s    │   ...     │   ...      │            ...      │
└─────────────────────┴───────────┴────────────┴─────────────────────┘
```

| 列 | 含义 |
|---|---|
| **Overall** | 总 tokens / wall_time，覆盖整段运行 |
| **Last 30s** | 末 30 秒滑窗 tok/s；运行短于 30 秒时回落到 Overall（避免误导） |
| **Steady (drop 20%)** | 丢前 20% wall_time 后剩余区间的 tok/s；剔除服务端冷启动 / batch scheduler ramp / KV cache 填充阶段的影响 |

| 行 | 含义 |
|---|---|
| Total Prompt | New Prompt + Cached Prompt |
| New Prompt | `prompt_tokens - cached_tokens`，服务端实际算 attention 的部分 |
| Cached Prompt | 服务端从 prefix cache 命中跳过 prefill 的部分 |
| Completion | 输出 token 速率 |

Steady-state 是评估「这个 endpoint 稳定状态下能跑多快」的最公平数字；Last 30s 适合看尾部抖动 / 长 trace 收尾阶段行为。

### First-Turn vs Subsequent-Turn TTFT

Detailed Performance 表里的 `Avg TTFT` 是所有 turn 的均值。在多轮 trace 重放场景下，第一 turn 是冷 prefill（8k+ input 全算），后续 turn 大量命中 prefix cache，两类 TTFT 量级可能差 2-10 倍。Request Metrics 区域会出现：

- `First-Turn TTFT (ms)`
- `Subsequent-Turn TTFT (ms)`

只在多轮路径下出现；单轮压测两列都不显示。**冷热混算的 `Avg TTFT` 看起来居中，跟用户感知的"首 token 多快"经常对不上**，分桶后才能看清。

## `--duration` 软退出 + `--number` 配合

trace 重放支持三种 cap 组合：

| 命令 | 含义 |
|---|---|
| `--number 50` | 跑 50 个 trace 就停，不限时 |
| `--duration 300` | 跑 300 秒就停，trace 跑多少看运气 |
| `--number 50 --duration 300` | 50 trace 上限 + 300s 上限，**whichever 先到就停** |

**推荐第三种**：limit 控制 sample size 让横向对比可重复，duration 兜底防止慢 endpoint 拖太久。

软退出语义（与上游 trie 一致）：deadline 到点后**不再启动新 trace**，但**已经在跑的 trace 会跑完所有 turn**才退出。这保证 `trace_summary.json` 里每个 trace 都是完整的，不会出现"被砍半"的 trace 拉偏 latency / cache hit 统计。

副作用：实际 wall_time 可能略大于 `--duration`（多出来的就是 in-flight trace 跑完所需的时间）。trace 越长这个超出越明显，office_work 平均 trace 60-150s，可能超出几十秒到 2 分钟。

## 后端要求与常见问题

### `ignore_eos` 后端依赖

trie workload 的 `assistant_response_length` 是从生产 trace 抓的真实输出长度，重放时希望模型**严格生成到这个长度**才能复现原 trace 的负载。这需要服务端支持 `extra_body.ignore_eos = True`，目前只有：

- vLLM（默认支持）
- SGLang（默认支持）
- TensorRT-LLM（需特定路径）

DashScope 兼容模式 / OpenAI 官方 API / 大多数云端推理服务**不实现 `ignore_eos`**，模型遇到自然 EOS 就停。这意味着实际跑出的 completion 长度会**短于** `assistant_response_length`，Completion tok/s 会被低估。这不影响 cache hit / decode TPS / latency 算法层正确性，但 tok/s 数字不能直接跟 trie 上游做绝对值对比。

### Chat template overhead

evalscope multi-turn 走 `/v1/chat/completions`，服务端会用模型的 chat template 把 messages 数组序列化成最终 token 序列。chat template 通常加 10-50 个 token / turn（system message + role marker + special token）。所以同样的 trace，evalscope 看到的 prompt_tokens 会比 trie（原本走 `/v1/completions` raw text）多几个百分点，Cache Hit Rate 数字会被分母拉低 2-3pp。这是接口路径选择带来的固有差异，跟算法无关。

### 合成 prompt 的字符差异

evalscope 在合成 prompt 时**主动排除特殊 token 和 byte-fallback token**（避免 CJK 多字节字符引发 3-5× token 膨胀）。上游 trie 的 `TokenizerManager` 不做这个过滤。结果：

- 同一 trace 用 evalscope 跑出来的 prompt 是"干净" ASCII-like 字符
- 用 trie 跑出来的 prompt 可能含特殊符号 / 控制字符

**chat 模型对这两种输入的响应完全不同**（前者倾向短回复，后者倾向 hallucinate 长回复），所以即使两边都不让 `ignore_eos` 生效，输出长度也会不同。

## 与上游 trie 数字对齐

本节给出在同 endpoint（DashScope qwen-plus）、同 workload（office_work，5 traces）、同时启动条件下，evalscope 与 trie（已 patch 成 chat-completions + real chat history + `temperature=0` + `limit=5`）的实测对比。

### 算法层（按 trace 汇总）

| 指标 | trie | evalscope | 偏差 |
|---|---|---|---|
| Latency mean (s) | 133.5 | 136.3 | **+2.1%** |
| TTFAT mean (s) | 118.6 | 117.4 | **-1.0%** |
| Decode TPS | 42.13 | 41.80 | **-0.8%** |
| Cache Hit (per-trace mean) | 32.8% | 34.7% | **+5.8%** |
| Eligible Cache (per-trace mean) | 34.6% | 36.6% | **+5.8%** |

**算法对齐 ±6%**。剩余差异主要来自 5-trace 的统计样本方差，跟 evalscope 实现无关。

### Workload 级（按时间汇总）

| 指标 | trie | evalscope | 偏差 |
|---|---|---|---|
| Total Prompt tok/s | 14777 | 14102 | -4.6% |
| Cached Prompt tok/s | 13999 | 5338 | -62% |
| Completion tok/s | 108.7 | 75.9 | -30% |

Workload 级三行数字差异较大，**这不是算法 bug**，根因清晰：

1. **Total Prompt -4.6%**：基本一致，量级对齐
2. **Cached Prompt -62% / New Prompt 大幅反向**：trie 自己 per-trace cache hit (32.8%) 与 workload-level (94.7%) 内部就矛盾，trie 的 workload 聚合口径与 per-trace 不一致；evalscope 两套口径自洽（per-trace 34.7% ≈ workload 37.8%）。**应以 per-trace 为准对齐**
3. **Completion -30%**：trie 与 evalscope 合成的 prompt 文本完全不同（词表过滤策略不同），chat 模型对不同字符分布给不同长度回复，这是合成 prompt 的字符差异 + `ignore_eos` 不生效共同造成

### 想做严格 ±5% 对齐怎么办

需要切到支持 `ignore_eos` 的本地后端（vLLM / SGLang），两边都跑 raw `/v1/completions` 或都跑 `/v1/chat/completions`、消除合成 prompt 的字符差异。在那种环境下 Cached / Completion / Throughput 等所有数字都能压到 ±5% 内。当前 DashScope 等云端推理服务无法满足，**只能在算法层做对齐验证**，这就是本节给出的 ±6% 那一组数字。
