# 多轮对话压测

多轮对话压测功能允许用户在真实的多轮交互场景下测试模型服务。与普通压测不同，多轮模式会将模型的**真实回复**追加到上下文，后续每轮请求都携带完整的历史对话，从而真实模拟用户与模型的连续对话过程，并能评估服务在上下文不断增长时的延迟、吞吐量及 KV 缓存命中率表现。

## 功能特性

- **真实上下文累积**：每轮请求成功后，将模型实际输出追加到对话历史，下一轮请求携带完整历史，而非仅发送当前用户消息。
- **KV 缓存可命中率估算**：基于客户端 token 计数，估算每轮请求中历史对话 token 占总输入 token 的比例，即理论上可被服务端 prefix caching 利用的上限；实际是否命中取决于服务端是否开启并维持了对应缓存。
- **多数据集支持**：提供随机合成（`random_multi_turn`）、真实对话（`share_gpt_zh_multi_turn` / `share_gpt_en_multi_turn`）、自定义本地数据（`custom_multi_turn`）、真实 Agent 轨迹（`swe_smith`）和生产 agentic trace 重放（`trie_agentic_coding` / `trie_code_qa` / `trie_office_work`）等数据集。
- **参数语义一致**：`--number` 为总对话数，`--parallel` 为并发对话数，与普通压测模式语义保持一致。

## 参数说明

### 多轮专属参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--multi-turn` | `bool` | 启用多轮对话压测模式 | `False` |
| `--min-turns` | `int` | 每个对话最少用户轮数，仅 `random_multi_turn` 使用 | `1` |
| `--max-turns` | `int` | 每个对话最多用户轮数；`random_multi_turn` **必须设置**；ShareGPT / `custom_multi_turn` 等数据集可选，用于截断过长对话；`swe_smith` live 构建时每条对话轮次从 `[min_turns, max_turns]` 随机采样 | `None` |
| `--dataset-offset` | `int` | 跳过数据集前 N 条对话，用于分片测试或避免缓存命中 | `0` |

### `multi_turn_args`（`swe_smith` 专属参数）

`swe_smith` 数据集的 live 构建模式支持通过 `MultiTurnArgs` 对象精细控制对话 token 长度目标。

每条对话的轮次数量从 `[--min-turns, --max-turns]` 中随机采样，每轮按以下 token 长度参数填充消息。

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `first_turn_length` | `int` 或 `[int, int]` | 第 1 轮目标 prompt token 数；从原始轨迹中截取恰好达到该长度的消息片段<br>• 单个整数：固定值，如 `65000`<br>• 两个整数的列表：`[最小值, 最大值]`，每条对话独立均匀随机采样，如 `[32000, 65000]` | `65000` |
| `subsequent_turn_length` | `int` 或 `[int, int]` | 后续每轮在上一轮基础上新增的目标 token 数；决定每轮 delta 的大小<br>• 单个整数：固定值<br>• 两个整数的列表：`[最小值, 最大值]`，每条对话独立均匀随机采样 | `500` |
| `chars_per_token` | `float` | 无 tokenizer 时的字符/token 估算比，用于预过滤原始轨迹 | `3.0` |
| `num_workers` | `int` | live 构建模式下并行 worker 数量（>1 使用 multiprocessing.Pool） | `4` |

### 已有参数在多轮模式下的语义

| 参数 | 多轮模式下的含义 |
|------|----------------|
| `--number` | 总 **对话数**；所有 worker 合计完成的对话数达到此值后停止 |
| `--parallel` | 同时并发执行的对话数（每个 worker 持有一条对话） |

## 工作流程

1. **加载对话池**：启动时从数据集文件顺序读取，将所有对话预加载到内存，最多加载 `--number` 条（防止大数据集占用过多内存）。

2. **启动 workers**：创建 `--parallel` 个并发协程（worker），各自独立运行，所有 worker 共享同一个全局对话计数器。

3. **对话分配（先到先得，顺序轮询）**：对话按顺序分配给 worker，谁先完成当前对话，谁先取下一条。所有对话用完后从头循环复用。

   以 `--parallel 3`、共 5 条对话为例：

   ```
   启动时：3 个 worker 同时开始

   Worker 0 → 取第 1 条对话
   Worker 1 → 取第 2 条对话
   Worker 2 → 取第 3 条对话

   ↓ 各自发送请求（等待回复期间其他 worker 可以继续执行）

   假设 Worker 1 最先跑完第 2 条对话的所有轮次：
   Worker 1 → 取第 4 条对话

   假设 Worker 0 第二个跑完第 1 条对话：
   Worker 0 → 取第 5 条对话

   第 5 条用完后若还未达到 --number：
   Worker 0 → 循环回到第 1 条对话重新开始

   ... 以此类推，谁先跑完当前对话，谁先取下一条
   ```

4. **单条对话执行（逐 turn）**：每个 worker 持有自己独立的对话历史，不与其他 worker 共享：

   ```
   第 1 轮：发送 [用户消息1]                                    → 收到模型回复1
   第 2 轮：发送 [用户消息1, 模型回复1, 用户消息2]              → 收到模型回复2
   第 3 轮：发送 [用户消息1, 模型回复1, 用户消息2, 模型回复2, 用户消息3] → ...
   ```

   每轮将用户消息追加到历史后发送请求，收到模型的**真实回复**后追加到历史，供下一轮携带。数据集中原有的 assistant 内容不会发给模型，仅用模型实际输出构建上下文。

5. **预算控制与停止**：全局对话计数在每条对话开始前同步递增，确保所有 worker 合计完成的对话数不超过 `--number`。一旦达到上限，所有 worker 停止，不再取新对话。

6. **失败处理**：某轮请求失败时，当前对话立即放弃，worker 取下一条新对话重新开始，不会将失败的上下文带入后续请求。

7. **指标汇总**：所有 worker 完成后，汇总全部延迟、吞吐量及多轮专属指标（平均上下文轮数、KV 缓存命中率）后输出结果。

> **注意**：请求成功率低于 100% 时，被中断的对话不会贡献后续轮次的上下文，KV 缓存命中率等指标可能偏低。

## 数据集

evalscope 提供以下多轮数据集：

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

**使用示例**：适用场景：快速评测服务在指定 prompt 长度分布和多轮深度下的性能，无需真实数据集。

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

输出示例：

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

**指标解读**：

- `Turns/Req: 1.60`：测试期间每次请求平均携带 1.60 轮上下文，符合 `--min-turns 2 --max-turns 5` 的随机采样分布（第 1 轮无历史，后续轮次历史逐步增长）。
- `Cache Hit (%): 58.1%`：约 58% 的输入 token 来自历史对话。

### share_gpt_multi_turn

包含两个数据集：`share_gpt_zh_multi_turn`（中文）与 `share_gpt_en_multi_turn`（英文）。使用来自 [swift/sharegpt](https://www.modelscope.cn/datasets/swift/sharegpt) 的真实对话数据（约 70k 条中文 / 英文对话），保留完整的 user + assistant 轮次交替结构，适合评测模型在真实对话分布下的表现。

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

**使用示例**：适用场景：使用真实用户对话分布评测服务，反映更贴近生产环境的性能。

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

若已本地下载数据集，可通过 `--dataset-path` 避免重复下载：

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

输出示例：

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

**指标解读**：

- `Turns/Req: 1.98`：受 `--max-turns 3` 限制，平均每请求包含约 2 轮上下文，符合预期（第 1 轮无历史，第 2、3 轮分别携带 1、2 轮历史，平均约 1.98）。
- `Cache Hit (%): 53.7%`：真实对话中上下文较长，历史 token 占比约 54%。

### custom_multi_turn

使用本地 JSONL 文件作为自定义多轮对话数据集，每行直接以 **OpenAI messages 格式**存储一条完整对话，无需任何格式转换，适合已有对话数据直接压测的场景。

- **必须指定** `--dataset-path` 指向本地 JSONL 文件
- **可选截断**：通过 `--max-turns` 限制每条对话最多使用的用户轮数

**JSONL 数据集格式**（每行一条对话，为 OpenAI messages 数组）：

```json
[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助你？"}, {"role": "user", "content": "帮我写一首诗"}]
[{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}, {"role": "user", "content": "Tell me more about it."}]
```

每行需满足：
- 必须是一个 JSON 数组
- 每个元素必须包含 `role` 和 `content` 字段
- `role` 取值为 `user` 或 `assistant`
- 至少包含一个 `user` 消息

运行时上下文结构（第 2 轮发送时）：

```json
[
  {"role": "user",      "content": "你好"},
  {"role": "assistant", "content": "<模型第 1 轮实际回复>"},
  {"role": "user",      "content": "帮我写一首诗"}
]
```

> **说明**：数据集中的 `assistant` 消息仅用于标识对话结构，**不会**被直接发送给模型。运行时 worker 始终将模型的实际输出追加到上下文，保证历史准确。

**使用示例**：适用场景：已有 OpenAI messages 格式的对话数据，直接用于多轮压测，无需转换格式。

首先准备 JSONL 数据文件（每行一条对话）：

```json
[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助你？"}, {"role": "user", "content": "帮我写一首诗"}, {"role": "assistant", "content": "好的，..."}, {"role": "user", "content": "再写一首"}]
[{"role": "user", "content": "介绍一下北京"}, {"role": "assistant", "content": "北京是中国的首都..."}, {"role": "user", "content": "有哪些著名景点？"}]
```

然后运行压测：

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

使用来自 [SWE-bench/SWE-smith-trajectories](https://www.modelscope.cn/datasets/SWE-bench/SWE-smith-trajectories) 的真实 Agent 代码修复轨迹数据，专为**长上下文 + 多轮 Agent 场景**压测设计。每条轨迹由工具调用、代码片段、Patch 结果等组成，单条 prompt 通常超过数万 token，适合评测模型在大上下文下的 prefill 吞吐和 KV 缓存命中率。

支持两种数据源模式：

- **预构建 JSON 模式**（推荐）：指定 `--dataset-path` 加载已生成的 `agentic_dataset.json`，无需 tokenizer，启动快。
- **Live 构建模式**（不指定 `--dataset-path`）：运行时从 ModelScope 拉取原始轨迹并动态构建对话，**必须**指定 `--tokenizer-path` 用于精确 token 计数。

两种模式共同特性：
- **支持偏移**：通过 `--dataset-offset` 跳过前 N 条对话，用于分片测试或规避缓存热点

> **说明**：每条对话的轮次数量从 `[--min-turns, --max-turns]` 中随机采样，每轮填充的消息量由 `first_turn_length` / `subsequent_turn_length` 决定。

**预构建数据集生成**

推荐在压测前使用 `examples/perf/build_swe_smith_dataset.py` 脚本预先构建 `agentic_dataset.json`，一次生成、多次复用，避免每次压测都重复下载和构建。

**主要参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 用于精确 token 计数的 tokenizer 路径（ModelScope 模型 ID 或本地路径） | `Qwen/Qwen2.5-7B-Instruct` |
| `--first-turn-length` | 第 1 轮目标 prompt token 数 | `65000` |
| `--subsequent-turn-length` | 后续每轮新增的目标 token 数 | `500` |
| `--min-turns` | 每条对话最少轮数 | `1` |
| `--max-turns` | 每条对话最多轮数；实际轮数从 `[min_turns, max_turns]` 随机采样（与 live 构建模式行为一致）；未设置时等于 `--min-turns` | `None` |
| `--number` | 要生成的对话条数 | `128` |
| `--output-path` | 输出文件路径 | `agentic_dataset.json` |
| `--seed` | 随机种子，保证可复现 | `42` |
| `--num-workers` | 并行 worker 数量 | CPU 核心数 |

```bash
python examples/perf/build_swe_smith_dataset.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --first-turn-length 8192 \
  --subsequent-turn-length 1024 \
  --min-turns 3 \
  --max-turns 8 \
  --number 128 \
  --output-path outputs/agentic_dataset.json \
  --seed 42 \
  --num-workers 8
```

**使用示例：预构建 JSON 模式（推荐）**

预先生成 `agentic_dataset.json` 后，通过 `--dataset-path` 加载，无需 tokenizer，启动更快：

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

> **说明**：`--dataset-offset` 可跳过数据集前 N 条对话，适合多机分片压测或规避 KV 缓存热点。

**使用示例：Live 构建模式**

自动从 ModelScope 拉取 SWE-smith-trajectories 数据集并实时构建对话，需指定 `--tokenizer-path`：

```bash
evalscope perf \
  --model YOUR_MODEL \
  --url OPENAI_API_COMPAT_URL \
  --api openai \
  --dataset swe_smith \
  --tokenizer-path YOUR_MODEL \
  --max-tokens 512 1024 \
  --min-tokens 512 \
  --multi-turn \
  --multi-turn-args '{
      "first_turn_length": [4096, 8192],
      "subsequent_turn_length": [512, 1024]
  }' \
  --min-turns 3 \
  --max-turns 8 \
  --seed 42 \
  --number 10 20 \
  --parallel 5 10 \
  --extra-args '{"ignore_eos": true}'
```

### trie trace 重放

重放生产环境真实 agentic 工作负载的 token 长度序列，用于测试推理服务在多轮、长尾、含 tool-call 等待的 agent 场景下的性能。数据来自 [applied-compute/trie](https://github.com/applied-compute/trie)（Apache-2.0），evalscope 把三个 workload 重新发布在 [ModelScope dataset `evalscope/trie-workloads`](https://www.modelscope.cn/datasets/evalscope/trie-workloads) 上，使用时自动下载。

每条 trace 只包含 token 长度元数据（每轮的 prompt 长度、输出长度、tool-call 等待时间等），不含真实对话文本。运行时客户端按这些长度自动合成 prompt，保证负载特征与生产 trace 一致。

**三个内置 workload**：

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

**必需参数**：`--tokenizer-path`（用于合成精确长度的 prompt）

**使用示例**：

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

> **注意**：trie trace 重放依赖 `ignore_eos` 让模型严格生成到录制长度。vLLM 和 SGLang 支持此参数；DashScope / OpenAI API 等云端服务不支持，模型遇到自然 EOS 就停，实际输出会短于录制长度，但不影响 latency / cache hit / decode TPS 的正确性。

## 多轮专属输出指标

除了 Performance Overview 和 Per-Request Metrics 两张表外，多轮模式还会额外输出以下表格。

### Per-Trace Metrics

按对话（trace）维度汇总的指标，跨所有完成的对话算 mean / p50 / p90 / p99 / max：

| 列 | 含义 | 这个数字异常时该看哪里 |
|---|---|---|
| **Latency (s)** | 对话第一 turn 开始 → 最后一 turn 完成的墙钟 | 单对话跑得慢 → 看 TTFAT 拆分（是 prefill 慢还是 decode 慢） |
| **First-Turn TTFT (s)** | 第一 turn 的 TTFT；反映冷 prefill 性能 | 高 → 服务端 prefill 阶段慢 |
| **TTFAT (s)** | 对话起点 → 末 turn 首 token 的墙钟，端到端「等多久能看到最终回复」 | 对话中间任何 turn 慢都会拉高 TTFAT |
| **Decode TPS** | 对话内每 turn `(completion-1)/(latency-ttft)` 的算术平均 | 反映 decode 阶段稳态速度 |
| **Cache Hit Rate (%)** | `sum(cached_tokens) / sum(prompt_tokens)`，含 turn 1（贡献 0 但计入分母） | 低 → 服务端 prefix cache 没开启或 evict 太激进 |
| **Eligible Cache Hit Rate (%)** | 分母只算「理论上应能 cache 的 prefix」，剔除首 turn 与当前 turn 新增内容 | 与 Cache Hit Rate 接近且都低 → 服务端没开 cache；前者高、后者低 → cache 开了但容量不够 |

同时输出 `trace_summary.json` 到结果目录。

### Workload Throughput

按时间维度汇总的 token 吞吐率，所有模式（单轮 / 多轮）均输出：

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

| 列 | 含义 |
|---|---|
| **Overall** | 总 tokens / wall_time，覆盖整段运行 |
| **Last 30s** | 末 30 秒滑窗 tok/s；运行短于 30 秒时回落到 Overall |
| **Steady (drop 20%)** | 丢前 20% wall_time 后剩余区间的 tok/s；剔除服务端冷启动阶段的影响 |

| 行 | 含义 |
|---|---|
| Total Prompt | New Prompt + Cached Prompt |
| New Prompt | `prompt_tokens - cached_tokens`，服务端实际算 attention 的部分 |
| Cached Prompt | 服务端从 prefix cache 命中跳过 prefill 的部分 |
| Completion | 输出 token 速率 |

Steady-state 是评估「这个 endpoint 稳定状态下能跑多快」的最公平数字；Last 30s 适合看尾部抖动。同时输出 `workload_throughput.json` 和 `workload_timeline.json`（raw cumulative-token 时间序列，pandas 友好）到结果目录。

### First-Turn vs Subsequent-Turn TTFT

Per-Request Metrics 表里的 `TTFT (ms)` 是所有 turn 的均值。在多轮场景下，第一 turn 是冷 prefill，后续 turn 大量命中 prefix cache，两类 TTFT 量级可能差 2-10 倍。Per-Request Metrics 表会额外出现：

- `First-Turn TTFT (ms)`
- `Subsequent-Turn TTFT (ms)`

只在多轮路径下出现；单轮压测不显示。

## `--duration` 软退出

多轮模式下 `--duration` 采用软退出语义：deadline 到点后**不再启动新对话**，但**已经在跑的对话会跑完所有 turn** 才退出。这保证每条对话都是完整的，不会出现被截断的对话拉偏统计。

三种 cap 组合：

| 命令 | 含义 |
|---|---|
| `--number 50` | 跑 50 个对话就停，不限时 |
| `--duration 300` | 跑 300 秒就停，对话跑多少看运气 |
| `--number 50 --duration 300` | 两个上限，**先到先停**（推荐） |

副作用：实际 wall_time 可能略大于 `--duration`（多出来的就是 in-flight 对话跑完所需的时间）。

## 常见问题

### Chat template 带来的 token 差异

多轮走 `/v1/chat/completions` 端点时，chat template 每轮会额外增加 10-50 个 token（role marker、special token 等），导致 Cache Hit Rate 比 raw completions 接口低 2-3pp，这是正常现象。
