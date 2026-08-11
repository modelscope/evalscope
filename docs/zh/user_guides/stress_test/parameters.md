# 参数说明

执行 `evalscope perf --help` 可获取全部参数说明。

## 基本设置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model` | `str` | 测试模型名称，或模型路径 | - |
| `--url` | `str` | API地址，支持`/chat/completions`、`/completions`和`/responses`端点 | - |
| `--name` | `str` | wandb/swanlab数据库结果名称和结果数据库名称 | `{model_name}_{current_time}` |
| `--api` | `str` | 服务API类型<br>• `openai`: OpenAI兼容Chat Completions API（需提供`--url`）<br>• `openai_responses`: OpenAI官方Responses API<br>• `openai_embedding`: OpenAI兼容Embedding API<br>• `openai_rerank`: OpenAI/Cohere兼容Rerank API<br>• `local`: 启动本地transformers推理<br>• `local_vllm`: 启动本地vLLM推理服务<br>• 自定义：参考[自定义API指南](./custom.md/#自定义请求-api) | - |
| `--port` | `int` | 本地推理服务端口<br>仅对`local`和`local_vllm`有效 | `8877` |
| `--attn-implementation` | `str` | Attention实现方式<br>仅在`api=local`时有效 | `None`<br>（可选：`flash_attention_2`、`eager`、`sdpa`） |
| `--api-key` | `str` | API密钥 | `None` |
| `--debug` | `bool` | 是否输出调试信息 | `False` |

## 网络配置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--total-timeout` | `int` | 每个请求的总超时时间（秒） | `21600`（6小时） |
| `--connect-timeout` | `int` | 网络连接超时（秒） | `None` |
| `--read-timeout` | `int` | 网络读取超时（秒） | `None` |
| `--headers` | `str` | 额外的HTTP头<br>格式：`key1=value1 key2=value2`<br>将用于每个查询 | - |
| `--no-test-connection` | `bool` | 不发送连接测试，直接开始压测 | `False` |

## 请求控制

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--parallel` | `list[int]` | 并发请求的数量<br>可传入多个值，空格分隔 | `1` |
| `--number` | `list[int]` | 发出的请求总数量<br>可传入多个值（需与`parallel`一一对应） | `1000` |
| `--rate` | `float` | 请求调度速率（请求/秒）<br>• `-1`: 不做速率节流；在默认 closed-loop 模式下，请求会尽快调度，但实际同时在飞的 HTTP 请求数仍受 `--parallel` 限制，并不会一次性全部发送到服务端<br>• `> 0`: 请求按泊松到达模型调度，间隔服从均值为 `1/rate` 的指数分布，即**平均**每秒调度 `rate` 个请求 | `-1` |
| `--log-every-n-query` | `int` | 每N个查询记录日志 | `100` |
| `--stream` | `bool` | 是否使用SSE流输出<br>需要启用以测量TTFT（Time to First Token）指标 | `True` |
| `--sleep-interval` | `int` | 每次性能测试之间的休眠时间（秒）<br>避免过载服务器 | `5` |
| `--open-loop` | `bool` | 启用开放环路（open-loop）模式：<br>请求按 `--rate` 指定的速率发出，无论服务端是否已处理完之前的请求。<br>• `--rate` 变为扫描变量（支持多值）<br>• `--number` 须与 `--rate` 等长，表示每轮发出的请求总数<br>• `--parallel` 在此模式下被忽略（内部设为 -1 / INF）<br>详见[使用示例](./examples.md#open-loop-开放环路) | `False` |
| `--warmup-num` | `float` | 预热请求数量或比例：<br>• `0`：禁用预热（默认）<br>• `>= 1`：绝对数量，如 `--warmup-num 10` 表示预热 10 个请求<br>• `0 < value < 1`：比例模式，如 `--warmup-num 0.1` 表示预热数量为 `--number` 的 10%<br>预热请求使用与正式压测相同的并发/速率发送，但**不计入性能指标**<br>适用于消除冷启动影响（如 KV-cache 填充、JIT 编译等）<br>详见[使用示例](./examples.md#warmup-预热) | `0` |
| `--duration` | `float` | 单次压测的墙钟时间预算（秒）<br>软退出语义：到点后**不再启动新请求**，但**已经在飞行中的请求会跑完**才退出<br>多轮模式下的"已在飞"指的是**已经 claim 的 trace 跑完所有剩余 turn**（trace-level soft exit，与上游 trie 一致）<br>与 `--number` 同时设置时取**先达到的那个**为停止条件 | `None` |

```{tip}
**Closed-loop 模式（默认）** 与 **Open-loop 模式**（`--open-loop`）的参数行为对比：

| | Closed-loop（默认） | Open-loop（`--open-loop`） |
|---|---|---|
| **`--rate`** | 控制请求调度速率（`-1` 表示无 pacing，但仍受 `--parallel` 并发上限约束；`R` 为泊松到达均值） | 控制请求发出速率；**必须 > 0**；支持多值（如 `5 10 20`），每个值对应一轮独立压测 |
| **`--number`** | 每轮总请求数，与 `--parallel` 等长 | 每轮总请求数，须与 `--rate` **等长** |
| **`--parallel`** | 同时在飞行中的最大请求数；每个 worker 收到响应后才发下一条（**背压保护**） | **被忽略**，并发上限为无穷大（INF）；请求按调度立即发出，不等待响应 |
| **适用场景** | 测量服务在受控并发下的延迟与吞吐 | 模拟真实流量（请求到达与服务时间无关）；扫描多速率点的吞吐-延迟曲线 |
```

## SLA设置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--sla-auto-tune` | `bool` | 是否启用SLA自动调优模式 | `False` |
| `--sla-variable` | `str` | 自动调优的变量<br>可选：`parallel`（并发数）、`rate`（请求速率） | `parallel` |
| `--sla-params` | `str` | SLA约束条件<br>JSON字符串<br>支持指标：`avg_latency`, `p99_latency`, `avg_ttft`, `p99_ttft`, `avg_tpot`, `p99_tpot`, `rps`, `tps`<br>支持操作符：`<=`, `<`, `min` (延时类); `>=`, `>`, `max` (吞吐类)<br>示例：`'[{"p99_latency": "<=2"}]'` | `None` |
| `--sla-upper-bound` | `int` | 被调优变量的搜索上界 | `65536` |
| `--sla-lower-bound` | `int` | 被调优变量的搜索下界 | `1` |
| `--sla-fixed-parallel` | `int` | 在 `--sla-variable=rate` 时使用的固定并发数；未设置时默认回退到 `--sla-upper-bound` 以兼容旧行为 | `None` |
| `--sla-num-runs` | `int` | 每个并发级别的运行次数（取平均值） | `3` |
| `--sla-number-multiplier` | `float` | 每次测试时请求总数相对于被调优变量（并发数或速率）的倍数，即 `number = round(variable × N)`；未设置时默认为 `2` | `None` |

```{seealso}
SLA自动调优功能使用详见[自动调优指南](./sla_auto_tune.md)。
```

## 数据集配置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--dataset` | `str` | 数据集模式，详见下方[数据集模式](#数据集模式) | - |
| `--dataset-path` | `str` | 数据集文件或目录路径<br>指向文件时直接读取；指向目录时在目录内查找对应数据文件（适用于离线环境使用已下载的数据集缓存） | - |
| `--data-source` | `str` | 数据集加载源，可选值：`modelscope`、`huggingface`、`local`<br>未指定时默认使用 `modelscope`；当 `--dataset-path` 为本地目录时自动视为 `local` | `modelscope` |
| `--dataset-args` | `str` | 数据集专属参数（JSON 字符串），按 `--dataset` 所选数据集的 schema 校验（传入未知键会直接报错） | - |

`--dataset-args` 承载的键按用途分布在下列小节：

| 键 | 用途 | 所在小节 |
|----|------|----------|
| `target_input_len` / `input_len_mode` | 把真实数据的输入截断到固定长度 | [长度控制](#长度控制) |
| `prefix_file` / `prefix_role` | 长上下文前缀注入，构造超长定长输入 | [长上下文前缀注入](#长上下文前缀注入) |
| `speed` / `model_override` / `model_mapping` / `match_output_length` | 生产流量回放的回放行为 | [生产流量回放](#生产流量回放) |
| 各类 token 长度参数 | 多轮对话数据集 | [多轮对话](#多轮对话) |

```{note}
`--multi-turn-args` 已废弃，请改用 `--dataset-args`（键名不变）。旧参数仍可用，会自动并入 `--dataset-args`（同名键以 `--dataset-args` 为准）。
```

### 数据集模式

**文本对话类**

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `openqa` | 从ModelScope自动下载[OpenQA](https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/summary)<br>prompt长度较短（一般<100 token）<br>指定`dataset_path`时使用jsonl文件的`question`字段 | ✓ |
| `longalpaca` | 从ModelScope自动下载[LongAlpaca-12k](https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/dataPeview)<br>prompt长度较长（一般>6000 token）<br>指定`dataset_path`时使用jsonl文件的`instruction`字段 | ✓ |
| `line_by_line` | 逐行将txt文件的每一行作为一个prompt<br>**必需提供`dataset_path`** | ✓（必需） |
| `random` | 根据`prefix-length`、`max-prompt-length`和`min-prompt-length`随机生成prompt<br>**必需指定`tokenizer-path`**<br>[使用示例](./examples.md#随机数据集) | ✗ |
| `custom` | 自定义数据集解析器<br>参考[自定义数据集指南](custom.md/#自定义数据集) | ✓ |

**多模态类**

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `flickr8k` | 从ModelScope自动下载[Flick8k](https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/dataPeview)<br>构建图文输入，数据集较大，适合评测多模态模型<br>支持`--dataset-path`指向本地数据集目录（离线环境） | ✓（目录） |
| `kontext_bench` | 从ModelScope自动下载[Kontext-Bench](https://modelscope.cn/datasets/black-forest-labs/kontext-bench/dataPeview)<br>构建图文输入，约1000条数据，适合快速评测多模态模型<br>支持`--dataset-path`指向本地数据集目录（离线环境） | ✓（目录） |
| `random_vl` | 随机生成图像和文本输入<br>在`random`基础上增加图像相关参数<br>[使用示例](./examples.md#随机图文数据集) | ✗ |

**Embedding 类**

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `embedding` | 从文件加载文本数据评测Embedding模型<br>支持Line-by-line(TXT)或JSONL格式（含`text`字段） | ✓ (必需) |
| `random_embedding` | 根据`max-prompt-length`和`min-prompt-length`随机生成query评测Embedding模型<br>**必需指定`tokenizer-path`** | ✗ |
| `embedding_batch` | 批量发送文本数据评测Embedding模型<br>从文件加载数据<br>支持`--extra-args '{"batch_size": 8}'`设置批次大小 | ✓ (必需) |
| `random_embedding_batch` | 批量发送根据`max-prompt-length`和`min-prompt-length`随机生成query数据评测Embedding模型<br>**必需指定`tokenizer-path`**<br>支持`--extra-args '{"batch_size": 8}'`设置批次大小 | ✗ |

**Rerank 类**

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `rerank` | 从文件加载Query-Document对评测Rerank模型<br>支持JSONL格式 (含`query`和`documents`字段) | ✓ (必需) |
| `random_rerank` | 根据`max-prompt-length`和`min-prompt-length`随机生成query数据评测Rerank模型<br>**必需指定`tokenizer-path`**<br>支持`--extra-args '{"num_documents": 10, "document_length_ratio": 5}'`设置文档数量和相对query的长度倍数 | ✗ |

**多轮对话类**

需配合 `--multi-turn` 使用，参数见[多轮对话](#多轮对话)，详见[多轮对话压测指南](./multi_turn.md)。

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `random_multi_turn` | 合成多轮对话，每轮随机生成 token 序列<br>**必需 `--tokenizer-path`、`--max-turns`**<br>[使用示例](./multi_turn.md#random_multi_turn) | ✗ |
| `share_gpt_zh_multi_turn` | 从 ModelScope 自动下载中文 [ShareGPT](https://www.modelscope.cn/datasets/swift/sharegpt) 数据集（约 70k 条），保留完整多轮对话<br>[使用示例](./multi_turn.md#share_gpt_multi_turn) | ✓ |
| `share_gpt_en_multi_turn` | 从 ModelScope 自动下载英文 [ShareGPT](https://www.modelscope.cn/datasets/swift/sharegpt) 数据集（约 70k 条），保留完整多轮对话 | ✓ |
| `custom_multi_turn` | 使用本地 JSONL 文件作为自定义多轮对话数据集<br>每行为 OpenAI messages 格式的 JSON 数组，适合已有对话数据直接压测<br>**必需提供`dataset_path`**<br>[使用示例](./multi_turn.md#custom_multi_turn) | ✓（必需） |

**生产流量回放类**

需配合 `--open-loop` 使用，trace 文件格式与回放参数见[生产流量回放](#生产流量回放)。

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `workload_trace` | 回放录制的生产流量 JSONL：按原始时间戳、请求体、headers 逐字重放，贴近真实负载（突发流量、异构请求、多模型路由）<br>每条请求自带 `model`，保留多模型混合路由<br>**必需 `--open-loop` 和 `--dataset-path`**；`--model`/`--number` 可选（trace 自带模型与条数） | ✓（必需） |

## 输入构造

控制送入模型的输入内容与长度。以下小节的 `--xxx` 为命令行参数，无前缀的键通过 `--dataset-args` 的 JSON 传入。

### 长度控制

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--max-prompt-length` | `int` | 最大输入prompt长度<br>超过该值时将丢弃prompt | `131072` |
| `--min-prompt-length` | `int` | 最小输入prompt长度<br>小于该值时将丢弃prompt | `0` |

想用真实数据（而非 `random`）压测某个**固定输入长度**时，用 `--dataset-args` 的下列键。支持 `openqa`、`longalpaca`、`line_by_line`（仅纯文本行）、ShareGPT（`share_gpt_zh` / `share_gpt_en`），**需配合 `--tokenizer-path`**。

| 键 | 说明 | 默认值 |
|------|------|--------|
| `target_input_len` | 目标输入 token 数。设置后每条 prompt 都会被截断到该长度 | 不启用 |
| `input_len_mode` | 对**短于目标**的 prompt 怎么处理：`cap`（原样保留，该条长度可能小于目标）；`drop`（丢弃，保证产出的每条都恰好等于目标）。`drop` 不能与 `prefix_file` 同用（见[长上下文前缀注入](#长上下文前缀注入)） | `cap` |

```bash
# 把每条输入截断到 2048 token
evalscope perf \
  --model qwen2.5 --url http://127.0.0.1:8000/v1/completions \
  --dataset share_gpt_zh --tokenizer-path /path/to/tokenizer \
  --dataset-args '{"target_input_len": 2048}'
```

长度口径为不含 chat template 开销的裸内容 token 数。ShareGPT 等多轮数据集默认只对**最后一轮 user 内容**做截断/过滤（与单轮一致）；仅当配置了 `prefix_file` 时才改为按整段对话所有消息内容之和计量、超长丢弃、不足由前缀补齐（见[长上下文前缀注入](#长上下文前缀注入)）。`line_by_line` 的 JSON 行（messages 数组 / 完整 request body）不走长度控制，同时设置本节参数会直接报错。

它和 `--max/min-prompt-length` 的区别：
- `--max/min-prompt-length` 只**筛选**、不改内容——长度不在区间内的样本被丢弃，你得到的是长短不一的真实样本；
- `target_input_len` 会**改写内容**——把每条 prompt 截到指定长度，适合“固定输入长度”的对照压测。

> 想让“每条恰好 N token”，只能用 `target_input_len`；把 `--min-prompt-length` 和 `--max-prompt-length` 设成相等是做不到的（真实数据几乎没有恰好等于 N 的，会被筛空）。`random` 数据集除外——它是现场生成的，min=max 即可定长，无需本参数。

### 长上下文前缀注入

真实指令集大多只有 4K-8K token，直接把 `target_input_len` 设成 128K 会导致 `drop` 模式筛空、`cap` 模式长短不一。`prefix_file` 允许指定一份长文本（如书籍、文档语料），框架按 token 预算把它精确切成 `target_input_len − prompt 长度` 的前缀与短 prompt 拼接，使**每条请求总输入恰好等于目标长度**，且保持真实人类语言的低熵特征（适合测 Prefix-Cache 命中率、MTP 接受率）。适用数据集与[长度控制](#长度控制)相同。

| 键 | 说明 | 默认值 |
|------|------|--------|
| `prefix_file` | 长前缀文本文件路径（UTF-8 纯文本）。**必须同时设置 `target_input_len`**，且不能与 `input_len_mode="drop"` 同用 | 不启用 |
| `prefix_role` | 前缀注入角色：`system`（作为开头的 system 消息注入，贴近真实 RAG 流量，推理框架对 system 的 Prefix-Cache 管理通常有专门优化）；`user`（直接拼在 user 消息内容最前面） | `system` |

```bash
# 用长文本前缀把每条请求精确对齐到 131072 token，前缀注入 system 角色
evalscope perf \
  --model qwen2.5 --url http://127.0.0.1:8000/v1/chat/completions \
  --dataset openqa --tokenizer-path /path/to/tokenizer \
  --dataset-args '{"target_input_len": 131072, "prefix_file": "/path/to/long_text.txt", "prefix_role": "system"}'
```

行为说明：
- **预算分配**：prompt 保持原样（超长时按 `input_len_mode` 截断），前缀精确切到 `target_input_len − 所有消息内容 token 数`，总长恰为目标值；多轮对话的历史一并计入（见[长度控制](#长度控制)的长度口径）。
- **与 `drop` 互斥**：`drop` 只保留已经填满 `target_input_len` 的 prompt，前缀预算恒为 0，注入必然失效，因此配置时直接报错。要定长请用 `cap` + 前缀补齐。
- **前缀不足**：前缀文件 token 数不足以填满剩余预算时，会循环重复（tile）补齐后再精确截断，并打 warning 提示。
- **降级规则**：`apply_chat_template` 关闭（如 `/v1/completions` 端点）时无法注入 system 消息，自动降级为纯文本前缀拼接并打 warning。
- **拼接边界**：`prefix_role="user"` 和纯文本降级模式下前缀与 prompt 直接相接，前缀与 prompt 的 token 数是分别计算的，拼接处两侧字符可能被 tokenizer 合并或拆分，因此实测总长可能与目标相差约 ±1 token。`prefix_role="system"`（chat template 模式）有消息标记隔断边界，不受此影响，始终精确。
- **缓存友好**：所有请求共享同一段前缀开头（长度随各条 prompt 略有差异），天然适配 Prefix-Cache 命中测试。

```{note}
本节的 `prefix_file` 注入的是**真实文本**前缀，用于真实数据集；`--prefix-length` 注入的是**随机 token** 前缀，只对 `random` 数据集有效（见下方[随机数据生成](#随机数据生成)）。两者用途不同，不要混用。
```

### 随机数据生成

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--prefix-length` | `int` | prompt 的随机 token 前缀长度<br>仅对 `random` 数据集有效；所有请求共享同一前缀，可用于制造 Prefix-Cache 命中 | `0` |
| `--image-width` | `int` | 随机VL数据集图像宽度 | `224` |
| `--image-height` | `int` | 随机VL数据集图像高度 | `224` |
| `--image-format` | `str` | 随机VL数据集图像格式 | `RGB` |
| `--image-num` | `int` | 随机VL数据集图像数量 | `1` |
| `--image-patch-size` | `int` | 图像的patch大小<br>仅用于本地图像token计算 | `28` |

`random` 数据集的长度由 `--min-prompt-length` / `--max-prompt-length` 决定（两者相等即定长），无需 `target_input_len`。

### Prompt 与模板

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--prompt` | `str` | 指定请求prompt<br>字符串或本地文件（通过`@/path/to/file`指定）<br>优先级高于`dataset`<br>示例：`@./prompt.txt` | - |
| `--query-template` | `str` | 指定查询模板<br>JSON字符串或本地文件（通过`@/path/to/file`指定）<br>示例：`@./query_template.json` | - |
| `--apply-chat-template` | `bool` | 是否应用聊天模板 | `None`（根据URL后缀自动判断） |
| `--tokenize-prompt` | `bool` | 在客户端将prompt tokenize为token ID列表，绕过服务端重新tokenize，通过`/v1/completions`直接发送 | `False` |

## 多轮对话

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--multi-turn` | `bool` | 启用多轮对话压测模式；`--number` 表示总发送 turn 数，`--parallel` 表示并发 turn 数 | `False` |
| `--min-turns` | `int` | 每个对话最少用户轮数；`random_multi_turn` 与 `swe_smith` 使用 | `1` |
| `--max-turns` | `int` | 每个对话最多用户轮数；`random_multi_turn` 必需；ShareGPT/`custom_multi_turn` 可选（截断过长对话）；`swe_smith` 用作每个对话轮数采样上界，未设置时回退到 `--min-turns` | `None` |
| `--num-workers` | `int` | CPU 密集型数据集/请求生成的 worker 进程数。<br>`0` = 根据 CPU 亲和性自动检测；`1` = 串行（无多进程）；`>1` = 显式指定 worker 数。<br>用于 `random`（长 prompt 并行生成）和 `swe_smith`（live 构建）。取代已废弃的 `multi_turn_args.num_workers`。 | `0` |

`swe_smith` 等多轮数据集的 token 长度参数通过 `--dataset-args` 传入。

```{seealso}
可用的多轮数据集见[数据集模式](#数据集模式)，完整用法详见[多轮对话压测指南](./multi_turn.md)。
```

## 生产流量回放

用 `--dataset workload_trace` 把录制的生产流量按**原始到达节奏**逐字回放，贴近真实负载。**必需 `--open-loop`**，无需 `--rate`（到达时刻由 trace 时间戳决定）。完整示例见[生产流量回放](./examples.md#生产流量回放)。

trace 文件为 JSONL，每行一条请求记录：

```json
{"body": {"model": "qwen-plus", "messages": [{"role": "user", "content": "hi"}]}, "timestamp": 1700000000.0}
{"body": {"model": "qwen-max", "messages": [{"role": "user", "content": "hello"}]}, "timestamp": 1700000001.5, "headers": {"X-Tag": "exp"}, "request_id": "req-42", "completion_tokens": 256}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `body` | ✓ | 完整请求体（dict 或 JSON 字符串），原样发送 |
| `timestamp` | ✓ | 到达时刻（数字或 ISO-8601 字符串），仅相对间隔有意义，须单调不减 |
| `headers` | | 该请求专属 HTTP 头（与 CLI headers 合并，CLI 优先；hop-by-hop 头会被剔除） |
| `request_id` | | 透传到结果，用于与原始请求关联 |
| `completion_tokens` | | 配合 `match_output_length` 使用 |

回放行为通过 `--dataset-args` 调整：

| 键 | 类型 | 说明 | 默认值 |
|----|------|------|--------|
| `speed` | float | 回放倍速（2.0 = 2× 快，0.5 = 2× 慢） | `1.0` |
| `model_override` | str | 把所有请求的 `model` 全量替换为该值 | 不启用 |
| `model_mapping` | dict | 按名映射 `model`（命中优先；未命中保留原值） | 不启用 |
| `match_output_length` | bool | 用记录的 `completion_tokens` 设 `max_tokens` 并启用 `ignore_eos`（需 vLLM 等支持；对约束解码请求自动跳过 `ignore_eos`） | `false` |

```{note}
`--model` 对 `workload_trace` **不会改写** trace body——每条请求保留自己的 `model`，从而保留多模型混合路由。需要改模型请用 `model_override` / `model_mapping`。
```

## 模型与生成参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--tokenizer-path` | `str` | 分词器权重路径<br>用于计算输入和输出的token数量<br>通常与模型权重在同一目录<br>压测 chat 接口时，该分词器需自带 chat template（详见下方说明） | `None` |
| `--frequency-penalty` | `float` | frequency_penalty值 | - |
| `--logprobs` | `bool` | 是否返回对数概率 | - |
| `--max-tokens` | `int` 或 `int int` | 可以生成的最大token数量<br>• 单个整数：固定值，如 `--max-tokens 2048`<br>• 两个整数：`最小值 最大值`，每次请求从该范围均匀随机采样，如 `--max-tokens 512 2048` | `2048` |
| `--min-tokens` | `int` | 生成的最少token数量<br>注意：并非所有模型服务都支持<br>对于`vLLM>=0.8.1`，需额外设置<br>`--extra-args '{"ignore_eos": true}'` | - |
| `--n-choices` | `int` | 生成的补全选择数量 | - |
| `--seed` | `int` | 随机种子 | `None` |
| `--stop` | `str` | 停止生成的tokens | - |
| `--stop-token-ids` | `list[int]` | 停止生成的token ID列表 | - |
| `--temperature` | `float` | 采样温度 | `0` |
| `--top-p` | `float` | top_p采样 | - |
| `--top-k` | `int` | top_k采样 | - |
| `--extra-args` | `str` | 额外传入请求体的参数<br>JSON字符串格式<br>示例：`'{"ignore_eos": true}'` | - |

### 分词器与 chat template

压测 `chat/completions` 接口时 `--apply-chat-template` 默认开启，客户端会先套上 chat template 再统计 token 数，使长度计算与服务端 `usage.prompt_tokens` 对齐；因此 `--tokenizer-path` 指向的分词器必须自带 Jinja 格式的 chat template。DeepSeek-V3.2 / V4 官方改为提供 `encoding` 脚本，base / pretrain 权重也不带模板，这类权重会直接报错，报错信息中列出了可选的处理方式。

两个易错点：不传 `--tokenizer-path` 时 `--min/max-prompt-length` 按字符数而非 token 数过滤；借用其它模型的分词器不会报错，但词表不一致会默默把 token 统计算偏。

## 结果输出

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--visualizer` | `str` | 可视化工具<br>可选：`wandb`、`swanlab`、`clearml`<br>设置后指标将保存到指定工具 | `None` |
| `--enable-progress-tracker` | `bool` | 是否开启进度追踪，将层级压测进度实时写入`progress.json`，可通过服务接口查询 | `False` |
| `--wandb-api-key` | `str` | wandb API密钥<br>**已废弃**，请使用`--visualizer wandb` | - |
| `--swanlab-api-key` | `str` | swanlab API密钥<br>**已废弃**，请使用`--visualizer swanlab` | - |
| `--outputs-dir` | `str` | 输出文件路径 | `./outputs` |
| `--no-timestamp` | `bool` | 输出目录不包含时间戳 | `False` |

## 其他参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--db-commit-interval` | `int` | 在写入SQLite数据库前缓冲的行数 | `1000` |
| `--queue-size-multiplier` | `int` | 请求队列的最大大小<br>计算方式：`parallel * multiplier` | `5` |
| `--in-flight-task-multiplier` | `int` | 最大调度任务数<br>计算方式：`parallel * multiplier` | `2` |
