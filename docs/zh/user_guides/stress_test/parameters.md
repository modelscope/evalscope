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
| `--open-loop` | `bool` | 启用开放环路（open-loop）模式：<br>请求按 `--rate` 指定的速率发出，无论服务端是否已处理完之前的请求。<br>• `--rate` 变为扫描变量（支持多值）<br>• `--number` 须与 `--rate` 等长，表示每轮发出的请求总数<br>• `--parallel` 在此模式下被忽略（内部设为 -1 / INF）<br>详见[使用示例](./examples.md#open-loop-开放环路模式) | `False` |
| `--warmup-num` | `float` | 预热请求数量或比例：<br>• `0`：禁用预热（默认）<br>• `>= 1`：绝对数量，如 `--warmup-num 10` 表示预热 10 个请求<br>• `0 < value < 1`：比例模式，如 `--warmup-num 0.1` 表示预热数量为 `--number` 的 10%<br>预热请求使用与正式压测相同的并发/速率发送，但**不计入性能指标**<br>适用于消除冷启动影响（如 KV-cache 填充、JIT 编译等）<br>详见[使用示例](./examples.md#warmup-预热压测) | `0` |
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

## Prompt设置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--max-prompt-length` | `int` | 最大输入prompt长度<br>超过该值时将丢弃prompt | `131072` |
| `--min-prompt-length` | `int` | 最小输入prompt长度<br>小于该值时将丢弃prompt | `0` |
| `--prefix-length` | `int` | prompt的前缀长度<br>仅对`random`数据集有效 | `0` |
| `--prompt` | `str` | 指定请求prompt<br>字符串或本地文件（通过`@/path/to/file`指定）<br>优先级高于`dataset`<br>示例：`@./prompt.txt` | - |
| `--query-template` | `str` | 指定查询模板<br>JSON字符串或本地文件（通过`@/path/to/file`指定）<br>示例：`@./query_template.json` | - |
| `--apply-chat-template` | `bool` | 是否应用聊天模板 | `None`（根据URL后缀自动判断） |
| `--image-width` | `int` | 随机VL数据集图像宽度 | `224` |
| `--image-height` | `int` | 随机VL数据集图像高度 | `224` |
| `--image-format` | `str` | 随机VL数据集图像格式 | `RGB` |
| `--image-num` | `int` | 随机VL数据集图像数量 | `1` |
| `--image-patch-size` | `int` | 图像的patch大小<br>仅用于本地图像token计算 | `28` |

## 数据集配置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--dataset` | `str` | 数据集模式，详见下表 | - |
| `--dataset-path` | `str` | 数据集文件或目录路径<br>指向文件时直接读取；指向目录时在目录内查找对应数据文件（适用于离线环境使用已下载的数据集缓存） | - |
| `--data-source` | `str` | 数据集加载源，可选值：`modelscope`、`huggingface`、`local`<br>未指定时默认使用 `modelscope`；当 `--dataset-path` 为本地目录时自动视为 `local` | `modelscope` |
| `--dataset-args` | `str` | 数据集专属参数（JSON 字符串），按 `--dataset` 所选数据集的 schema 校验（传入未知键会直接报错）。承载真实文本数据集的定长输入参数 `target_input_len` / `input_len_mode`，并取代已废弃的 `--multi-turn-args`。详见下方「dataset-args：数据集专属参数」小节 | - |

### dataset 模式说明

**文本对话类**

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `openqa` | 从ModelScope自动下载[OpenQA](https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/summary)<br>prompt长度较短（一般<100 token）<br>指定`dataset_path`时使用jsonl文件的`question`字段 | ✓ |
| `longalpaca` | 从ModelScope自动下载[LongAlpaca-12k](https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/dataPeview)<br>prompt长度较长（一般>6000 token）<br>指定`dataset_path`时使用jsonl文件的`instruction`字段 | ✓ |
| `line_by_line` | 逐行将txt文件的每一行作为一个prompt<br>**必需提供`dataset_path`** | ✓（必需） |
| `random` | 根据`prefix-length`、`max-prompt-length`和`min-prompt-length`随机生成prompt<br>**必需指定`tokenizer-path`**<br>[使用示例](./examples.md#使用random数据集) | ✗ |
| `custom` | 自定义数据集解析器<br>参考[自定义数据集指南](custom.md/#自定义数据集) | ✓ |

**多模态类**

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `flickr8k` | 从ModelScope自动下载[Flick8k](https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/dataPeview)<br>构建图文输入，数据集较大，适合评测多模态模型<br>支持`--dataset-path`指向本地数据集目录（离线环境） | ✓（目录） |
| `kontext_bench` | 从ModelScope自动下载[Kontext-Bench](https://modelscope.cn/datasets/black-forest-labs/kontext-bench/dataPeview)<br>构建图文输入，约1000条数据，适合快速评测多模态模型<br>支持`--dataset-path`指向本地数据集目录（离线环境） | ✓（目录） |
| `random_vl` | 随机生成图像和文本输入<br>在`random`基础上增加图像相关参数<br>[使用示例](./examples.md#使用random图文数据集) | ✗ |

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

需配合 `--multi-turn` 使用，详见[多轮对话压测指南](./multi_turn.md)。

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `random_multi_turn` | 合成多轮对话，每轮随机生成 token 序列<br>**必需 `--tokenizer-path`、`--max-turns`**<br>[使用示例](./multi_turn.md#random_multi_turn) | ✗ |
| `share_gpt_zh_multi_turn` | 从 ModelScope 自动下载中文 [ShareGPT](https://www.modelscope.cn/datasets/swift/sharegpt) 数据集（约 70k 条），保留完整多轮对话<br>[使用示例](./multi_turn.md#share_gpt_multi_turn) | ✓ |
| `share_gpt_en_multi_turn` | 从 ModelScope 自动下载英文 [ShareGPT](https://www.modelscope.cn/datasets/swift/sharegpt) 数据集（约 70k 条），保留完整多轮对话 | ✓ |
| `custom_multi_turn` | 使用本地 JSONL 文件作为自定义多轮对话数据集<br>每行为 OpenAI messages 格式的 JSON 数组，适合已有对话数据直接压测<br>**必需提供`dataset_path`**<br>[使用示例](./multi_turn.md#custom_multi_turn) | ✓（必需） |

### dataset-args：数据集专属参数

`--dataset-args` 用一个 JSON 字符串为不同数据集传入专属参数（键名写错会直接报错，便于排查）。常见两个场景：

**场景一：把真实数据的输入截断到固定长度**

想用真实数据（而非 `random`）压测某个**固定输入长度**时用它。支持 `openqa`、`longalpaca`、`line_by_line`、单轮 ShareGPT（`share_gpt_zh` / `share_gpt_en`），**需配合 `--tokenizer-path`**。

```bash
# 把每条输入截断到 2048 token
evalscope perf \
  --model qwen2.5 --url http://127.0.0.1:8000/v1/completions \
  --dataset share_gpt_zh --tokenizer-path /path/to/tokenizer \
  --dataset-args '{"target_input_len": 2048}'
```

| 键 | 说明 | 默认值 |
|------|------|--------|
| `target_input_len` | 目标输入 token 数。设置后每条 prompt 都会被截断到该长度 | 不启用 |
| `input_len_mode` | 对**短于目标**的 prompt 怎么处理：`cap`（原样保留，该条长度可能小于目标）；`drop`（丢弃，保证产出的每条都恰好等于目标） | `cap` |

它和 `--max/min-prompt-length` 的区别：
- `--max/min-prompt-length` 只**筛选**、不改内容——长度不在区间内的样本被丢弃，你得到的是长短不一的真实样本；
- `target_input_len` 会**改写内容**——把每条 prompt 截到指定长度，适合“固定输入长度”的对照压测。

> 想让“每条恰好 N token”，只能用 `target_input_len`；把 `--min-prompt-length` 和 `--max-prompt-length` 设成相等是做不到的（真实数据几乎没有恰好等于 N 的，会被筛空）。`random` 数据集除外——它是现场生成的，min=max 即可定长，无需本参数。

**场景二：多轮对话数据集的长度参数**

`swe_smith` 等多轮数据集的 token 长度参数也通过 `--dataset-args` 传入，详见[多轮对话压测指南](./multi_turn.md)。

```{note}
`--multi-turn-args` 已废弃，请改用 `--dataset-args`（键名不变）。旧参数仍可用，会自动并入 `--dataset-args`（同名键以 `--dataset-args` 为准）。
```

## 模型设置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--tokenizer-path` | `str` | 分词器权重路径<br>用于计算输入和输出的token数量<br>通常与模型权重在同一目录 | `None` |
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
| `--tokenize-prompt` | `bool` | 在客户端将prompt tokenize为token ID列表，绕过服务端重新tokenize，通过`/v1/completions`直接发送 | `False` |

## 数据存储

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--visualizer` | `str` | 可视化工具<br>可选：`wandb`、`swanlab`、`clearml`<br>设置后指标将保存到指定工具 | `None` |
| `--enable-progress-tracker` | `bool` | 是否开启进度追踪，将层级压测进度实时写入`progress.json`，可通过服务接口查询 | `False` |
| `--wandb-api-key` | `str` | wandb API密钥<br>**已废弃**，请使用`--visualizer wandb` | - |
| `--swanlab-api-key` | `str` | swanlab API密钥<br>**已废弃**，请使用`--visualizer swanlab` | - |
| `--outputs-dir` | `str` | 输出文件路径 | `./outputs` |
| `--no-timestamp` | `bool` | 输出目录不包含时间戳 | `False` |

## 多轮对话设置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--multi-turn` | `bool` | 启用多轮对话压测模式；`--number` 表示总发送 turn 数，`--parallel` 表示并发 turn 数 | `False` |
| `--min-turns` | `int` | 每个对话最少用户轮数；`random_multi_turn` 与 `swe_smith` 使用 | `1` |
| `--max-turns` | `int` | 每个对话最多用户轮数；`random_multi_turn` 必需；ShareGPT/`custom_multi_turn` 可选（截断过长对话）；`swe_smith` 用作每个对话轮数采样上界，未设置时回退到 `--min-turns` | `None` |
| `--num-workers` | `int` | CPU 密集型数据集/请求生成的 worker 进程数。<br>`0` = 根据 CPU 亲和性自动检测；`1` = 串行（无多进程）；`>1` = 显式指定 worker 数。<br>用于 `random`（长 prompt 并行生成）和 `swe_smith`（live 构建）。取代已废弃的 `multi_turn_args.num_workers`。 | `0` |

```{seealso}
多轮对话压测使用详见[多轮对话压测指南](./multi_turn.md)。
```

## 其他参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--db-commit-interval` | `int` | 在写入SQLite数据库前缓冲的行数 | `1000` |
| `--queue-size-multiplier` | `int` | 请求队列的最大大小<br>计算方式：`parallel * multiplier` | `5` |
| `--in-flight-task-multiplier` | `int` | 最大调度任务数<br>计算方式：`parallel * multiplier` | `2` |
