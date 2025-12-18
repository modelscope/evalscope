# 参数说明

执行 `evalscope perf --help` 可获取全部参数说明。

## 基本设置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model` | `str` | 测试模型名称，或模型路径 | - |
| `--url` | `str` | API地址，支持`/chat/completion`和`/completion`端点 | - |
| `--name` | `str` | wandb/swanlab数据库结果名称和结果数据库名称 | `{model_name}_{current_time}` |
| `--api` | `str` | 服务API类型<br>• `openai`: OpenAI兼容API（需提供`--url`）<br>• `local`: 启动本地transformers推理<br>• `local_vllm`: 启动本地vLLM推理服务<br>• 自定义：参考[自定义API指南](./custom.md/#自定义请求-api) | - |
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
| `--rate` | `float` | 每秒生成的请求数量<br>• `-1`: 所有请求在时间0生成，无间隔<br>• 其他值: 使用泊松过程生成请求间隔 | `-1` |
| `--log-every-n-query` | `int` | 每N个查询记录日志 | `10` |
| `--stream` | `bool` | 是否使用SSE流输出<br>需要启用以测量TTFT（Time to First Token）指标 | `True` |
| `--sleep-interval` | `int` | 每次性能测试之间的休眠时间（秒）<br>避免过载服务器 | `5` |

```{tip}
在本工具的实现中请求生成与发送是分开的：
- `--rate`参数控制每秒生成的请求数量，请求将放入请求队列
- `--parallel`参数控制发送请求的worker数量，worker从请求队列获取请求并发送，且在上一请求回复后才发送下一请求
```

## SLA设置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--sla-auto-tune` | `bool` | 是否启用SLA自动调优模式 | `False` |
| `--sla-variable` | `str` | 自动调优的变量<br>可选：`parallel`（并发数）、`rate`（请求速率） | `parallel` |
| `--sla-params` | `str` | SLA约束条件<br>JSON字符串<br>支持指标：`avg_latency`, `p99_latency`, `avg_ttft`, `p99_ttft`, `avg_tpot`, `p99_tpot`, `rps`, `tps`<br>支持操作符：`<=`, `<`, `min` (延时类); `>=`, `>`, `max` (吞吐类)<br>示例：`'[{"p99_latency": "<=2"}]'` | `None` |
| `--sla-upper-bound` | `int` | 自动调优时的最大并发数/速率限制 | `65536` |
| `--sla-lower-bound` | `int` | 自动调优时的最小并发数/速率限制 | `1` |
| `--sla-num-runs` | `int` | 每个并发级别的运行次数（取平均值） | `3` |

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
| `--dataset-path` | `str` | 数据集文件路径<br>与数据集结合使用 | - |

### dataset 模式说明

| 模式 | 说明 | 支持dataset-path |
|------|------|------------------|
| `openqa` | 从ModelScope自动下载[OpenQA](https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/summary)<br>prompt长度较短（一般<100 token）<br>指定`dataset_path`时使用jsonl文件的`question`字段 | ✓ |
| `longalpaca` | 从ModelScope自动下载[LongAlpaca-12k](https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/dataPeview)<br>prompt长度较长（一般>6000 token）<br>指定`dataset_path`时使用jsonl文件的`instruction`字段 | ✓ |
| `line_by_line` | 逐行将txt文件的每一行作为一个prompt<br>**必需提供`dataset_path`** | ✓（必需） |
| `flickr8k` | 从ModelScope自动下载[Flick8k](https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/dataPeview)<br>构建图文输入，数据集较大，适合评测多模态模型 | ✗ |
| `kontext_bench` | 从ModelScope自动下载[Kontext-Bench](https://modelscope.cn/datasets/black-forest-labs/kontext-bench/dataPeview)<br>构建图文输入，约1000条数据，适合快速评测多模态模型 | ✗ |
| `random` | 根据`prefix-length`、`max-prompt-length`和`min-prompt-length`随机生成prompt<br>**必需指定`tokenizer-path`**<br>[使用示例](./examples.md#使用random数据集) | ✗ |
| `random_vl` | 随机生成图像和文本输入<br>在`random`基础上增加图像相关参数<br>[使用示例](./examples.md#使用random图文数据集) | ✗ |
| `custom` | 自定义数据集解析器<br>参考[自定义数据集指南](custom.md/#自定义数据集) | ✓ |

## 模型设置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--tokenizer-path` | `str` | 分词器权重路径<br>用于计算输入和输出的token数量<br>通常与模型权重在同一目录 | `None` |
| `--frequency-penalty` | `float` | frequency_penalty值 | - |
| `--logprobs` | `bool` | 是否返回对数概率 | - |
| `--max-tokens` | `int` | 可以生成的最大token数量 | - |
| `--min-tokens` | `int` | 生成的最少token数量<br>注意：并非所有模型服务都支持<br>对于`vLLM>=0.8.1`，需额外设置<br>`--extra-args '{"ignore_eos": true}'` | - |
| `--n-choices` | `int` | 生成的补全选择数量 | - |
| `--seed` | `int` | 随机种子 | `None` |
| `--stop` | `str` | 停止生成的tokens | - |
| `--stop-token-ids` | `list[int]` | 停止生成的token ID列表 | - |
| `--temperature` | `float` | 采样温度 | `0` |
| `--top-p` | `float` | top_p采样 | - |
| `--top-k` | `int` | top_k采样 | - |
| `--extra-args` | `str` | 额外传入请求体的参数<br>JSON字符串格式<br>示例：`'{"ignore_eos": true}'` | - |

## 数据存储

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--visualizer` | `str` | 可视化工具<br>可选：`wandb`、`swanlab`、`clearml`<br>设置后指标将保存到指定工具 | `None` |
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
