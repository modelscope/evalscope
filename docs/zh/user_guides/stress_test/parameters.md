# 参数说明

执行 `evalscope perf --help` 可获取全部参数说明：


## 基本设置
- `--model` 测试模型名称。
- `--url` 指定API地址，支持`/chat/completion`和`/completion`两种endpoint。
- `--name` wandb/swanlab数据库结果名称和结果数据库名称，默认为: `{model_name}_{current_time}`，可选。
- `--api` 指定服务API，目前支持[openai|dashscope|local|local_vllm]。
  - 指定为`openai`，则使用支持OpenAI的API，需要提供`--url`参数。
  - 指定为`dashscope`，则使用支持DashScope的API，需要提供`--url`参数。
  - 指定为`local`，则使用本地文件作为模型，并使用transformers进行推理。`--model`为模型文件路径，也可为model_id，将自动从modelscope下载模型，例如`Qwen/Qwen2.5-0.5B-Instruct`。
  - 指定为`local_vllm`，则使用本地文件作为模型，并启动vllm推理服务。`--model`为模型文件路径，也可为model_id，将自动从modelscope下载模型，例如`Qwen/Qwen2.5-0.5B-Instruct`。
  - 您也可以自定义API，请参考[自定义API指南](./custom.md/#自定义请求-api)。
- `--port` 本地推理服务端口，默认为8877，仅对`local`和`local_vllm`有效。
- `--attn-implementation` Attention实现方式，默认为None，可选[flash_attention_2|eager|sdpa]，仅在`api`为`local`时有效。
- `--api-key` API密钥，可选。
- `--debug` 输出调试信息。

## 网络配置
- `--connect-timeout` 网络连接超时，默认为600s。
- `--read-timeout` 网络读取超时，默认为600s。
- `--headers` 额外的HTTP头，格式为`key1=value1 key2=value2`。该头将用于每个查询。
- `--no-test-connection` 不发送连接测试，直接开始压测，默认为False。

## 请求控制
- `--parallel` 并发请求的数量，可以传入多个值，用空格隔开；默认为1。
- `--number` 发出的请求的总数量，可以传入多个值，用空格隔开（需与`parallel`一一对应）；默认为1000。
- `--rate` 每秒生成的请求数量（并不发送请求），默认为-1，表示所有请求将在时间0生成，没有间隔；否则，我们使用泊松过程生成请求间隔。
  ```{tip}
  在本工具的实现中请求生成与发送是分开的：
  `--rate`参数用于控制每秒生成的请求数量，请求将放入请求队列。
  `--parallel`参数用于控制发送请求的worker数量，worker将从请求队列获取请求并发送，且在上一请求回复后才发送下一请求。
  ```
- `--log-every-n-query` 每n个查询记录日志，默认为10。
- `--stream` 使用SSE流输出，默认为True。注意：需要设置`--stream`以测量Time to First Token (TTFT)指标；设置`--no-stream`将不使用流式输出。
- `--sleep-interval` 在每次性能测试之间的休眠时间，单位为秒，默认为5秒。该参数可以帮助避免过载服务器。

## Prompt设置
- `--max-prompt-length` 最大输入prompt长度，默认为`131072`，大于该值时，将丢弃prompt。
- `--min-prompt-length` 最小输入prompt长度，默认为0，小于该值时，将丢弃prompt。
- `--prefix-length` promt的前缀长度，默认为0，仅对于`random`数据集有效。
- `--prompt` 指定请求prompt，一个字符串或本地文件，使用优先级高于`dataset`。使用本地文件时，通过`@/path/to/file`指定文件路径，例如`@./prompt.txt`。
- `--query-template` 指定查询模板，一个`JSON`字符串或本地文件，使用本地文件时，通过`@/path/to/file`指定文件路径，例如`@./query_template.json`。
- `--apply-chat-template` 是否应用聊天模板，默认为None，将根据url后缀是否为`chat/completion`自动选择。

## 数据集配置
- `--dataset` 可以指定如下数据集模式，您也可以使用python自定义数据集解析器，参考[自定义数据集指南](custom.md/#自定义数据集)。
  - `openqa` 使用jsonl文件的 `question` 字段作为prompt。不指定`dataset_path`将从modelscope自动下载[数据集](https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/summary)，prompt长度较短，一般在 100 token 以下。
  - `longalpaca` 使用jsonl文件的 `instruction` 字段作为prompt。不指定`dataset_path`将从modelscope自动下载[数据集](https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/dataPeview)，prompt长度较长，一般在 6000 token 以上。
  - `flickr8k` 将构建图文输入，适合评测多模态模型；从modelscope自动下载[数据集](https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/dataPeview)，不支持指定`dataset_path`。
  - `line_by_line` 必需提供`dataset_path`，逐行将txt文件的每一行作为一个提示，。
  - `random` 根据`prefix-length`，`max-prompt-length`和`min-prompt-length`随机生成prompt，必需指定`tokenizer-path`，[使用示例](./examples.md#使用random数据集)。
- `--dataset-path` 数据集文件的路径，与数据集结合使用。

## 模型设置
- `--tokenizer-path` 可选，指定分词器权重路径，用于计算输入和输出的token数量，通常与模型权重在同一目录下。
- `--frequency-penalty` frequency_penalty值。
- `--logprobs` 对数概率。
- `--max-tokens` 可以生成的最大token数量。
- `--min-tokens` 生成的最少token数量，不是所有模型服务都支持该参数，请查看对应API文档。对于`vLLM>=0.8.1`版本，需要额外设置`--extra-args '{"ignore_eos": true}'`。
- `--n-choices` 生成的补全选择数量。
- `--seed` 随机种子，默认为0。
- `--stop` 停止生成的tokens。
- `--stop-token-ids` 设置停止生成的token的ID。
- `--temperature` 采样温度，默认为0。
- `--top-p` top_p采样。
- `--top-k` top_k采样。
- `--extra-args` 额外传入请求体的参数，格式为json字符串，例如`'{"ignore_eos": true}'`。

## 数据存储
- `--wandb-api-key` wandb API密钥，如果设置，则度量将保存到wandb。
- `--swanlab-api-key` swanlab API密钥，如果设置，则度量将保存到swanlab。
- `--outputs-dir` 输出文件路径，默认为`./outputs`。
