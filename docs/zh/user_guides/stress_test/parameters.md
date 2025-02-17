# 参数说明

执行 `evalscope perf --help` 可获取全部参数说明：


## 基本设置
- `--model` 测试模型名称。
- `--url` 指定API地址。
- `--name` wandb数据库结果名称和结果数据库名称，默认为: `{model_name}_{current_time}`，可选。
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
- `--connect-timeout` 网络连接超时，默认为120s。
- `--read-timeout` 网络读取超时，默认为120s。
- `--headers` 额外的HTTP头，格式为`key1=value1 key2=value2`。该头将用于每个查询。

## 请求控制
- `--number` 发出的请求的总数量；默认为None，表示基于数据集数量发送请求。
- `--parallel` 并发请求的数量，默认为1。
- `--rate` 每秒生成的请求数量（并不发送请求），默认为-1，表示所有请求将在时间0生成，没有间隔；否则，我们使用泊松过程生成请求间隔。
  ```{tip}
  在本工具的实现中请求生成与发送是分开的：
  `--rate`参数用于控制每秒生成的请求数量，请求将放入请求队列。
  `--parallel`参数用于控制发送请求的worker数量，worker将从请求队列获取请求并发送，且在上一请求回复后才发送下一请求。 两个参数不建议同时设置，也即在本工具中，`--rate`参数仅对`--parallel`为1时有效。
  ```
- `--log-every-n-query` 每n个查询记录日志，默认为10。
- `--stream` 使用SSE流输出，默认为False。

## Prompt设置
- `--max-prompt-length` 最大输入prompt长度，默认为`sys.maxsize`，大于该值时，将丢弃prompt。
- `--min-prompt-length` 最小输入prompt长度，默认为0，小于该值时，将丢弃prompt。
- `--prompt` 指定请求prompt，一个字符串或本地文件，使用优先级高于`dataset`。使用本地文件时，通过`@/path/to/file`指定文件路径，例如`@./prompt.txt`。
- `--query-template` 指定查询模板，一个`JSON`字符串或本地文件，使用本地文件时，通过`@/path/to/file`指定文件路径，例如`@./query_template.json`。

## 数据集配置
- `--dataset` 指定数据集[openqa|longalpaca|line_by_line|flickr8k]，您也可以使用python自定义数据集解析器，参考[自定义数据集指南](custom.md/#自定义数据集)。
  - `line_by_line`逐行将每一行作为一个提示，需提供`dataset_path`。
  - `longalpaca` 将获取 `item['instruction']` 作为提示，不指定`dataset_path`将从modelscope自动下载。
  - `openqa` 将获取 `item['question']` 作为提示，不指定`dataset_path`将从modelscope自动下载。
  - `flickr8k` 将构建图文输入，适合评测多模态模型；从modelscope自动下载数据集，不支持指定`dataset_path`。
- `--dataset-path` 数据集文件的路径，与数据集结合使用。openqa与longalpaca可不指定数据集路径，将自动下载；line_by_line必须指定本地数据集文件，将一行一行加载。

## 模型设置
- `--tokenizer-path` 可选，指定分词器权重路径，用于计算输入和输出的token数量，通常与模型权重在同一目录下。
- `--frequency-penalty` frequency_penalty值。
- `--logprobs` 对数概率。
- `--max-tokens` 可以生成的最大token数量。
- `--min-tokens` 生成的最少token数量。
- `--n-choices` 生成的补全选择数量。
- `--seed` 随机种子，默认为42。
- `--stop` 停止生成的tokens。
- `--stop-token-ids` 设置停止生成的token的ID。
- `--temperature` 采样温度。
- `--top-p` top_p采样。

## 数据存储
- `--wandb-api-key` wandb API密钥，如果设置，则度量将保存到wandb。
- `--outputs-dir` 输出文件路径，默认为`./outputs`。