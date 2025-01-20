# 参数说明

执行 `evalscope eval --help` 可获取全部参数说明。

## 模型参数
- `--model`: 被评测的模型名称。
  - 指定为模型在[ModelScope](https://modelscope.cn/)中的`id`，将自动下载模型，例如[Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary)；
  - 指定为模型的本地路径，例如`/path/to/model`，将从本地加载模型；
  - 评测目标为模型API服务时，需要指定为服务对应的模型id，例如`Qwen2.5-0.5B-Instruct`。
- `--model-id`: 被评测的模型的别名，用于报告展示。默认为`model`的最后一部分，例如`Qwen/Qwen2.5-0.5B-Instruct`的`model-id`为`Qwen2.5-0.5B-Instruct`。
- `--api-url`: (仅在`eval-type=service`时有效) 模型API端点，默认为`None`；支持传入本地或远端的OpenAI API格式端点，例如`http://127.0.0.1:8000/v1/chat/completions`。
- `--api-key`: (仅在`eval-type=service`时有效) 模型API端点密钥，默认为`EMPTY`
- `--model-args`: 模型加载参数，以逗号分隔，`key=value`形式，默认参数：
  - `revision`: 模型版本，默认为`master`
  - `precision`: 模型精度，默认为`torch.float16`
  - `device_map`: 模型分配设备，默认为`auto`
- `--generation-config`: 生成参数，以逗号分隔，`key=value`形式，默认参数：
  - `do_sample`: 是否使用采样，默认为`false`
  - `max_length`: 最大长度，默认为2048
  - `max_new_tokens`: 生成最大长度，默认为512
- `--chat-template`: 模型推理模板，默认为`None`，表示使用transformers的`apply_chat_template`；支持传入jinjia模版字符串，来自定义推理模板
- `--template-type`: 模型推理模板，已弃用，参考`--chat-template`


## 数据集参数
- `--datasets`: 数据集名称，支持输入多个数据集，使用空格分开，数据集将自动从modelscope下载，支持的数据集参考[数据集列表](./supported_dataset.md#支持的数据集)
- `--dataset-args`: 评测数据集的设置参数，以`json`格式传入，key为数据集名称，value为参数，注意需要跟`--datasets`参数中的值一一对应：
  - `local_path`: 数据集本地路径，指定后将尝试加载本地数据
  - `prompt_template`: 评测数据集的prompt模板，指定后将拼接在每个评测数据内容之前
  - `subset_list`: 评测数据子集列表，指定后将只使用子集数据
  - `few_shot_num`: few-shot的数量
  - `few_shot_random`: 是否随机采样few-shot数据，默认为`False`
- `--dataset-dir`: 数据集下载路径，默认为`~/.cache/modelscope/datasets`
- `--dataset-hub`: 数据集下载源，默认为`modelscope`，可选`huggingface`
- `--limit`: 每个数据集最大评测数据量，不填写则默认为全部评测，可用于快速验证

## 评测参数
- `--eval-stage`: 评测阶段，可选`all`, `infer`, `review`
  - `all`: 完整评测，包含推理和评测
  - `infer`: 仅进行推理，不进行评测
  - `review`: 仅进行数据评测，不进行推理
- `--eval-type`: 评测类型，可选`checkpoint`, `custom`, `service`；默认为`checkpoint`
- `--eval-backend`: 评测后端，可选`Native`, `OpenCompass`, `VLMEvalKit`, `RAGEval`, `ThirdParty`，默认为`Native`
  - `OpenCompass`用于评测大语言模型
  - `VLMEvalKit`用于评测多模态模型
  - `RAGEval`用于评测RAG流程、Embedding模型、Reranker模型、CLIP模型
    ```{seealso}
    其他评测后端[使用指南](../user_guides/backend/index.md)
    ```
  - `ThirdParty` 用于其他特殊任务评测，例如[ToolBench](../third_party/toolbench.md), [LongBench](../third_party/longwriter.md)
- `--eval-config`: 使用非`Native`评测后端时，需要传入该参数


## 其他参数

- `--work-dir`: 模型评测输出路径，默认为`./outputs/{timestamp}`
- `--use-cache`: 使用本地缓存的路径，默认为`None`；如果为指定路径，例如`outputs/20241210_194434`，将重用路径下的模型推理结果，若未完成推理则会继续推理，之后进行评测。
- `--seed`: 随机种子，默认为`42`
- `--debug`: 是否开启调试模式，默认为`false`
- `--dry-run`: 预检参数，不进行推理，只打印参数，默认为`false`