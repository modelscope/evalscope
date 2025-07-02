# 参数说明

执行 `evalscope eval --help` 可获取全部参数说明。

## 模型参数
- `--model`: 被评测的模型名称。
  - 指定为模型在[ModelScope](https://modelscope.cn/)中的`id`，将自动下载模型，例如[Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary)；
  - 指定为模型的本地路径，例如`/path/to/model`，将从本地加载模型；
  - 评测目标为模型API服务时，需要指定为服务对应的模型id，例如`Qwen2.5-0.5B-Instruct`。
- `--model-id`: 被评测的模型的别名，用于报告展示。默认为`model`的最后一部分，例如`Qwen/Qwen2.5-0.5B-Instruct`的`model-id`为`Qwen2.5-0.5B-Instruct`。
- `--model-args`: 模型加载参数，以逗号分隔的`key=value`形式；或以json字符串格式传入，将解析为字典。默认参数：
  - `revision`: 模型版本，默认为`master`
  - `precision`: 模型精度，默认为`torch.float16`
  - `device_map`: 模型分配设备，默认为`auto`
- `--model-task`: 模型任务类型，默认为`text_generation`，可选`text_generation`, `image_generation`
- `--generation-config`: 生成参数，以逗号分隔的`key=value`形式；或以json字符串格式传入，将解析为字典:
  - 若使用本地模型推理（基于Transformers）包括如下参数（[全部参数指南](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig)）：
    - `do_sample`: 是否使用采样，默认为`false`
    - `max_length`: 最大长度，默认为2048
    - `max_new_tokens`: 生成最大长度，默认为512
    - `num_return_sequences`: 生成序列数量，默认为1；设置大于1时，将生成多个序列，需要设置`do_sample=True`
    - `temperature`: 生成温度
    - `top_k`: 生成top-k
    - `top_p`: 生成top-p
  - 若使用模型API服务推理（`eval-type`设置为`service`），包括如下参数（具体请参考部署的模型服务）：
    - `max_tokens`: 生成最大长度，默认为2048
    - `temperature`: 生成温度, 默认为0.0
    - `n`: 生成序列数量，默认为1（注意：lmdeploy目前仅支持n=1）
  ```bash
  # 例如用key=value形式传入
  --model-args revision=master,precision=torch.float16,device_map=auto
  --generation-config do_sample=true,temperature=0.5
  # 或者用json字符串传入更复杂的参数
  --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}'
  --generation-config '{"do_sample":true,"temperature":0.5,"chat_template_kwargs":{"enable_thinking": false}}'
  ```
- `--chat-template`: 模型推理模板，默认为`None`，表示使用transformers的`apply_chat_template`；支持传入jinjia模版字符串，来自定义推理模板
- `--template-type`: 模型推理模板，已弃用，参考`--chat-template`

**以下参数仅在`eval-type=service`时有效：**
- `--api-url`: 模型API端点，默认为`None`；支持传入本地或远端的OpenAI API格式端点，例如`http://127.0.0.1:8000/v1`。
- `--api-key`: 模型API端点密钥，默认为`EMPTY`
- `--timeout`: 模型API请求超时时间，默认为`None`
- `--stream`:  是否使用流式传输，默认为`False`

## 数据集参数
- `--datasets`: 数据集名称，支持输入多个数据集，使用空格分开，数据集将自动从modelscope下载，支持的数据集参考[数据集列表](./supported_dataset/llm.md)
- `--dataset-args`: 评测数据集的设置参数，以`json`字符串格式传入，将解析为字典，注意需要跟`--datasets`参数中的值对应：
  - `dataset_id` (或`local_path`): 可指定数据集本地路径，指定后将尝试从本地加载数据。
  - `prompt_template`: 评测数据集的prompt模板，指定后将使用模板生成prompt。例如`gsm8k`的模版为`Question: {query}\nLet's think step by step\nAnswer:`，数据集的问题将填充到模板`query`字段中。
  - `query_template`: 评测数据集的query模板，指定后将使用模板生成query。例如`general_mcq`的模版为`问题：{question}\n{choices}\n答案: {answer}\n\n`，数据集的问题将填充到模板`question`字段中，选项填充到`choices`字段中，答案填充到`answer`字段中（答案填充仅对few-shot生效）。
  - `system_prompt`: 评测数据集的系统prompt。
  - `model_adapter`: 评测数据集的模型适配器，指定后将使用给定的模型适配器评测，目前支持`generation`, `multiple_choice_logits`, `continuous_logits`；对于service评测，目前仅支持`generation`；部分多选题数据集支持`logits`输出。
  - `subset_list`: 评测数据子集列表，指定后将只使用子集数据。
  - `few_shot_num`: few-shot的数量。
  - `few_shot_random`: 是否随机采样few-shot数据，默认为`False`。
  - `metric_list`: 评测数据集的指标列表，指定后使用给定的指标评测，目前支持`AverageAccuracy`, `AveragePass@1`, `Pass@[1-16]`。例如`humaneval`数据集可指定`["Pass@1", "Pass@5"]`，注意此时需要指定`n=5`让模型返回5个结果。
  - `filters`: 评测数据集的过滤器，指定后将使用给定的过滤器过滤评测结果，可用来处理推理模型的输出，目前支持：
    - `remove_until {string}`: 过滤掉模型输出结果中指定字符串之前的部分。
    - `extract {regex}`: 提取模型输出结果中指定正则表达式匹配的部分。
    例如`ifeval`数据集可指定`{"remove_until": "</think>"}`，将过滤掉模型输出结果中`</think>`之前的部分，避免影响打分。
  ```bash
  # 例如
  --datasets gsm8k arc
  --dataset-args '{"gsm8k": {"few_shot_num": 4, "few_shot_random": false}, "arc": {"dataset_id": "/path/to/arc"}}, "ifeval": {"filters": {"remove_until": "</think>"}}'
  ```
- `--dataset-dir`: 数据集下载路径，默认为`~/.cache/modelscope/datasets`
- `--dataset-hub`: 数据集下载源，默认为`modelscope`，可选`huggingface`
- `--limit`: 每个数据集最大评测数据量，不填写则默认为全部评测，可用于快速验证。支持int和float类型，int表示评测数据集的前`N`条数据，float表示评测数据集的前`N%`条数据。例如`0.1`表示评测数据集的前10%的数据，`100`表示评测数据集的前100条数据。

## 评测参数
- `--eval-batch-size`: 评测批量大小，默认为`1`；在`eval-type=service`时，表示并发评测的请求数，默认为`8`
- `--eval-stage`: （已弃用，参考`--use-cache`）评测阶段，可选`all`, `infer`, `review`, 默认为`all`
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

## Judge参数
LLM-as-a-Judge评测参数，使用裁判模型来判断正误，包括以下参数：

- `--judge-strategy`: 使用裁判模型的策略，可选：
  - `auto`: 默认策略，根据数据集是否需要judge来决定是否使用裁判模型
  - `llm`: 总是使用裁判模型
  - `rule`: 不使用裁判模型，使用规则判断
  - `llm_recall`: 先使用规则判断，若规则判断失败再使用裁判模型
- `--judge-worker-num`: 裁判模型并发数，默认为`1`
- `--judge-model-args`: 设置裁判模型参数，以`json`字符串格式传入，将解析为字典，支持如下字段：
  - `api_key`: 模型API端点密钥，未设置时将从环境变量`MODELSCOPE_SDK_TOKEN`中读取，默认为`EMPTY`
  - `api_url`: 模型API端点，未设置时将从环境变量`MODELSCOPE_API_BASE`中读取，默认为`https://api-inference.modelscope.cn/v1/`
  - `model_id`: 模型ID，未设置时将从环境变量`MODELSCOPE_JUDGE_LLM`中读取，默认为`Qwen/Qwen3-235B-A22B`
    ```{seealso}
    关于ModelScope的模型推理服务的更多信息，请参考[ModelScope API推理服务](https://modelscope.cn/docs/model-service/API-Inference/intro)
    ```
  - `system_prompt`: 评测数据集的系统prompt
  - `prompt_template`: 评测数据集的prompt模板
  - `generation_config`: 模型生成参数，与`--generation-config`参数相同。
  - `score_type`: 预置的模型打分方式，可选：
    - `pattern`: （默认选项）直接判断模型输出与参考答案是否相同，适合有参考答案评测。
      <details><summary>默认prompt_template</summary>

      ```text
      Your job is to look at a question, a gold target, and a predicted answer, and return a letter "A" or "B" to indicate whether the predicted answer is correct or incorrect.

      [Question]
      {question}

      [Reference Answer]
      {gold}

      [Predicted Answer]
      {pred}

      Evaluate the model's answer based on correctness compared to the reference answer.
      Grade the predicted answer of this new question as one of:
      A: CORRECT
      B: INCORRECT

      Just return the letters "A" or "B", with no text around it.
      ```
      </details>
    - `numeric`: 判断模型输出在prompt条件下的打分，适合无参考答案评价。
      <details><summary>默认prompt_template</summary>

      ```text
      Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.

      Begin your evaluation by providing a short explanation. Be as objective as possible.

      After providing your explanation, you must rate the response on a scale of 0 (worst) to 1 (best) by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.5]]\"

      [Question]
      {question}

      [Response]
      {pred}
      ```
      </details>
  - `score_pattern`：解析模型输出的正则表达式，`pattern`模式默认为`(A|B)`；`numeric`模式默认为`\[\[(\d+(?:\.\d+)?)\]\]`，用于提取模型打分结果。
  - `score_mapping`：`pattern`模式下的分数映射字典，默认为`{'A': 1.0, 'B': 0.0}`
- `--analysis-report`: 是否生成分析报告，默认为`false`；如果设置该参数，将使用judge model生成分析报告，报告中包含模型评测结果的分析解读和建议。报告输出语言将根据`locale.getlocale()`自动判断。

## 其他参数

- `--work-dir`: 模型评测输出路径，默认为`./outputs/{timestamp}`，文件夹结构示例如下：
  ```text
  .
  ├── configs
  │   └── task_config_b6f42c.yaml
  ├── logs
  │   └── eval_log.log
  ├── predictions
  │   └── Qwen2.5-0.5B-Instruct
  │       └── general_qa_example.jsonl
  ├── reports
  │   └── Qwen2.5-0.5B-Instruct
  │       └── general_qa.json
  └── reviews
      └── Qwen2.5-0.5B-Instruct
          └── general_qa_example.jsonl
  ```
- `--use-cache`: 使用本地缓存的路径，默认为`None`；如果为指定路径，例如`outputs/20241210_194434`，将重用路径下的模型推理结果，若未完成推理则会继续推理，之后进行评测。
- `--seed`: 随机种子，默认为`42`
- `--debug`: 是否开启调试模式，默认为`false`
- `--ignore-errors`: 是否忽略模型生成过程中的错误，默认为`false`
- `--dry-run`: 预检参数，不进行推理，只打印参数，默认为`false`
