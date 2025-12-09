# 参数说明

执行 `evalscope eval --help` 可获取全部参数说明。

## 模型参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model` | `str` | 被评测的模型名称<br>• ModelScope模型ID（如`Qwen/Qwen2.5-0.5B-Instruct`）<br>• 本地模型路径（如`/path/to/model`）<br>• API服务的模型ID（如`Qwen2.5-0.5B-Instruct`） | - |
| `--model-id` | `str` | 模型别名，用于报告展示 | `model`的最后一部分 |
| `--api-url` | `str` | 模型API端点，支持OpenAI兼容格式<br>示例：`http://127.0.0.1:8000/v1` | `None` |
| `--api-key` | `str` | 模型API端点密钥 | `EMPTY` |
| `--model-args` | `str` | 模型加载参数，逗号分隔的`key=value`或JSON字符串<br>• `revision`: 模型版本<br>• `precision`: 模型精度<br>• `device_map`: 设备分配 | `revision=master`<br>`precision=torch.float16`<br>`device_map=auto` |
| `--model-task` | `str` | 模型任务类型 | `text_generation`<br>（可选：`image_generation`） |
| `--chat-template` | `str` | 模型推理模板，支持Jinja模板字符串 | `None`（使用transformers默认） |

**示例：**
```bash
# key=value形式
--model-args revision=master,precision=torch.float16,device_map=auto

# JSON字符串形式
--model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}'
```

## 模型推理参数

`--generation-config` 参数支持以下配置项（逗号分隔的`key=value`或JSON字符串）：

| 参数 | 类型 | 说明 | 支持的后端 |
|------|------|------|------------|
| `timeout` | `int` | 请求超时时间（秒） | 所有 |
| `retries` | `int` | 重试次数，默认为5 | OpenAI兼容 |
| `retry_interval` | `int` | 重试间隔时间（秒），默认10 | OpenAI兼容 |
| `stream` | `bool` | 是否流式返回响应 | 所有 |
| `max_tokens` | `int` | 最大生成token数量 | 所有 |
| `top_p` | `float` | Nucleus采样，考虑概率质量为top_p的token | 所有 |
| `temperature` | `float` | 采样温度，范围0~2，越高越随机 | 所有 |
| `frequency_penalty` | `float` | 范围-2.0~2.0，正值惩罚重复token | OpenAI兼容 |
| `presence_penalty` | `float` | 范围-2.0~2.0，正值惩罚已出现token | OpenAI兼容 |
| `logit_bias` | `dict` | token id到偏置值的映射（-100~100）<br>示例：`"42=10,43=-10"` | OpenAI兼容 |
| `seed` | `int` | 随机种子 | OpenAI兼容 |
| `do_sample` | `bool` | 是否采用采样策略（否则贪婪解码） | Transformers |
| `top_k` | `int` | 从top_k最可能的词中采样 | Anthropic, Google, HuggingFace, vLLM, SGLang |
| `logprobs` | `bool` | 是否返回输出token的对数概率 | OpenAI, Grok, TogetherAI, HuggingFace, llama-cpp-python, vLLM, SGLang |
| `top_logprobs` | `int` | 返回概率最高的前N个token（范围0~20） | OpenAI, Grok, HuggingFace, vLLM, SGLang |
| `parallel_tool_calls` | `bool` | 工具调用是否支持并行 | OpenAI, Groq |
| `max_tool_output` | `int` | 工具输出的最大字节数 | 所有（默认16*1024） |
| `extra_body` | `dict` | 向OpenAI兼容服务发送的额外请求体 | OpenAI兼容服务 |
| `extra_query` | `dict` | 向OpenAI兼容服务发送的额外查询参数 | OpenAI兼容服务 |
| `extra_headers` | `dict` | 向OpenAI兼容服务发送的额外请求头 | OpenAI兼容服务 |
| `height` | `int` | 图像生成模型专用，指定图像高度 | 图像生成模型 |
| `width` | `int` | 图像生成模型专用，指定图像宽度 | 图像生成模型 |
| `num_inference_steps` | `int` | 图像生成模型专用，推理步数 | 图像生成模型 |
| `guidance_scale` | `float` | 图像生成模型专用，指导尺度 | 图像生成模型 |

**示例：**
```bash
# key=value形式
--generation-config do_sample=true,temperature=0.5

# JSON字符串形式（支持更复杂参数）
--generation-config '{"do_sample":true,"temperature":0.5,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 数据集参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--datasets` | `list[str]` | 数据集名称列表，空格分隔<br>参考[数据集列表](./supported_dataset/llm.md) | - |
| `--dataset-dir` | `str` | 数据集下载路径 | `~/.cache/modelscope/datasets` |
| `--dataset-hub` | `str` | 数据集下载源 | `modelscope`<br>（可选：`huggingface`） |
| `--limit` | `int`/`float` | 每个数据集最大评测数据量<br>• int：评测前N条数据<br>• float：评测前N%数据<br>示例：`100`或`0.1` | `None`（全部评测） |
| `--repeats` | `int` | 重复推理一个样例多次 | `1` |
| `--dataset-args` | `str` | 数据集配置参数（JSON字符串），详见下表 | `{}` |

### dataset-args 配置项

`--dataset-args` 为JSON字符串，每个数据集可配置以下参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dataset_id` | `str` | 数据集modelscope id/本地路径 |
| `review_timeout` | `float` | 评测样本超时时间（秒），代码类任务建议设置 |
| `prompt_template` | `str` | Prompt模板，示例：`Question: {query}\nAnswer:` |
| `system_prompt` | `str` | 系统prompt |
| `subset_list` | `list[str]` | 评测数据子集列表 |
| `few_shot_num` | `int` | few-shot示例数量 |
| `few_shot_random` | `bool` | 是否随机采样few-shot数据 |
| `shuffle` | `bool` | 是否打乱数据 |
| `shuffle_choices` | `bool` | 是否打乱选项顺序（仅多选题） |
| `metric_list` | `list[str]` | 指标列表，默认支持`acc` |
| `aggregation` | `str` | 评测结果聚合方式，默认`mean`。可选：`mean_and_pass_at_k`、`mean_and_vote_at_k`、`mean_and_pass_hat_k`（均需设置`repeats=k`）。<br>• `pass_at_k`：同一样例生成k次至少一次通过的概率（如`humaneval`设`repeats=5`）<br>• `vote_at_k`：对同一样例k次结果投票后计分<br>• `pass_hat_k`：同一样例k次全部通过的概率（如`tau2_bench`设`repeats=3`） |
| `filters` | `dict` | 输出过滤器<br>• `remove_until`: 过滤指定字符串之前的内容<br>• `extract`: 提取正则匹配的内容 |
| `force_redownload` | `bool` | 是否强制重新下载数据集 |
| `extra_params` | `dict` | 数据集相关额外参数，参考[各数据集说明](./supported_dataset/index.md) |
| `sandbox_config` | `dict` | Sandbox配置（详见下方Sandbox参数） |

**sandbox_config 配置项：**

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `image` | `str` | Docker镜像名称 | `python:3.11-slim` |
| `network_enabled` | `bool` | 是否启用网络 | `true` |
| `tools_config` | `dict` | 工具配置字典 | `{'shell_executor': {}, 'python_executor': {}}` |

**示例：**
```bash
--datasets gsm8k arc ifeval hle \
--dataset-args '{
  "gsm8k": {
    "few_shot_num": 4,
    "few_shot_random": false
  },
  "arc": {
    "dataset_id": "/path/to/arc"
  },
  "ifeval": {
    "filters": {
      "remove_until": "</think>"
    }
  },
  "hle": {
    "extra_params": {
      "include_multi_modal": false
    }
  }
}'
```

## 评测参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--eval-type` | `str` | 评测类型<br>• `llm_ckpt`: 本地模型推理（transformers）<br>• `openai_api`: OpenAI兼容API服务<br>• `text2image`: 文本转图像模型（diffusers）<br>• `mock_llm`: 模拟推理（功能验证） | `None`（自动判断） |
| `--eval-batch-size` | `int` | 评测批量大小<br>`eval-type=service`时表示并发请求数 | `1`（service模式为`8`） |
| `--eval-backend` | `str` | 评测后端<br>• `Native`: 默认后端<br>• `OpenCompass`: 大语言模型评测<br>• `VLMEvalKit`: 多模态模型评测<br>• `RAGEval`: RAG/Embedding/Reranker/CLIP评测<br>• `ThirdParty`: 特殊任务评测 | `Native` |
| `--eval-config` | `str` | 非Native后端的配置文件路径 | - |

## Judge参数

LLM-as-a-Judge评测参数，使用裁判模型判断正误：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--judge-strategy` | `str` | 裁判模型策略<br>• `auto`: 根据数据集自动决定<br>• `llm`: 总是使用裁判模型<br>• `rule`: 只使用规则判断<br>• `llm_recall`: 规则失败后使用裁判模型 | `auto` |
| `--judge-worker-num` | `int` | 裁判模型并发数 | `1` |
| `--judge-model-args` | `str` | 裁判模型配置（JSON字符串），详见下表 | - |
| `--analysis-report` | `bool` | 是否生成分析报告（自动判断语言） | `false` |

### judge-model-args 配置项

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `api_key` | `str` | API密钥 | 从`MODELSCOPE_SDK_TOKEN`读取，默认`EMPTY` |
| `api_url` | `str` | API端点 | 从`MODELSCOPE_API_BASE`读取，<br>默认`https://api-inference.modelscope.cn/v1/` |
| `model_id` | `str` | 模型ID | 从`MODELSCOPE_JUDGE_LLM`读取，<br>默认`Qwen/Qwen3-235B-A22B` |
| `system_prompt` | `str` | 系统prompt | - |
| `prompt_template` | `str` | Prompt模板 | 根据`score_type`自动选择 |
| `generation_config` | `dict` | 生成参数（同`--generation-config`） | - |
| `score_type` | `str` | 打分方式<br>• `pattern`: 判断与参考答案是否相同<br>• `numeric`: 无参考答案打分（0-1） | `pattern` |
| `score_pattern` | `str` | 解析输出的正则表达式 | `pattern`模式：`(A\|B)`<br>`numeric`模式：`\[\[(\d+(?:\.\d+)?)\]\]` |
| `score_mapping` | `dict` | `pattern`模式的分数映射 | `{'A': 1.0, 'B': 0.0}` |

```{seealso}
关于ModelScope模型推理服务，请参考[ModelScope API推理服务](https://modelscope.cn/docs/model-service/API-Inference/intro)
```

<details><summary>pattern 模式默认prompt模板</summary>

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

<details><summary>numeric 模式默认prompt模板</summary>

```text
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.

Begin your evaluation by providing a short explanation. Be as objective as possible.

After providing your explanation, you must rate the response on a scale of 0 (worst) to 1 (best) by strictly following this format: "[[rating]]", for example: "Rating: [[0.5]]"

[Question]
{question}

[Response]
{pred}
```
</details>

## Sandbox参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--use-sandbox` | `bool` | 是否使用[ms-enclave](https://github.com/modelscope/ms-enclave)隔离代码运行环境<br>目前仅对代码评测任务（如`humaneval`）有效 | `false` |
| `--sandbox-manager-config` | `str` | Sandbox管理器配置（JSON字符串）<br>• `base_url`: 管理器URL（默认`None`为本地管理器） | `{}` |
| `--sandbox-type` | `str` | Sandbox类型 | `docker` |

## 其他参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--work-dir` | `str` | 评测输出路径（详见下方目录结构） | `./outputs` |
| `--no-timestamp` | `bool` | 是否不在工作目录中添加时间戳 | `false` |
| `--use-cache` | `str` | 复用本地缓存路径（如`outputs/20241210_194434`）<br>重用推理结果和评测结果 | `None` |
| `--rerun-review` | `bool` | 只重新运行评测（重用推理结果） | `false` |
| `--seed` | `int` | 随机种子 | `42` |
| `--debug` | `bool` | 是否开启调试模式 | `false` |
| `--ignore-errors` | `bool` | 是否忽略生成过程中的错误 | `false` |
| `--dry-run` | `bool` | 预检参数，不执行推理，只打印参数 | `false` |

### work-dir 目录结构示例

```text
./outputs/{timestamp}/
├── configs/
│   └── task_config_b6f42c.yaml      # 任务配置
├── logs/
│   └── eval_log.log                 # 评测日志
├── predictions/
│   └── {model_id}/
│       └── {dataset}.jsonl          # 模型推理结果
├── reports/
│   └── {model_id}/
│       └── {dataset}.json           # 评测报告
└── reviews/
    └── {model_id}/
        └── {dataset}.jsonl          # 评测结果详情
```
