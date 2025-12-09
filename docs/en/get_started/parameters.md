# Parameters

Run `evalscope eval --help` to get the full list of parameters.

## Model Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--model` | `str` | Name of the model to be evaluated<br>• ModelScope model ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)<br>• Local model path (e.g., `/path/to/model`)<br>• Model ID for API service (e.g., `Qwen2.5-0.5B-Instruct`) | - |
| `--model-id` | `str` | Alias for the evaluated model, used in reports | Last part of `model` |
| `--api-url` | `str` | Model API endpoint, supports OpenAI-compatible format<br>Example: `http://127.0.0.1:8000/v1` | `None` |
| `--api-key` | `str` | Model API endpoint key | `EMPTY` |
| `--model-args` | `str` | Model loading parameters, comma-separated `key=value` or JSON string<br>• `revision`: Model revision<br>• `precision`: Model precision<br>• `device_map`: Device allocation | `revision=master`<br>`precision=torch.float16`<br>`device_map=auto` |
| `--model-task` | `str` | Model task type | `text_generation`<br>(Options: `image_generation`) |
| `--chat-template` | `str` | Model inference template, supports Jinja template string | `None` (uses transformers default) |

**Example:**
```bash
# key=value format
--model-args revision=master,precision=torch.float16,device_map=auto

# JSON string format
--model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}'
```

## Model Inference Parameters

The `--generation-config` parameter supports the following options (comma-separated `key=value` or JSON string):

| Parameter | Type | Description | Supported Backends |
|-----------|------|-------------|--------------------|
| `timeout` | `int` | Request timeout (seconds) | All |
| `retries` | `int` | Number of retries, default is 5. | OpenAI-compatible |
| `retry_interval` | `int` | Retry interval (seconds), default is 10. | OpenAI-compatible |
| `stream` | `bool` | Whether to return responses in streaming mode | All |
| `max_tokens` | `int` | Maximum number of tokens generated | All |
| `top_p` | `float` | Nucleus sampling; only considers tokens accounting for top_p probability mass | All |
| `temperature` | `float` | Sampling temperature, range 0~2; higher means more randomness | All |
| `frequency_penalty` | `float` | Range -2.0~2.0; positive values penalize repeated tokens | OpenAI-compatible |
| `presence_penalty` | `float` | Range -2.0~2.0; positive values penalize already appeared tokens | OpenAI-compatible |
| `logit_bias` | `dict` | Mapping of token IDs to bias values (-100~100)<br>Example: `"42=10,43=-10"` | OpenAI, Grok, vLLM |
| `seed` | `int` | Random seed | OpenAI, Google, Mistral, Groq, HuggingFace, vLLM |
| `do_sample` | `bool` | Whether to use sampling strategy (otherwise greedy decoding) | Transformers |
| `top_k` | `int` | Sample next token from the top_k most likely candidates | Anthropic, Google, HuggingFace, vLLM, SGLang |
| `logprobs` | `bool` | Whether to return log probabilities for output tokens | OpenAI, Grok, TogetherAI, HuggingFace, llama-cpp-python, vLLM, SGLang |
| `top_logprobs` | `int` | Return the top N tokens and their probabilities (range 0~20) | OpenAI, Grok, HuggingFace, vLLM, SGLang |
| `parallel_tool_calls` | `bool` | Whether to support parallel tool calls | OpenAI, Groq |
| `max_tool_output` | `int` | Maximum bytes for tool output | All (default 16*1024) |
| `extra_body` | `dict` | Extra request body for OpenAI-compatible services | OpenAI-compatible services |
| `extra_query` | `dict` | Extra query parameters for OpenAI-compatible services | OpenAI-compatible services |
| `extra_headers` | `dict` | Extra headers for OpenAI-compatible services | OpenAI-compatible services |
| `height` | `int` | For image generation models, specifies image height | Image generation models |
| `width` | `int` | For image generation models, specifies image width | Image generation models |
| `num_inference_steps` | `int` | For image models, number of inference steps | Image generation models |
| `guidance_scale` | `float` | For image models, guidance scale | Image generation models |

**Example:**
```bash
# key=value format
--generation-config do_sample=true,temperature=0.5

# JSON string format (supports more complex parameters)
--generation-config '{"do_sample":true,"temperature":0.5,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Dataset Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--datasets` | `list[str]` | Dataset name list, space-separated<br>Refer to [Dataset List](./supported_dataset/llm.md) | - |
| `--dataset-dir` | `str` | Dataset download path | `~/.cache/modelscope/datasets` |
| `--dataset-hub` | `str` | Dataset source | `modelscope`<br>(Options: `huggingface`) |
| `--limit` | `int`/`float` | Maximum samples to evaluate per dataset<br>• int: First N samples<br>• float: First N% of samples<br>Example: `100` or `0.1` | `None` (evaluate all) |
| `--repeats` | `int` | Number of times to repeat inference on the same sample | `1` |
| `--dataset-args` | `str` | Dataset configuration parameters (JSON string), see table below | `{}` |

### dataset-args Configuration Options

`--dataset-args` is a JSON string; each dataset can be configured with the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_id` | `str` | ModelScope dataset ID or local path |
| `review_timeout` | `float` | Timeout for evaluation samples (seconds), recommended for code tasks |
| `prompt_template` | `str` | Prompt template, example: `Question: {query}\nAnswer:` |
| `system_prompt` | `str` | System prompt |
| `subset_list` | `list[str]` | List of dataset subsets to evaluate |
| `few_shot_num` | `int` | Number of few-shot examples |
| `few_shot_random` | `bool` | Whether to randomly sample few-shot data |
| `shuffle` | `bool` | Whether to shuffle the data |
| `shuffle_choices` | `bool` | Whether to shuffle choice order (multiple-choice only) |
| `metric_list` | `list[str]` | Metric list, default supports `acc` |
| `aggregation` | `str` | Aggregation method for evaluation results, default is `mean`. Options: `mean_and_pass_at_k`, `mean_and_vote_at_k`, `mean_and_pass_hat_k` (all require setting `repeats=k`).<br>• `pass_at_k`: Probability that the same sample passes at least once in k generations (e.g., set `repeats=5` for `humaneval`)<br>• `vote_at_k`: Scoring by voting on k results for the same sample<br>• `pass_hat_k`: Probability that the same sample passes all k times (e.g., set `repeats=3` for `tau2_bench`) |
| `filters` | `dict` | Output filters<br>• `remove_until`: Remove content before specified string<br>• `extract`: Extract regex-matched content |
| `force_redownload` | `bool` | Whether to force re-download the dataset |
| `extra_params` | `dict` | Dataset-related extra parameters, refer to [dataset documentation](./supported_dataset/index.md) |
| `sandbox_config` | `dict` | Sandbox configuration (see Sandbox Parameters below) |

**sandbox_config Options:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `image` | `str` | Docker image name | `python:3.11-slim` |
| `network_enabled` | `bool` | Whether to enable networking | `true` |
| `tools_config` | `dict` | Tool configuration dictionary | `{'shell_executor': {}, 'python_executor': {}}` |

**Example:**
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

## Evaluation Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--eval-type` | `str` | Evaluation type<br>• `llm_ckpt`: Local model inference (transformers)<br>• `openai_api`: OpenAI-compatible API service<br>• `text2image`: Text-to-image model (diffusers)<br>• `mock_llm`: Simulated inference (for verification) | `None` (auto-detect) |
| `--eval-batch-size` | `int` | Evaluation batch size<br>For `eval-type=service`, means concurrent requests | `1` (service mode: `8`) |
| `--eval-backend` | `str` | Evaluation backend<br>• `Native`: Default backend<br>• `OpenCompass`: LLM evaluation<br>• `VLMEvalKit`: Multimodal model evaluation<br>• `RAGEval`: RAG/Embedding/Reranker/CLIP evaluation<br>• `ThirdParty`: Special task evaluation | `Native` |
| `--eval-config` | `str` | Configuration file path for non-Native backends | - |

```{seealso}
Refer to the [other backend usage guide](../user_guides/backend/index.md)
```

## Judge Parameters

LLM-as-a-Judge evaluation parameters using a judge model to determine correctness:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--judge-strategy` | `str` | Judge model strategy<br>• `auto`: Automatically decide based on dataset requirements<br>• `llm`: Always use judge model<br>• `rule`: Use rule-based judgment only<br>• `llm_recall`: Use judge model after rule-based judgment fails | `auto` |
| `--judge-worker-num` | `int` | Judge model concurrency | `1` |
| `--judge-model-args` | `str` | Judge model configuration (JSON string), see table below | - |
| `--analysis-report` | `bool` | Whether to generate analysis report (language auto-detected) | `false` |

### judge-model-args Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `api_key` | `str` | API key | Read from `MODELSCOPE_SDK_TOKEN`, default `EMPTY` |
| `api_url` | `str` | API endpoint | Read from `MODELSCOPE_API_BASE`,<br>default `https://api-inference.modelscope.cn/v1/` |
| `model_id` | `str` | Model ID | Read from `MODELSCOPE_JUDGE_LLM`,<br>default `Qwen/Qwen3-235B-A22B` |
| `system_prompt` | `str` | System prompt | - |
| `prompt_template` | `str` | Prompt template | Auto-selected based on `score_type` |
| `generation_config` | `dict` | Generation parameters (same as `--generation-config`) | - |
| `score_type` | `str` | Scoring method<br>• `pattern`: Judge if answer matches reference<br>• `numeric`: Score without reference (0-1) | `pattern` |
| `score_pattern` | `str` | Regex to parse output | `pattern` mode: `(A\|B)`<br>`numeric` mode: `\[\[(\d+(?:\.\d+)?)\]\]` |
| `score_mapping` | `dict` | Score mapping for `pattern` mode | `{'A': 1.0, 'B': 0.0}` |

```{seealso}
For more information on ModelScope model inference services, refer to [ModelScope API Inference Services](https://modelscope.cn/docs/model-service/API-Inference/intro)
```

<details><summary>pattern Mode Default Prompt Template</summary>

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

<details><summary>numeric Mode Default Prompt Template</summary>

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

## Sandbox Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--use-sandbox` | `bool` | Whether to use [ms-enclave](https://github.com/modelscope/ms-enclave) to isolate code execution environment<br>Currently only effective for code evaluation tasks (e.g., `humaneval`) | `false` |
| `--sandbox-manager-config` | `str` | Sandbox manager configuration (JSON string)<br>• `base_url`: Manager URL (default `None` for local manager) | `{}` |
| `--sandbox-type` | `str` | Sandbox type | `docker` |

## Other Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--work-dir` | `str` | Evaluation output path (see directory structure below) | `./outputs` |
| `--no-timestamp` | `bool` | Do not add timestamp to work_dir | `false` |
| `--use-cache` | `str` | Reuse local cache path (e.g., `outputs/20241210_194434`)<br>Reuses inference and evaluation results | `None` |
| `--rerun-review` | `bool` | Only rerun evaluation (reuses inference results) | `false` |
| `--seed` | `int` | Random seed | `42` |
| `--debug` | `bool` | Whether to enable debug mode | `false` |
| `--ignore-errors` | `bool` | Whether to ignore errors during generation | `false` |
| `--dry-run` | `bool` | Dry run to check parameters without executing inference | `false` |

### work-dir Directory Structure Example

```text
./outputs/{timestamp}/
├── configs/
│   └── task_config_b6f42c.yaml      # Task configuration
├── logs/
│   └── eval_log.log                 # Evaluation log
├── predictions/
│   └── {model_id}/
│       └── {dataset}.jsonl          # Model inference results
├── reports/
│   └── {model_id}/
│       └── {dataset}.json           # Evaluation report
└── reviews/
    └── {model_id}/
        └── {dataset}.jsonl          # Evaluation result details
```
