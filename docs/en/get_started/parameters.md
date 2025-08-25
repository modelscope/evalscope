# Parameters

Run `evalscope eval --help` to get the full list of parameters.

## Model Parameters

- `--model`: The name of the model to be evaluated.
  - Specify the model's `id` on [ModelScope](https://modelscope.cn/) to download the model automatically, e.g. [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary).
  - Specify a local path to load the model from, e.g. `/path/to/model`.
  - For evaluating a model API service, specify the model id corresponding to the service, e.g. `Qwen2.5-0.5B-Instruct`.
- `--model-id`: Alias for the evaluated model, used in reports. Defaults to the last part of `model`, e.g. for `Qwen/Qwen2.5-0.5B-Instruct`, the `model-id` is `Qwen2.5-0.5B-Instruct`.
- `--api-url`: The model API endpoint, default is `None`; supports local or remote OpenAI API format endpoints, e.g. `http://127.0.0.1:8000/v1`.
- `--api-key`: The model API endpoint key, default is `EMPTY`.
- `--model-args`: Model loading parameters, either as comma-separated `key=value` pairs, or as a JSON string which will be parsed as a dictionary. Default parameters:
  - `revision`: Model revision, default is `master`.
  - `precision`: Model precision, default is `torch.float16`.
  - `device_map`: Device allocation for the model, default is `auto`.
- `--model-task`: Model task type, default is `text_generation`, optional values are `text_generation`, `image_generation`.
- `--chat-template`: Model inference template, default is `None` (uses transformers’ `apply_chat_template`); you can pass a Jinja template string to customize the inference template.

## Model Inference Parameters

- `--generation-config`: Generation parameters, either as comma-separated `key=value` pairs, or as a JSON string (parsed as a dictionary):
  - `timeout`: Optional integer, request timeout (seconds).
  - `stream`: Optional boolean, whether to return responses in streaming mode (depends on the model).
  - `system_message`: Optional string to override the default system message.
  - `max_tokens`: Optional integer, the maximum number of tokens generated (depends on the model).
  - `top_p`: Optional float, nucleus sampling; the model only considers tokens accounting for top_p probability mass.
  - `temperature`: Optional float, sampling temperature (0~2); higher means more randomness, lower means more deterministic output.
  - `frequency_penalty`: Optional float (-2.0~2.0); positive values penalize repeated tokens to reduce repetition. Supported by OpenAI, Google, Grok, Groq, vLLM, SGLang.
  - `presence_penalty`: Optional float (-2.0~2.0); positive values penalize already appeared tokens to encourage new topics. Supported by OpenAI, Google, Grok, Groq, vLLM, SGLang.
  - `logit_bias`: Optional dict mapping token ids to bias values (-100~100), e.g. `"42=10,43=-10"`. Supported by OpenAI, Grok, vLLM.
  - `seed`: Optional integer, random seed. Supported by OpenAI, Google, Mistral, Groq, HuggingFace, vLLM.
  - `do_sample`: Optional boolean, whether to use sampling strategy (otherwise greedy decoding). Supported by Transformers models.
  - `top_k`: Optional integer, sample next token from the top_k most likely candidates. Supported by Anthropic, Google, HuggingFace, vLLM, SGLang.
  - `logprobs`: Optional boolean, whether to return log probabilities for output tokens. Supported by OpenAI, Grok, TogetherAI, HuggingFace, llama-cpp-python, vLLM, SGLang.
  - `top_logprobs`: Optional integer (0~20), return the top top_logprobs tokens and their probabilities for each position. Supported by OpenAI, Grok, HuggingFace, vLLM, SGLang.
  - `parallel_tool_calls`: Optional boolean, whether to support parallel tool calls (default True). Supported by OpenAI, Groq.
  - `max_tool_output`: Optional integer, maximum bytes for tool output. Default is 16*1024.
  - `cache_prompt`: Optional "auto", boolean, or None; whether to cache prompt prefix. Default is "auto", enabled when using tools. Supported by Anthropic.
  - `reasoning_effort`: Optional enum (`'low'`, `'medium'`, `'high'`), restricts reasoning effort, default `'medium'`. Supported by OpenAI o1 models.
  - `reasoning_tokens`: Optional integer, max tokens for reasoning. Supported by Anthropic Claude models.
  - `reasoning_summary`: Optional enum (`'concise'`, `'detailed'`, `'auto'`); whether to provide a summary of reasoning steps. `'auto'` uses the most detailed summary available. Supported by OpenAI reasoning models.
  - `reasoning_history`: Optional enum (`'none'`, `'all'`, `'last'`, `'auto'`); whether reasoning content is included in chat message history.
  - `response_schema`: Optional ResponseSchema object, request returns formatted according to JSONSchema (output still requires validation). Supported by OpenAI, Google, Mistral.
  - `extra_body`: Optional dict, extra request body for OpenAI-compatible services. Supported by OpenAI, vLLM, SGLang.
  - `height`: Optional integer, for image generation models, specifies image height.
  - `width`: Optional integer, for image generation models, specifies image width.
  - `num_inference_steps`: Optional integer, for image models, number of inference steps.
  - `guidance_scale`: Optional float, for image models, guidance scale.

Example usage:
```bash
# Pass as key=value form
--model-args revision=master,precision=torch.float16,device_map=auto
--generation-config do_sample=true,temperature=0.5
# Or pass as JSON string for more complex parameters
--model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}'
--generation-config '{"do_sample":true,"temperature":0.5,"chat_template_kwargs":{"enable_thinking": false}}'
```

## Dataset Parameters

- `--datasets`: Dataset name(s), supports multiple datasets separated by spaces. Datasets will be automatically downloaded from ModelScope. Supported datasets are listed at [Dataset List](./supported_dataset/llm.md).
- `--dataset-args`: Evaluation dataset settings, passed as a JSON string and parsed into a dictionary. Must correspond to values in `--datasets`:
  - `dataset_id` (or `local_path`): Specify local path for the dataset; if set, data will be loaded locally.
  - `prompt_template`: Prompt template for the evaluation dataset; will be used to generate prompts. For example, `gsm8k`’s template is `Question: {query}\nLet's think step by step\nAnswer:` and the dataset question will be filled in the `query` field.
  - `query_template`: Query template for the evaluation dataset; used to generate queries. For example, `general_mcq`’s template is `问题：{question}\n{choices}\n答案: {answer}\n\n`, dataset question fills the `question` field, choices fill `choices`, and answer fills `answer` (answer is only filled for few-shot).
  - `system_prompt`: System prompt for the evaluation dataset.
  - `model_adapter`: Model adapter for the dataset; will use the specified adapter for evaluation. Currently supports `generation`, `multiple_choice_logits`, `continuous_logits`. For service evaluation, only `generation` is supported; some multiple choice datasets support `logits` output.
  - `subset_list`: List of dataset subsets; only data from specified subsets will be used.
  - `few_shot_num`: Number of few-shot samples.
  - `few_shot_random`: Whether to sample few-shot data randomly (default `False`).
  - `metric_list`: List of metrics for the dataset; will use specified metrics for evaluation. Currently supports `acc`, `Pass@k`, etc. For example, `humaneval` dataset can specify `["Pass@1", "Pass@5"]`. Note: you must set `repeats=5` to let the model infer 5 times for each sample.
  - `filters`: Filters for the dataset; will use specified filters to process results. Used to handle model output. Currently supported:
    - `remove_until {string}`: Remove everything in the model output before the specified string.
    - `extract {regex}`: Extract part of the model output matching the specified regex.
    For example, in `ifeval`, specify `{"remove_until": "</think>"}` to remove everything before `</think>` in model output, to avoid affecting scoring.
- `--dataset-dir`: Dataset download path, default is `~/.cache/modelscope/datasets`.
- `--dataset-hub`: Dataset source, default is `modelscope`, optional value is `huggingface`.
- `--limit`: Max number of samples to evaluate per dataset. If not set, evaluates all data. Supports int and float. Int means the first `N` samples, float means the first `N%` samples in the dataset. For example, `0.1` means the first 10% of samples, `100` means the first 100 samples.

Example usage:
```bash
--datasets gsm8k arc
--dataset-args '{"gsm8k": {"few_shot_num": 4, "few_shot_random": false}, "arc": {"dataset_id": "/path/to/arc"}}, "ifeval": {"filters": {"remove_until": "</think>"}}'
```

## Evaluation Parameters

- `--eval-type`: Evaluation type, choose based on model inference method, default is `llm_ckpt`:
  - `llm_ckpt`: Local model inference; downloads the model from ModelScope and uses Transformers for inference.
  - `openai_api`: Online model service inference; supports any OpenAI API-compatible service.
  - `text2image`: Local text-to-image model inference; downloads the model from ModelScope and uses Diffusers pipeline for inference.
  - `mock_llm`: Simulated LLM inference, for function verification.
- `--eval-batch-size`: Evaluation batch size, default is `1`; for `eval-type=service`, this means concurrent evaluation requests, default is `8`.
- `--eval-backend`: Evaluation backend, optional values are `Native`, `OpenCompass`, `VLMEvalKit`, `RAGEval`, `ThirdParty`; default is `Native`.
  - `OpenCompass` for LLM evaluation
  - `VLMEvalKit` for multimodal model evaluation
  - `RAGEval` for RAG pipeline, embedding, reranker, CLIP model evaluation
    ```{seealso}
    Refer to the [other backend usage guide](../user_guides/backend/index.md)
    ```
  - `ThirdParty` for other special tasks, e.g. [ToolBench](../third_party/toolbench.md), [LongBench](../third_party/longwriter.md)
- `--eval-config`: Required when using non-`Native` evaluation backend

## Judge Parameters

The LLM-as-a-Judge evaluation parameters use a judge model to determine correctness, including the following parameters:

- `--judge-strategy`: The strategy for using the judge model, options include:
  - `auto`: The default strategy, which decides whether to use the judge model based on the dataset requirements
  - `llm`: Always use the judge model
  - `rule`: Do not use the judge model, use rule-based judgment instead
  - `llm_recall`: First use rule-based judgment, and if it fails, then use the judge model
- `--judge-worker-num`: The concurrency number for the judge model, default is `1`
- `--judge-model-args`: Sets the parameters for the judge model, passed in as a `json` string and parsed as a dictionary, supporting the following fields:
  - `api_key`: The API endpoint key for the model. If not set, it will be retrieved from the environment variable `MODELSCOPE_SDK_TOKEN`, with a default value of `EMPTY`.
  - `api_url`: The API endpoint for the model. If not set, it will be retrieved from the environment variable `MODELSCOPE_API_BASE`, with a default value of `https://api-inference.modelscope.cn/v1/`.
  - `model_id`: The model ID. If not set, it will be retrieved from the environment variable `MODELSCOPE_JUDGE_LLM`, with a default value of `Qwen/Qwen3-235B-A22B`.
    ```{seealso}
    For more information on ModelScope's model inference services, please refer to [ModelScope API Inference Services](https://modelscope.cn/docs/model-service/API-Inference/intro).
    ```
  - `system_prompt`: System prompt for evaluating the dataset
  - `prompt_template`: Prompt template for evaluating the dataset
  - `generation_config`: Model generation parameters, same as the `--generation-config` parameter.
  - `score_type`: Preset model scoring method, options include:
    - `pattern`: (Default option) Directly judge whether the model output matches the reference answer, suitable for evaluations with reference answers.
      <details><summary>Default prompt_template</summary>

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
    - `numeric`: Judge the model output score under prompt conditions, suitable for evaluations without reference answers.
      <details><summary>Default prompt_template</summary>

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
  - `score_pattern`: Regular expression for parsing model output, default for `pattern` mode is `(A|B)`; default for `numeric` mode is `\[\[(\d+(?:\.\d+)?)\]\]`, used to extract model scoring results.
  - `score_mapping`: Score mapping dictionary for `pattern` mode, default is `{'A': 1.0, 'B': 0.0}`
- `--analysis-report`: Whether to generate an analysis report, default is `false`; if this parameter is set, an analysis report will be generated using the judge model, including analysis interpretation and suggestions for the model evaluation results. The report output language will be automatically determined based on `locale.getlocale()`.

## Other Parameters

- `--work-dir`: Model evaluation output path, default is `./outputs/{timestamp}`, folder structure example is as follows:
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
- `--use-cache`: Path to use for local caching, default is `None`. If a specific path is provided (e.g. `outputs/20241210_194434`), the model inference results and evaluation results in that path will be reused; if inference is incomplete, it will continue from where it left off, and then proceed to evaluation.
- `--rerun-review`: Boolean value. Set to True if you want to reuse the model inference results and only rerun the evaluation. Default is False; if evaluation results exist locally, evaluation will be skipped.
- `--seed`: Random seed, default is `42`.
- `--debug`: Whether to enable debug mode, default is `false`.
- `--ignore-errors`: Whether to ignore errors during model generation, default is `false`.
