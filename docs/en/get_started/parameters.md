# Parameters

Run `evalscope eval --help` to get a complete list of parameter descriptions.

## Model Parameters
- `--model`: The name of the model being evaluated.
  - Specify the model's `id` in [ModelScope](https://modelscope.cn/), and it will automatically download the model, for example, [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary);
  - Specify the local path to the model, for example, `/path/to/model`, to load the model from the local environment;
  - When the evaluation target is the model API endpoint, it needs to be specified as `model_id`, for example, `Qwen2.5-0.5B-Instruct`.
- `--model-id`: An alias for the model being evaluated. Defaults to the last part of `model`, for example, the `model-id` for `Qwen/Qwen2.5-0.5B-Instruct` is `Qwen2.5-0.5B-Instruct`.
- `--model-args`: Model loading parameters, separated by commas in `key=value` format, with default parameters:
  - `revision`: Model version, defaults to `master`
  - `precision`: Model precision, defaults to `torch.float16`
  - `device_map`: Device allocation for the model, defaults to `auto`
- `--generation-config`: Generation parameters, separated by commas in `key=value` format, with default parameters:
  - `do_sample`: Whether to use sampling, defaults to `false`
  - `max_length`: Maximum length, defaults to 2048
  - `max_new_tokens`: Maximum length of generation, defaults to 512
- `--chat-template`: Model inference template, defaults to `None`, indicating the use of transformers' `apply_chat_template`; supports passing in a jinja template string to customize the inference template.
- `--template-type`: Model inference template, deprecated, refer to `--chat-template`.
- `--api-url`: (Valid only when `eval-type=service`) Model API endpoint, defaults to `None`; supports passing in local or remote OpenAI API format endpoints, for example, `http://127.0.0.1:8000/v1/chat/completions`.
- `--api-key`: (Valid only when `eval-type=service`) Model API endpoint key, defaults to `EMPTY`.

## Dataset Parameters
- `--datasets`: Dataset name, supports inputting multiple datasets separated by spaces, datasets will automatically be downloaded from ModelScope, supported datasets refer to [Dataset List](./supported_dataset.md#supported-datasets).
- `--dataset-args`: Configuration parameters for the evaluation dataset, passed in `json` format, where the key is the dataset name and the value is the parameter, note that it needs to correspond one-to-one with the values in the `--datasets` parameter:
  - `local_path`: Local path for the dataset, once specified, it will attempt to load local data.
  - `prompt_template`: Prompt template for the evaluation dataset, once specified, it will be concatenated before each evaluation data entry.
  - `subset_list`: List of subsets for the evaluation dataset, once specified, only subset data will be used.
  - `few_shot_num`: Number of few-shots.
  - `few_shot_random`: Whether to randomly sample few-shot data, defaults to `False`.
- `--dataset-dir`: Dataset download path, defaults to `~/.cache/modelscope/datasets`.
- `--dataset-hub`: Dataset download source, defaults to `modelscope`, alternative is `huggingface`.
- `--limit`: Maximum evaluation data amount for each dataset, if not specified, defaults to all data for evaluation, can be used for quick validation.

## Evaluation Parameters
- `--eval-stage`: Evaluation stage, options are `all`, `infer`, `review`
  - `all`: Complete evaluation, including inference and evaluation.
  - `infer`: Only perform inference, without evaluation.
  - `review`: Only perform data evaluation, without inference.
- `--eval-type`: Evaluation type, options are `checkpoint`, `custom`, `service`; defaults to `checkpoint`.
- `--eval-backend`: Evaluation backend, options are `Native`, `OpenCompass`, `VLMEvalKit`, `RAGEval`, `ThirdParty`, defaults to `Native`.
  - `OpenCompass` is used for evaluating large language models.
  - `VLMEvalKit` is used for evaluating multimodal models.
  - `RAGEval` is used for evaluating RAG processes, embedding models, re-ranking models, CLIP models.
    ```{seealso}
    Other evaluation backends [User Guide](../user_guides/backend/index.md)
    ```
  - `ThirdParty` is used for other special task evaluations, such as [ToolBench](../third_party/toolbench.md), [LongBench](../third_party/longwriter.md).
- `--eval-config`: This parameter needs to be passed when using a non-`Native` evaluation backend.

## Other Parameters
- `--work-dir`: Output path for model evaluation, default is `./outputs/{timestamp}`.
- `--use-cache`: Use local cache path, default is `None`; if a path is specified, such as `outputs/20241210_194434`, it will reuse the model inference results from that path. If inference is not completed, it will continue inference and then proceed to evaluation.
- `--seed`: Random seed, default is `42`.
- `--debug`: Whether to enable debug mode, default is `false`.
- `--dry-run`: Pre-check parameters without performing inference, only prints parameters, default is `false`.