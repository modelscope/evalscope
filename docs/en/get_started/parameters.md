# Parameters

Run `evalscope eval --help` to get a complete list of parameter descriptions.

## Model Parameters
- `--model`: Specifies the `model_id` of the model in [ModelScope](https://modelscope.cn/), which can be automatically downloaded, for example, [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary); it can also be a local path to the model, e.g., `/path/to/model`.
- `--model-args`: Model loading parameters, separated by commas in `key=value` format. Default parameters:
  - `revision`: Model version, default is `master`.
  - `precision`: Model precision, default is `auto`.
  - `device_map`: Device allocation for the model, default is `auto`.
- `--generation-config`: Generation parameters, separated by commas in `key=value` format. Default parameters:
  - `do_sample`: Whether to use sampling, default is `false`.
  - `max_length`: Maximum length, default is 2048.
  - `max_new_tokens`: Maximum length of generated tokens, default is 512.
- `--chat-template`: Model inference template, default is `None`, which means using transformers' `apply_chat_template`; supports passing a Jinja template string for custom inference templates.
- `--template-type`: Model inference template, deprecated, refer to `--chat-template`.

## Dataset Parameters
- `--datasets`: Dataset names, supporting multiple datasets separated by spaces. Datasets will be automatically downloaded from ModelScope; refer to the [Dataset List](./supported_dataset.md#1-native-supported-datasets) for supported datasets.
- `--dataset-args`: Settings for the evaluation dataset in JSON format, where the key is the dataset name and the value is the parameters. Note that they must correspond one-to-one with the values in the `--datasets` parameter:
  - `local_path`: Local path to the dataset. If specified, it will attempt to load local data.
  - `prompt_template`: Prompt template for the evaluation dataset. If specified, it will be prefixed to each evaluation data entry.
  - `subset_list`: List of subsets for evaluation data. If specified, only subset data will be used.
  - `few_shot_num`: Number of few-shot samples.
  - `few_shot_random`: Whether to randomly sample few-shot data. Defaults to `true` if not set.
- `--dataset-dir`: Path for downloading datasets, default is `~/.cache/modelscope/datasets`.
- `--dataset-hub`: Source for downloading datasets, default is `modelscope`, can also be `huggingface`.
- `--limit`: Maximum amount of evaluation data per dataset. If not specified, it defaults to evaluating all data, which can be used for quick validation.

## Evaluation Parameters
- `--eval-stage`: Evaluation stage, options are `all`, `infer`, `review`:
  - `all`: Full evaluation, including inference and evaluation.
  - `infer`: Only perform inference, no evaluation.
  - `review`: Only perform data evaluation, no inference.
- `--eval-type`: Evaluation type, options are `checkpoint`, `custom`, default is `checkpoint`.
- `--eval-backend`: Evaluation backend, options are `Native`, `OpenCompass`, `VLMEvalKit`, `RAGEval`, `ThirdParty`, default is `Native`.
  - `OpenCompass` is used for evaluating large language models.
  - `VLMEvalKit` is used for evaluating multimodal models.
  - `RAGEval` is used for evaluating RAG processes, embedding models, reranker models, CLIP models.
    ```{seealso}
    - Other evaluation backend [Usage Guide](../user_guides//backend/index.md)
    ```
  - `ThirdParty` is used for other special task evaluations, such as [ToolBench](../third_party/toolbench.md) and [LongBench](../third_party/longwriter.md).
- `--eval-config`: Required when using non-`Native` evaluation backends.

## Other Parameters
- `--work-dir`: Output path for model evaluation, default is `./outputs/{timestamp}`.
- `--use-cache`: Use local cache path, default is `None`; if a path is specified, such as `outputs/20241210_194434`, it will reuse the model inference results from that path. If inference is not completed, it will continue inference and then proceed to evaluation.
- `--seed`: Random seed, default is `42`.
- `--debug`: Whether to enable debug mode, default is `false`.
- `--dry-run`: Pre-check parameters without performing inference, only prints parameters, default is `false`.