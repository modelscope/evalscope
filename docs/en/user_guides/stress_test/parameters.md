# Parameter Description

Execute `evalscope perf --help` to get a full parameter description:

## Basic Settings
- `--model`: Name of the test model.
- `--url`: Specify the API address.
- `--name`: Name for the wandb database result and result database, default is `{model_name}_{current_time}`, optional.
- `--api`: Specify the service API, currently supports [openai|dashscope|local|local_vllm].
  - Select `openai` to use the API supporting OpenAI, requiring the `--url` parameter.
  - Select `dashscope` to use the API supporting DashScope, requiring the `--url` parameter.
  - Select `local` to use local files as models and perform inference using transformers. `--model` should be the model file path or model_id, which will be automatically downloaded from modelscope, e.g., `Qwen/Qwen2.5-0.5B-Instruct`.
  - Select `local_vllm` to use local files as models and start the vllm inference service. `--model` should be the model file path or model_id, which will be automatically downloaded from modelscope, e.g., `Qwen/Qwen2.5-0.5B-Instruct`.
  - You can also use a custom API, refer to [Custom API Guide](./custom.md#custom-request-api).
- `--port`: The port for the local inference service, defaulting to 8877. This is only applicable to `local` and `local_vllm`.
- `--attn-implementation`: Attention implementation method, default is None, optional [flash_attention_2|eager|sdpa], only effective when `api` is `local`.
- `--api-key`: API key, optional.
- `--debug`: Output debug information.

## Network Configuration
- `--connect-timeout`: Network connection timeout, default is 120 seconds.
- `--read-timeout`: Network read timeout, default is 120 seconds.
- `--headers`: Additional HTTP headers, formatted as `key1=value1 key2=value2`. This header will be used for each query.

## Request Control
- `--number`: Number of requests sent, default is None, meaning requests are sent based on the dataset size.
- `--parallel` sets the number of workers for concurrent requests, with a default of 1.
- `--rate` specifies the number of requests generated per second (not sent), with a default of -1, indicating that all requests will be generated at time 0 with no interval; otherwise, we use a Poisson process to generate request intervals.
  ```{tip}
  In the implementation of this tool, request generation and sending are separated:
  The `--rate` parameter is used to control the number of requests generated per second, and these requests will be placed in a request queue.
  The `--parallel` parameter is used to control the number of workers sending requests; workers will take requests from the queue and send them, only sending the next request after receiving a response for the previous one. It is not recommended to set both parameters simultaneously; thus, in this tool, the `--rate` parameter is only effective when `--parallel` is set to 1.
  ```
- `--log-every-n-query`: Log every n queries, default is 10.
- `--stream`: Use SSE stream output, default is False.

## Prompt Settings
- `--max-prompt-length`: Maximum input prompt length, default is `sys.maxsize`. Prompts exceeding this length will be discarded.
- `--min-prompt-length`: Minimum input prompt length, default is 0. Prompts shorter than this will be discarded.
- `--prompt`: Specify request prompt, a string or local file, taking precedence over `dataset`. When using a local file, specify the file path with `@/path/to/file`, e.g., `@./prompt.txt`.
- `--query-template`: Specify query template, a `JSON` string or local file. When using a local file, specify the file path with `@/path/to/file`, e.g., `@./query_template.json`.

## Dataset Configuration
- `--dataset` specifies the dataset [openqa|longalpaca|line_by_line|flickr8k]. You can also use a custom dataset parser in Python; refer to the [Custom Dataset Guide](./custom.md#custom-dataset).
  - `line_by_line` treats each line as a separate prompt and requires a `dataset_path`.
  - `longalpaca` will use `item['instruction']` as the prompt. If `dataset_path` is not specified, it will be automatically downloaded from modelscope.
  - `openqa` will use `item['question']` as the prompt. If `dataset_path` is not specified, it will be automatically downloaded from modelscope.
  - `flickr8k` will construct image-text inputs, making it suitable for evaluating multimodal models; the dataset will be automatically downloaded from modelscope, and specifying `dataset_path` is not supported.
- `--dataset-path`: Path to the dataset file, used in combination with the dataset. `openqa` and `longalpaca` do not need a specified dataset path and will be downloaded automatically; `line_by_line` requires a local dataset file, which will be loaded line by line.

## Model Settings
- `--tokenizer-path`: Optional, specify the path to tokenizer weights for calculating the number of input and output tokens, usually in the same directory as the model weights.
- `--frequency-penalty`: Frequency penalty value.
- `--logprobs`: Log probabilities.
- `--max-tokens`: Maximum number of tokens that can be generated.
- `--min-tokens`: Minimum number of tokens to be generated.
- `--n-choices`: Number of completion choices generated.
- `--seed`: Random seed, default is 42.
- `--stop`: Tokens that stop generation.
- `--stop-token-ids`: Set stop token IDs.
- `--temperature`: Sampling temperature.
- `--top-p`: Top_p sampling.
- `--top-k`: Top_k sampling.

## Data Storage
- `--wandb-api-key`: wandb API key, if set, metrics will be saved to wandb.
- `--outputs-dir` specifies the output file path, with a default value of `./outputs`.