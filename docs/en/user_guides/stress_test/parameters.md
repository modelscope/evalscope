# Parameter Description

Execute `evalscope perf --help` to get a full parameter description:

## Basic Settings
- `--model`: Name of the test model.
- `--url` specifies the API address, supporting two types of endpoints: `/chat/completion` and `/completion`.
- `--name`: Name for the wandb/swanlab database result and result database, default is `{model_name}_{current_time}`, optional.
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
- `--connect-timeout`: Network connection timeout, default is 600 seconds.
- `--read-timeout`: Network read timeout, default is 600 seconds.
- `--headers`: Additional HTTP headers, formatted as `key1=value1 key2=value2`. This header will be used for each query.
- `--no-test-connection`: Do not send a connection test, start the stress test directly, default is False.

## Request Control
- `--parallel` specifies the number of concurrent requests, and you can input multiple values separated by spaces; the default is 1.
- `--number` indicates the total number of requests to be sent, and you can input multiple values separated by spaces (must correspond one-to-one with `parallel`); the default is 1000.
- `--rate` defines the number of requests generated per second (without sending them), with a default of -1, meaning all requests are generated at time 0 with no interval; otherwise, a Poisson process is used to generate request intervals.
  ```{tip}
  In the implementation of this tool, request generation and sending are separate:
  The `--rate` parameter controls the number of requests generated per second, which are placed in a request queue.
  The `--parallel` parameter controls the number of workers sending requests, with each worker retrieving requests from the queue and sending them, only proceeding to the next request after receiving a response to the previous one.
  ```
- `--log-every-n-query`: Log every n queries, default is 10.
- `--stream` uses SSE (Server-Sent Events) stream output, default is True. Note: Setting `--stream` is necessary to measure the Time to First Token (TTFT) metric; setting `--no-stream` will disable streaming output.
- `--sleep-interval`: The sleep time between each performance test, in seconds, default is 5 seconds. This parameter can help avoid overloading the server.

## Prompt Settings
- `--max-prompt-length`: The maximum input prompt length, default is `131072`. Prompts exceeding this length will be discarded.
- `--min-prompt-length`: The minimum input prompt length, default is 0. Prompts shorter than this will be discarded.
- `--prefix-length`: The length of the prompt prefix, default is 0. This is only effective for the `random` dataset.
- `--prompt`: Specifies the request prompt, which can be a string or a local file. This has higher priority than `dataset`. When using a local file, specify the file path with `@/path/to/file`, e.g., `@./prompt.txt`.
- `--query-template`: Specifies the query template, which can be a `JSON` string or a local file. When using a local file, specify the file path with `@/path/to/file`, e.g., `@./query_template.json`.
- `--apply-chat-template` determines whether to apply the chat template, default is None. It will automatically choose based on whether the URL suffix is `chat/completion`.

## Dataset Configuration
Here's the English translation:

- `--dataset` can specify the following dataset modes. You can also use a custom Python dataset parser, refer to the [Custom Dataset Guide](./custom.md#custom-dataset).
  - `openqa` uses the `question` field of a jsonl file as the prompt. If `dataset_path` is not specified, it will automatically download the [dataset](https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/summary) from ModelScope. The prompt length is relatively short, generally under 100 tokens.
  - `longalpaca` uses the `instruction` field of a jsonl file as the prompt. If `dataset_path` is not specified, it will automatically download the [dataset](https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/dataPeview) from ModelScope. The prompt length is relatively long, generally over 6000 tokens.
  - `flickr8k` will construct image-text input, suitable for evaluating multimodal models; it automatically downloads the [dataset](https://www.modelscope.cn/datasets/clip-benchmark/wds_flickr8k/dataPeview) from ModelScope and does not support specifying `dataset_path`.
  - `line_by_line` requires providing `dataset_path`, and uses each line of the txt file as a prompt.
  - `random` generates prompts randomly based on `prefix-length`, `max-prompt-length`, and `min-prompt-length`. It requires specifying `tokenizer-path`. [Usage example](./examples.md#using-the-random-dataset).
- `--dataset-path` is the path to the dataset file, used in conjunction with the dataset.

## Model Settings
- `--tokenizer-path`: Optional. Specifies the tokenizer weights path, used to calculate the number of tokens in the input and output, usually located in the same directory as the model weights.
- `--frequency-penalty`: The frequency_penalty value.
- `--logprobs`: Logarithmic probabilities.
- `--max-tokens`: The maximum number of tokens that can be generated.
- `--min-tokens`: The minimum number of tokens to generate. Not all model services support this parameter; please check the corresponding API documentation. For `vLLM>=0.8.1` versions, you need to additionally set `--extra-args '{"ignore_eos": true}'`.
- `--n-choices`: The number of completion choices to generate.
- `--seed`: The random seed, default is 0.
- `--stop`: Tokens that stop the generation.
- `--stop-token-ids`: Sets the IDs of tokens that stop the generation.
- `--temperature`: Sampling temperature, default is 0.0
- `--top-p`: Top-p sampling.
- `--top-k`: Top-k sampling.
- `--extra-args`: Additional parameters to be passed in the request body, formatted as a JSON string. For example: `'{"ignore_eos": true}'`.

## Data Storage
- `--wandb-api-key`: wandb API key, if set, metrics will be saved to wandb.
- `--swanlab-api-key`: swanlab API key, if set, metrics will be saved to swanlab.
- `--outputs-dir` specifies the output file path, with a default value of `./outputs`.
