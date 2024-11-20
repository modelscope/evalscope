# Model Inference Performance Stress Test

A stress testing tool for large language models that can be customized to support various dataset formats and different API protocol formats, with default support for OpenAI API format.

## Quick Start

The model inference performance stress test tool can be launched using the following two methods:

::::{tab-set}
:::{tab-item} Command Line
```bash
evalscope perf \
    --url "http://127.0.0.1:8000/v1/chat/completions" \
    --parallel 1 \
    --model qwen2.5 \
    --number 15 \
    --api openai \
    --dataset openqa \
    --stream
```
:::

:::{tab-item} Python Script
```python
from evalscope.perf.main import run_perf_benchmark

task_cfg = {"url": "http://127.0.0.1:8000/v1/chat/completions",
            "parallel": 1,
            "model": "qwen2.5",
            "number": 15,
            "api": "openai",
            "dataset": "openqa",
            "stream": True}
run_perf_benchmark(task_cfg)
```
:::
::::

Parameter Description:

- `url`: URL address for requests
- `parallel`: Number of parallel request tasks
- `model`: Name of the model used
- `number`: Number of requests
- `api`: API service used
- `dataset`: Name of the dataset
- `stream`: Whether to enable streaming

### Output Results
```text
Benchmarking summary: 
+----------------------------------------------+------------------------------------------------+
| key                                          | Value                                          |
+==============================================+================================================+
| Time taken for tests (seconds)               | 7.539                                          |
+----------------------------------------------+------------------------------------------------+
| Number of concurrency                        | 1                                              |
+----------------------------------------------+------------------------------------------------+
| Total requests                               | 15                                             |
+----------------------------------------------+------------------------------------------------+
| Succeeded requests                           | 15                                             |
+----------------------------------------------+------------------------------------------------+
| Failed requests                              | 0                                              |
+----------------------------------------------+------------------------------------------------+
| Average QPS                                  | 1.99                                           |
+----------------------------------------------+------------------------------------------------+
| Average latency                              | 0.492                                          |
+----------------------------------------------+------------------------------------------------+
| Average time to first token                  | 0.026                                          |
+----------------------------------------------+------------------------------------------------+
| Throughput (average output tokens per second)| 334.006                                        |
+----------------------------------------------+------------------------------------------------+
| Average time per output token                | 0.00299                                        |
+----------------------------------------------+------------------------------------------------+
| Average package per request                  | 167.867                                        |
+----------------------------------------------+------------------------------------------------+
| Average package latency                      | 0.003                                          |
+----------------------------------------------+------------------------------------------------+
| Average input tokens per request             | 40.133                                         |
+----------------------------------------------+------------------------------------------------+
| Average output tokens per request            | 167.867                                        |
+----------------------------------------------+------------------------------------------------+
| Expected number of requests                  | 15                                             |
+----------------------------------------------+------------------------------------------------+
| Result DB path                               | ./outputs/qwen2.5_benchmark_20241107_201413.db |
+----------------------------------------------+------------------------------------------------+

Percentile results: 
+------------+---------------------+---------+
| Percentile | First Chunk Latency | Latency |
+------------+---------------------+---------+
|    10%     |       0.0178        | 0.1577  |
|    25%     |       0.0183        | 0.2358  |
|    50%     |       0.0199        | 0.4311  |
|    66%     |       0.0218        | 0.6317  |
|    75%     |       0.0429        | 0.7121  |
|    80%     |       0.0432        | 0.7957  |
|    90%     |       0.0432        | 0.9153  |
|    95%     |       0.0433        | 0.9897  |
|    98%     |       0.0433        | 0.9897  |
|    99%     |       0.0433        | 0.9897  |
+------------+---------------------+---------+
```

### Metrics Explanation

| **Metric**                                  | **Description**                | **Value**       |
|---------------------------------------------|--------------------------------|-----------------|
| Total requests                              | Total number of requests       | 15              |
| Succeeded requests                          | Number of successful requests  | 15              |
| Failed requests                             | Number of failed requests      | 0               |
| Average QPS                                 | Average requests per second    | 1.99            |
| Average latency                             | Average latency of all requests| 0.492           |
| Throughput (average output tokens per second)| Number of output tokens per second| 334.006      |
| Average time to first token                 | Average time to first token    | 0.026           |
| Average input tokens per request            | Average input tokens per request| 40.133         |
| Average output tokens per request           | Average output tokens per request| 167.867       |
| Average time per output token               | Average time per output token  | 0.00299         |
| Average package per request                 | Average packages per request   | 167.867         |
| Average package latency                     | Average package latency        | 0.003           |
| Percentile of time to first token (p10, ..., p99)| Percentile for first token delay|               |
| Percentile of request latency (p10, ..., p99)| Percentile for request latency |                 |


## Full Parameter Description

Execute `evalscope perf --help` to get a full parameter description:

### Basic Settings
- `--model`: Name of the test model.
- `--url`: Specify the API address.
- `--name`: Name for the wandb database result and result database, default is `{model_name}_{current_time}`, optional.
- `--api`: Specify the service API, currently supports [openai|dashscope|local|local_vllm].
  - Select `openai` to use the API supporting OpenAI, requiring the `--url` parameter.
  - Select `dashscope` to use the API supporting DashScope, requiring the `--url` parameter.
  - Select `local` to use local files as models and perform inference using transformers. `--model` should be the model file path or model_id, which will be automatically downloaded from modelscope, e.g., `Qwen/Qwen2.5-0.5B-Instruct`.
  - Select `local_vllm` to use local files as models and start the vllm inference service. `--model` should be the model file path or model_id, which will be automatically downloaded from modelscope, e.g., `Qwen/Qwen2.5-0.5B-Instruct`.
- `--attn-implementation`: Attention implementation method, default is None, optional [flash_attention_2|eager|sdpa], only effective when `api` is `local`.
- `--api-key`: API key, optional.
- `--debug`: Output debug information.

### Network Configuration
- `--connect-timeout`: Network connection timeout, default is 120 seconds.
- `--read-timeout`: Network read timeout, default is 120 seconds.
- `--headers`: Additional HTTP headers, formatted as `key1=value1 key2=value2`. This header will be used for each query.

### Request Control
- `--number`: Number of requests sent, default is None, meaning requests are sent based on the dataset size.
- `--parallel`: Set the number of concurrent requests, default is 1.
- `--rate`: Number of requests per second, default is None. If set to -1, all requests are sent at time 0. Otherwise, requests are synthesized using a Poisson process. Mutually exclusive with parallel.
- `--log-every-n-query`: Log every n queries, default is 10.
- `--stream`: Use SSE stream output, default is False.

### Prompt Settings
- `--max-prompt-length`: Maximum input prompt length, default is `sys.maxsize`. Prompts exceeding this length will be discarded.
- `--min-prompt-length`: Minimum input prompt length, default is 0. Prompts shorter than this will be discarded.
- `--prompt`: Specify request prompt, a string or local file, taking precedence over `dataset`. When using a local file, specify the file path with `@/path/to/file`, e.g., `@./prompt.txt`.
- `--query-template`: Specify query template, a `JSON` string or local file. When using a local file, specify the file path with `@/path/to/file`, e.g., `@./query_template.json`.

### Dataset Configuration
- `--dataset`: Specify the dataset [openqa|longalpaca|line_by_line]. You can define a custom dataset parser using Python and specify the Python file path, referring to `dataset_plugin_base.py`.
- `--dataset-path`: Path to the dataset file, used in combination with the dataset. `openqa` and `longalpaca` do not need a specified dataset path and will be downloaded automatically; `line_by_line` requires a local dataset file, which will be loaded line by line.

### Model Settings
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

### Data Storage
- `--wandb-api-key`: wandb API key, if set, metrics will be saved to wandb.

## Example

### Using Local Model Inference

This project supports inference using local transformers and vllm (vllm needs to be installed first). The `--model` can be filled with a modelscope model name, such as `Qwen/Qwen2.5-0.5B-Instruct`; or you can directly specify the model weight path, such as `/path/to/model_weights`, without needing to specify the `--url` parameter.

**Inference using transformers**

```bash
evalscope perf \
 --model 'Qwen/Qwen2.5-0.5B-Instruct' \
 --attn-implementation flash_attention_2 \  # Optional, or choose from [flash_attention_2|eager|sdpa]
 --number 20 \
 --rate 2 \
 --api local \
 --dataset openqa
```

**Inference using vllm**
```bash
evalscope perf \
 --model 'Qwen/Qwen2.5-0.5B-Instruct' \
 --number 20 \
 --rate 2 \
 --api local_vllm \
 --dataset openqa
```

### Using `prompt`
```bash
evalscope perf \
 --url 'http://127.0.0.1:8000/v1/chat/completions' \
 --rate 2 \
 --model 'qwen2.5' \
 --log-every-n-query 10 \
 --number 20 \
 --api openai \
 --temperature 0.9 \
 --max-tokens 1024 \
 --prompt 'Write a science fiction story, please begin your performance'
```
You can also use a local file as a prompt:
```bash
evalscope perf \
 --url 'http://127.0.0.1:8000/v1/chat/completions' \
 --rate 2 \
 --model 'qwen2.5' \
 --log-every-n-query 10 \
 --number 20 \
 --api openai \
 --temperature 0.9 \
 --max-tokens 1024 \
 --prompt @prompt.txt
```

### Complex Requests
Using `stop`, `stream`, `temperature`, etc.:

```bash
evalscope perf \
 --url 'http://127.0.0.1:8000/v1/chat/completions' \
 --rate 2 \
 --model 'qwen2.5' \
 --log-every-n-query 10 \
 --read-timeout 120 \
 --connect-timeout 120 \
 --number 20 \
 --max-prompt-length 128000 \
 --min-prompt-length 128 \
 --api openai \
 --temperature 0.7 \
 --max-tokens 1024 \
 --stop '<|im_end|>' \
 --dataset openqa \
 --stream
```

### Using `query-template`

You can set request parameters in the `query-template`:

```bash
evalscope perf \
 --url 'http://127.0.0.1:8000/v1/chat/completions' \
 --rate 2 \
 --model 'qwen2.5' \
 --log-every-n-query 10 \
 --read-timeout 120 \
 --connect-timeout 120 \
 --number 20 \
 --max-prompt-length 128000 \
 --min-prompt-length 128 \
 --api openai \
 --query-template '{"model": "%m", "messages": [{"role": "user","content": "%p"}], "stream": true, "skip_special_tokens": false, "stop": ["<|im_end|>"], "temperature": 0.7, "max_tokens": 1024}' \
 --dataset openqa 
```
Where `%m` and `%p` will be replaced by the model name and the prompt.

You can set request parameters in the query-template:

```{code-block} json
:caption: template.json

{
   "model":"%m",
   "messages":[
      {
         "role":"user",
         "content":"%p"
      }
   ],
   "stream":true,
   "skip_special_tokens":false,
   "stop":[
      "<|im_end|>"
   ],
   "temperature":0.7,
   "max_tokens":1024
}
```
```bash
evalscope perf \
 --url 'http://127.0.0.1:8000/v1/chat/completions' \
 --rate 2 \
 --model 'qwen2.5' \
 --log-every-n-query 10 \
 --read-timeout 120 \
 --connect-timeout 120 \
 --number 20 \
 --max-prompt-length 128000 \
 --min-prompt-length 128 \
 --api openai \
 --query-template @template.json \
 --dataset openqa 
```

Here’s the translation of the provided text into English:

### Using wandb to Record Test Results

Please install wandb:
```bash
pip install wandb
```

When starting, add the following parameters:
```bash
--wandb-api-key 'wandb_api_key'
--name 'name_of_wandb_log'
```  

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)

### Debugging Requests
Use the `--debug` option to output the requests and responses.

### Analyzing Results
During testing, this tool saves all data into an SQLite3 database, including requests and responses. You can analyze the test data after testing.

```python
import sqlite3
import base64
import pickle
import json

result_db_path = 'db_name.db'
con = sqlite3.connect(result_db_path)
query_sql = "SELECT request, response_messages, prompt_tokens, completion_tokens \
                FROM result WHERE success='1'"

with con:
    rows = con.execute(query_sql).fetchall()
    if len(rows) > 0:
        for row in rows:
            request = row[0]
            responses = row[1]
            request = base64.b64decode(request)
            request = pickle.loads(request)
            responses = base64.b64decode(responses)
            responses = pickle.loads(responses)
            response_content = ''
            for response in responses:
                response = json.loads(response)
                if not response['choices']:
                   continue
                response_content += response['choices'][0]['delta']['content']
            print('prompt: %s, tokens: %s, completion: %s, tokens: %s' %
                  (request['messages'][0]['content'], row[2], response_content,
                   row[3]))
```

## Speed Benchmark
To conduct speed tests and obtain the speed benchmarks reported by [Qwen Official](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html), please use `--dataset [speed_benchmark|speed_benchmark_long]`.

- `speed_benchmark`: Tests prompts of lengths [1, 6144, 14336, 30720] with a fixed output of 2048 tokens.
- `speed_benchmark_long`: Tests prompts of lengths [63488, 129024] with a fixed output of 2048 tokens.

```{note}
For speed testing, the `--url` option should use the `/v1/completions` endpoint instead of the `/v1/chat/completions` endpoint to avoid the additional processing of chat templates affecting the input length.
```

### Inference Based on Transformer
```bash
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --rate 1 \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --attn-implementation flash_attention_2 \
 --log-every-n-query 5 \
 --connect-timeout 6000 \
 --read-timeout 6000 \
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local \
 --dataset speed_benchmark \
 --debug
```

Example Output:
```
Speed Benchmark Results:
+---------------+-----------------+----------------+
| Prompt Tokens | Speed(tokens/s) | GPU Memory(GB) |
+---------------+-----------------+----------------+
|       1       |      50.69      |      0.97      |
|     6144      |      51.36      |      1.23      |
|     14336     |      49.93      |      1.59      |
|     30720     |      49.56      |      2.34      |
+---------------+-----------------+----------------+
```

### Inference Based on vLLM
```bash
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --rate 1 \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --log-every-n-query 5 \
 --connect-timeout 6000 \
 --read-timeout 6000 \
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local_vllm \
 --dataset speed_benchmark 
```
Example Output (Note that vllm reserves GPU memory, so GPU usage is not shown here):
```
Speed Benchmark Results:
+---------------+-----------------+----------------+
| Prompt Tokens | Speed(tokens/s) | GPU Memory(GB) |
+---------------+-----------------+----------------+
|       1       |     343.08      |      0.0       |
|     6144      |     334.71      |      0.0       |
|     14336     |     318.88      |      0.0       |
|     30720     |     292.86      |      0.0       |
+---------------+-----------------+----------------+
```

## Custom Request API
Currently supported API request formats are openai and dashscope. To extend the API, you can create a subclass of `ApiPluginBase` and use the `@register_api("api name")` decorator.

You can build the request using `model`, `prompt`, and `query_template` in the `build_request` method. You may refer to `opanai_api.py`.

The `parse_responses` method will return the number of `prompt_tokens` and `completion_tokens`.

```python
@register_api('custom')
class CustomPlugin(ApiPluginBase):
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        
    @abstractmethod
    def build_request(self, messages: List[Dict], param: QueryParameters) -> Dict:
        """Build a api request body.

        Args:
            messages (List[Dict]): The messages generated by the dataset.
            param (QueryParameters): The query parameters.

        Raises:
            NotImplementedError: Not implemented.

        Returns:
            Dict: The api request body.
        """
        raise NotImplementedError
    
    @abstractmethod
    def parse_responses(self, 
                        responses: List, 
                        request: Any=None,
                        **kwargs: Any) -> Tuple[int, int]:
        """Parse responses and return number of request and response tokens.

        Args:
            responses (List[bytes]): List of HTTP response bodies. For stream output,
                there are multiple responses, each is bytes, for general only one. 
            request (Any): The request body.

        Returns:
            Tuple: (Number of prompt_tokens and number of completion_tokens).
        """
        raise NotImplementedError  
```

## Custom Datasets
Currently supported dataset formats include:

- `line_by_line`: Each line is treated as a separate prompt, requiring a `dataset_path`.

- `longalpaca`: Uses `item['instruction']` as the prompt. If `dataset_path` is not specified, it will be automatically downloaded from modelscope.

- `openqa`: Uses `item['question']` as the prompt. If `dataset_path` is not specified, it will be automatically downloaded from modelscope.

### Extending Datasets
To extend datasets, you can create a subclass of `DatasetPluginBase` and use the `@register_dataset('dataset name')` decorator,

Then implement the `build_messages` method to return a prompt.

```python
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset

@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return a prompt.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            prompt = item.strip()
            if len(prompt) > self.query_parameters.min_prompt_length and len(
                    prompt) < self.query_parameters.max_prompt_length:
                yield [{'role': 'user', 'content': prompt}]
```