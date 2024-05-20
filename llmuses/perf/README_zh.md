## Quick Start
专注于大语言模型的压测工具，可定制化支持各种不通的数据集格式，以及不同的API协议格式。
### Usage

#### Command line  
```bash
llmuses perf --help
usage: llmuses <command> [<args>] perf [-h] --model MODEL [--url URL] [--connect-timeout CONNECT_TIMEOUT] [--read-timeout READ_TIMEOUT] [--max-prompt-length MAX_PROMPT_LENGTH] [--min-prompt-length MIN_PROMPT_LENGTH]
                                       [--prompt PROMPT] [-n NUMBER] [--parallel PARALLEL] [--rate RATE] [--log-every-n-query LOG_EVERY_N_QUERY] [--headers KEY1=VALUE1 [KEY1=VALUE1 ...]] [--wandb-api-key WANDB_API_KEY]
                                       [--name NAME] [--debug] [--tokenizer-path TOKENIZER_PATH] [--api API] [--query-template QUERY_TEMPLATE] [--dataset DATASET] [--dataset-path DATASET_PATH]

options:
  -h, --help            show this help message and exit
  --model MODEL         The test model name.
  --url URL
  --connect-timeout CONNECT_TIMEOUT
                        The network connection timeout
  --read-timeout READ_TIMEOUT
                        The network read timeout
  --max-prompt-length MAX_PROMPT_LENGTH
                        Maximum input prompt length
  --min-prompt-length MIN_PROMPT_LENGTH
                        Minimum input prompt length.
  --prompt PROMPT       Specified the request prompt, all the query will use this prompt, You can specify local file via @file_path, the prompt will be the file content.
  -n NUMBER, --number NUMBER
                        How many requests to be made, if None, will will send request base dataset or prompt.
  --parallel PARALLEL   Set number of concurrency request, default 1
  --rate RATE           Number of requests per second. default None, if it set to -1,then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times. Mutual exclusion
                        with parallel
  --log-every-n-query LOG_EVERY_N_QUERY
                        Logging every n query.
  --headers KEY1=VALUE1 [KEY1=VALUE1 ...]
                        Extra http headers accepts by key1=value1 key2=value2. The headers will be use for each query.You can use this parameter to specify http authorization and other header.
  --wandb-api-key WANDB_API_KEY
                        The wandb api key, if set the metric will be saved to wandb.
  --name NAME           The wandb db result name and result db name, default: {model_name}_{current_time}
  --debug               Debug request send.
  --tokenizer-path TOKENIZER_PATH
                        Specify the tokenizer weight path, used to calculate the number of input and output tokens,usually in the same directory as the model weight.
  --api API             Specify the service api, current support [openai|dashscope|zhipu]you can define your custom parser with python, and specify the python file path, reference api_plugin_base.py,
  --query-template QUERY_TEMPLATE
                        Specify the query template, should be a json string, or local file,with local file, specified with @local_file_path,will will replace model and prompt in the template.
  --dataset DATASET     Specify the dataset [openqa|longalpaca|line_by_line]you can define your custom dataset parser with python, and specify the python file path, reference dataset_plugin_base.py,
  --dataset-path DATASET_PATH
                        Path to the dataset file, Used in conjunction with dataset. If dataset is None, each line defaults to a prompt.
```

#### Start the client

```bash
##### 通过open qa数据集测试qwen在vllm和llmdeploy下的性能
#### 数据集地址: https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/blob/main/open_qa.jsonl
llmuses perf --url 'http://IP:PORT/v1/chat/completions' --parallel 1 --model 'qwen' --dataset 'datasets/open_qa.jsonl'  --log-every-n-query 1 --read-timeout=120  --parser 'openai_openqa_qwen' -n 10 --max-prompt-length 128000 --tokenizer-path /data/models/Qwen1.5-32B/
```
##### 输出
```bash
2024-04-28 11:21:36,161 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 5.003, concurrency: 1, completed: 1, succeed: 1, failed: 0, qps: 0.2, latency: 4.015, time to first token: 0.003, throughput(output tokens per second): 18.987, time per output token: 0.05267, package per request: 97.0, time per package: 0.042, input tokens per request: 16.0, output tokens per request: 95.0
2024-04-28 11:21:37,165 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 6.009, concurrency: 1, completed: 2, succeed: 2, failed: 0, qps: 0.333, latency: 2.594, time to first token: 0.003, throughput(output tokens per second): 20.302, time per output token: 0.04926, package per request: 63.0, time per package: 0.042, input tokens per request: 20.0, output tokens per request: 61.0
2024-04-28 11:21:39,171 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 8.015, concurrency: 1, completed: 3, succeed: 3, failed: 0, qps: 0.374, latency: 2.51, time to first token: 0.003, throughput(output tokens per second): 22.084, time per output token: 0.04528, package per request: 61.0, time per package: 0.042, input tokens per request: 20.667, output tokens per request: 59.0
2024-04-28 11:21:43,179 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 12.023, concurrency: 1, completed: 4, succeed: 4, failed: 0, qps: 0.333, latency: 2.803, time to first token: 0.002, throughput(output tokens per second): 21.958, time per output token: 0.04554, package per request: 68.0, time per package: 0.042, input tokens per request: 18.75, output tokens per request: 66.0
2024-04-28 11:21:46,187 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 15.031, concurrency: 1, completed: 5, succeed: 5, failed: 0, qps: 0.333, latency: 2.845, time to first token: 0.002, throughput(output tokens per second): 22.288, time per output token: 0.04487, package per request: 69.0, time per package: 0.042, input tokens per request: 19.4, output tokens per request: 67.0
2024-04-28 11:21:53,197 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 22.04, concurrency: 1, completed: 6, succeed: 6, failed: 0, qps: 0.272, latency: 3.584, time to first token: 0.002, throughput(output tokens per second): 23.049, time per output token: 0.04339, package per request: 86.667, time per package: 0.042, input tokens per request: 19.333, output tokens per request: 84.667
2024-04-28 11:22:01,207 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 30.051, concurrency: 1, completed: 7, succeed: 7, failed: 0, qps: 0.233, latency: 4.172, time to first token: 0.002, throughput(output tokens per second): 22.995, time per output token: 0.04349, package per request: 100.714, time per package: 0.042, input tokens per request: 21.429, output tokens per request: 98.714
2024-04-28 11:22:02,211 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 31.056, concurrency: 1, completed: 8, succeed: 8, failed: 0, qps: 0.258, latency: 3.834, time to first token: 0.002, throughput(output tokens per second): 23.345, time per output token: 0.04284, package per request: 92.625, time per package: 0.042, input tokens per request: 22.5, output tokens per request: 90.625
2024-04-28 11:22:06,218 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 35.062, concurrency: 1, completed: 9, succeed: 9, failed: 0, qps: 0.257, latency: 3.845, time to first token: 0.002, throughput(output tokens per second): 23.33, time per output token: 0.04286, package per request: 92.889, time per package: 0.042, input tokens per request: 21.444, output tokens per request: 90.889
2024-04-28 11:22:10,226 - perf - http_client.py - statistic_benchmark_metric_worker - 422 - INFO - Time: 39.07, concurrency: 1, completed: 10, succeed: 10, failed: 0, qps: 0.256, latency: 3.859, time to first token: 0.007, throughput(output tokens per second): 23.317, time per output token: 0.04289, package per request: 93.1, time per package: 0.042, input tokens per request: 21.8, output tokens per request: 91.1
Benchmarking summary: 
 Time taken for tests: 39.070 seconds
 Expected number of requests: 10
 Number of concurrency: 1
 Total requests: 10
 Succeed requests: 10
 Failed requests: 0
 Average QPS: 0.256
 Average latency: 3.859
 Throughput(average output tokens per second): 23.317
 Average time to first token: 0.007
 Average input tokens per request: 21.800
 Average output tokens per request: 91.100
 Average time per output token: 0.04289
 Average package per request: 93.100
 Average package latency: 0.042
 Percentile of time to first token: 
     p50: 0.0021
     p66: 0.0023
     p75: 0.0025
     p80: 0.0030
     p90: 0.0526
     p95: 0.0526
     p98: 0.0526
     p99: 0.0526
 Percentile of request latency: 
     p50: 3.9317
     p66: 3.9828
     p75: 4.0153
     p80: 7.2801
     p90: 7.7003
     p95: 7.7003
     p98: 7.7003
     p99: 7.7003
```

#### How to log metrics to wandb
--wandb-api-key 'your_wandb_api_key'  --name 'name_of_wandb_and_result_db'

#### How to debug
--debug 
with --debug option, we will output the request and response.
