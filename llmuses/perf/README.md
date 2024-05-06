## Quick Start
A stress testing tool that focuses on large language models and can be customized to support various data set formats and different API protocol formats.
### Usage

#### Command line  
```bash
llmuses perf --help
usage: llmuses <command> [<args>] perf [-h] --model MODEL [--url URL] [--dataset DATASET] [--connect-timeout CONNECT_TIMEOUT] [--read-timeout READ_TIMEOUT] [--max-prompt-length MAX_PROMPT_LENGTH] [--min-prompt-length MIN_PROMPT_LENGTH]
                                       [--prompt PROMPT] [-n NUMBER] [--parallel PARALLEL] [--rate RATE] [--log-every-n-query LOG_EVERY_N_QUERY] [--parameters KEY1=VALUE1 [KEY1=VALUE1 ...]] [--headers KEY1=VALUE1 [KEY1=VALUE1 ...]]
                                       [--parser PARSER] [--wandb-api-key WANDB_API_KEY] [--name NAME] [--debug] [--tokenizer-path TOKENIZER_PATH]

options:
  -h, --help            show this help message and exit
  --model MODEL         The test model name.
  --url URL
  --dataset DATASET     Path to the dataset, with prompt line by line
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
  --rate RATE           Number of requests per second. default None, if it set to -1,then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times. Mutual exclusion with parallel
  --log-every-n-query LOG_EVERY_N_QUERY
                        Logging every n query.
  --parameters KEY1=VALUE1 [KEY1=VALUE1 ...]
                        Extra parameters accepts by key1=value1 key2=value2. The parameters will be use for each query.You can use this parameter to specify sample parameters such as top_p, top_k
  --headers KEY1=VALUE1 [KEY1=VALUE1 ...]
                        Extra http headers accepts by key1=value1 key2=value2. The headers will be use for each query.You can use this parameter to specify http auchorization and other header.
  --parser PARSER       Specify the request/response processor, the prompt generator, current support,[openai_prompt|openai_openqa_qwen|openai_openqa_llama3|openai_longalpaca_12k_qwen|openai_longalpaca_12k_llama3],you can define your custom
                        parser python file path, reference llm_parser_base.py,
  --wandb-api-key WANDB_API_KEY
                        The wandb api key, if set the metric will be saved to wandb.
  --name NAME           The wandb db result name and result db name, default: {model_name}_{current_time}
  --debug               Debug request send.
  --tokenizer-path TOKENIZER_PATH
                        Specify the tokenizer weight path used to calculate the number of input and output tokens,usually in the same directory as the model weight.
```

#### Start the client

```bash
##### open qa dataset and 
#### dataset address: https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/blob/main/open_qa.jsonl
llmuses perf --url 'http://IP:PORT/v1/chat/completions' --parallel 1 --model 'qwen' --dataset 'THE_PAT/open_qa.jsonl'  --log-every-n-query 1 --read-timeout=120  --parser 'openai_openqa_qwen' -n 10 --max-prompt-length 128000 --tokenizer-path /data/models/Qwen1.5-32B/
```

#### How to log metrics to wandb
--wandb-api-key 'your_wandb_api_key'  --name 'name_of_wandb_and_result_db'  

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)

#### How to debug
--debug 
with --debug option, we will output the request and response.
