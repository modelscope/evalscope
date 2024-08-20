# 模型推理性能压测
一个专注于大型语言模型的压力测试工具，可以自定义以支持各种数据集格式和不同的API协议格式。

## 使用方法

### 命令行
```bash
evalscope perf --help
usage: evalscope <command> [<args>] perf [-h] --model MODEL [--url URL] [--connect-timeout CONNECT_TIMEOUT] [--read-timeout READ_TIMEOUT] [-n NUMBER] [--parallel PARALLEL] [--rate RATE]
                                       [--log-every-n-query LOG_EVERY_N_QUERY] [--headers KEY1=VALUE1 [KEY1=VALUE1 ...]] [--wandb-api-key WANDB_API_KEY] [--name NAME] [--debug] [--tokenizer-path TOKENIZER_PATH]
                                       [--api API] [--max-prompt-length MAX_PROMPT_LENGTH] [--min-prompt-length MIN_PROMPT_LENGTH] [--prompt PROMPT] [--query-template QUERY_TEMPLATE] [--dataset DATASET]
                                       [--dataset-path DATASET_PATH] [--frequency-penalty FREQUENCY_PENALTY] [--logprobs] [--max-tokens MAX_TOKENS] [--n-choices N_CHOICES] [--seed SEED] [--stop STOP] [--stream]
                                       [--temperature TEMPERATURE] [--top-p TOP_P]
options:
  -h, --help            显示此帮助信息并退出
  --model MODEL         测试模型名称。
  --url URL
  --connect-timeout CONNECT_TIMEOUT
                        网络连接超时
  --read-timeout READ_TIMEOUT
                        网络读取超时
  -n NUMBER, --number NUMBER
                        发出的请求数量，如果为None，则将基于数据集或提示发送请求。
  --parallel PARALLEL   设置并发请求的数量，默认为1
  --rate RATE           每秒请求的数量，默认为None。如果设置为-1，则所有请求将在时间0发送。否则，我们使用泊松过程合成请求到达时间。与parallel互斥
  --log-every-n-query LOG_EVERY_N_QUERY
                        每n个查询记录日志。
  --headers KEY1=VALUE1 [KEY1=VALUE1 ...]
                        额外的HTTP头，格式为key1=value1 key2=value2。该头将用于每个查询。您可以使用此参数指定HTTP授权和其他头信息。
  --wandb-api-key WANDB_API_KEY
                        wandb API密钥，如果设置，则度量将保存到wandb。
  --name NAME           wandb数据库结果名称和结果数据库名称，默认为: {model_name}_{current_time}
  --debug               调试请求发送。
  --tokenizer-path TOKENIZER_PATH
                        指定分词器权重路径，用于计算输入和输出的token数量，通常与模型权重在同一目录下。
  --api API             指定服务API，目前支持[openai|dashscope]，您可以使用python定义自定义解析器，并指定python文件路径，参考api_plugin_base.py。
  --max-prompt-length MAX_PROMPT_LENGTH
                        最大输入prompt长度
  --min-prompt-length MIN_PROMPT_LENGTH
                        最小输入prompt长度。
  --prompt PROMPT       指定请求prompt，所有查询将使用此prompt。您可以通过@file_path指定本地文件，prompt将为文件内容。
  --query-template QUERY_TEMPLATE
                        指定查询模板，应该是一个JSON字符串或本地文件，使用本地文件时，通过@local_file_path指定，将在模板中替换模型和prompt。
  --dataset DATASET     指定数据集[openqa|longalpaca|line_by_line]，您可以使用python定义自定义数据集解析器，并指定python文件路径，参考dataset_plugin_base.py。
  --dataset-path DATASET_PATH
                        数据集文件的路径，与数据集结合使用。如果数据集为None，则每一行默认为一个prompt。
  --frequency-penalty FREQUENCY_PENALTY
                        frequency_penalty值。
  --logprobs            对数概率。
  --max-tokens MAX_TOKENS
                        可以生成的最大token数量。
  --n-choices N_CHOICES
                        生成的补全选择数量。
  --seed SEED           随机种子。
  --stop STOP           停止生成的tokens。
  --stop-token-ids      设置停止生成的token的ID。
  --stream              使用SSE流输出。
  --temperature TEMPERATURE
                        采样温度。
  --top-p TOP_P         top_p采样。
```
### 结果
```bash
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

#### 指标说明

| **指标**                                     | **说明**                       | **数值**        |
|------------------------------------------|-----------------------------|-----------------|
| Total requests                          | 总请求数                     | 10              |
| Succeeded requests                      | 成功请求                     | 10              |
| Failed requests                         | 失败请求                      | 0               |
| Average QPS                            | 每秒平均查询                 | 0.256           |
| Average latency                         | 所有请求的平均延迟          | 3.859           |
| Throughput                              | 每秒输出tokens              | 23.317          |
| Average time to first token            | 第一次token的平均时间       | 0.007           |
| Average input tokens per request        | 每个请求的平均输入tokens    | 21.800          |
| Average output tokens per request       | 每个请求的平均输出tokens    | 91.100          |
| Average time per output token           | 每个输出token的平均时间     | 0.04289         |
| Average package per request             | 每个请求的平均包数          | 93.100          |
| Average package latency                  | 每个包的平均延迟            | 0.042           |
| Percentile of time to first token (p50, ..., p99) | 第一次token的时间百分位     |   p50=0.0021, ..., p99=0.0526         |
| Percentile of request latency (p50, ..., p99)          | 请求延迟的百分位          |  p50=3.9317, ..., p99=7.7003         |


### 请求参数  
您可以在`query-template`中设置请求参数，也可以使用（`--stop`，`--stream`，`--temperature`等），参数将替换或添加到请求中。
#### 带参数的请求
示例请求 llama3 
```bash
evalscope perf --url 'http://127.0.0.1:8000/v1/chat/completions' --parallel 128 --model 'qwen' --log-every-n-query 10 --read-timeout=120 --dataset-path './datasets/open_qa.jsonl' -n 1 --max-prompt-length 128000 --api openai --stream --stop '<|im_end|>' --dataset openqa --debug
```

```bash
evalscope perf 'http://host:port/v1/chat/completions' --parallel 128 --model 'qwen' --log-every-n-query 10 --read-timeout=120 -n 10000 --max-prompt-length 128000 --tokenizer-path "THE_PATH_TO_TOKENIZER/Qwen1.5-32B/" --api openai --query-template '{"model": "%m", "messages": [{"role": "user","content": "%p"}], "stream": true,"skip_special_tokens": false,"stop": ["<|im_end|>"]}' --dataset openqa --dataset-path 'THE_PATH_TO_DATASETS/open_qa.jsonl'
```

#### query-template 使用说明
当你需要处理更复杂的请求时，可以使用模板来简化命令行。如果同时存在模板和参数，则参数中的值将优先使用。
query-template 示例：
```bash
evalscope perf --url 'http://127.0.0.1:8000/v1/chat/completions' --parallel 12 --model 'llama3' --log-every-n-query 10 --read-timeout=120 -n 1 --max-prompt-length 128000 --api openai --query-template '{"model": "%m", "messages": [], "stream": true, "stream_options":{"include_usage": true},"n": 3, "stop_token_ids": [128001, 128009]}' --dataset openqa --dataset-path './datasets/open_qa.jsonl'
```
对于 `messages` 字段, 数据集处理器的 message 将替换 query-template 中的 messages.

#### 启动客户端
```bash
# 测试 openai 服务
evalscope perf --url 'https://api.openai.com/v1/chat/completions' --parallel 1 --headers 'Authorization=Bearer YOUR_OPENAI_API_KEY' --model 'gpt-4o' --dataset-path 'THE_DATA_TO/open_qa.jsonl'  --log-every-n-query 10 --read-timeout=120  -n 100 --max-prompt-length 128000 --api openai --stream --dataset openqa
```

```bash
##### open qa dataset and 
#### dataset address: https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/blob/main/open_qa.jsonl
evalscope perf --url 'http://IP:PORT/v1/chat/completions' --parallel 1 --model 'qwen' --log-every-n-query 1 --read-timeout=120 -n 1000 --max-prompt-length 128000 --tokenizer-path "THE_PATH_TO_TOKENIZER/Qwen1.5-32B/" --api openai --query-template '{"model": "%m", "messages": [{"role": "user","content": "%p"}], "stream": true,"skip_special_tokens": false,"stop": ["<|im_end|>"]}' --dataset openqa --dataset-path 'THE_PATH_TO_DATASETS/open_qa.jsonl'
```

### FAQ

#### 如何使用wandb记录测试结果
--wandb-api-key 'your_wandb_api_key'  --name 'name_of_wandb_and_result_db'  

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)

#### 如何调试
使用 --debug 选项，我们将输出请求和响应。


#### 如何分析结果
该工具在测试期间会将所有数据保存到 sqlite3 数据库中，包括请求和响应。您可以在测试后分析测试数据。

```python
import sqlite3
import base64
import pickle
import json
result_db_path = 'db_name.db'
con = sqlite3.connect(result_db_path)
query_sql = "SELECT request, response_messages, prompt_tokens, completion_tokens \
                FROM result WHERE success='True'"
# how to save base64.b64encode(pickle.dumps(benchmark_data["request"])).decode("ascii"), 
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
                response_content += response['choices'][0]['delta']['content']
            print('prompt: %s, tokens: %s, completion: %s, tokens: %s' % (request['messages'][0]['content'], row[2], response_content, row[3]))
```

#### 支持的 API 有哪些
目前支持的 API 请求有 openai、dashscope 和 zhipu。您可以通过 --api 指定 API。

您可以使用 --query-template 自定义您的请求，您可以指定一个 JSON 字符串如下：
```json
{"model": "%m", "messages": [{"role": "user","content": "%p"}], "stream": true,"skip_special_tokens": false,"stop": ["<|im_end|>"]}
```

或使用本地文件 `@to_query_template_path`。我们将用 `model` 替换 `%m`，用 `prompt` 替换 `%p`。


#### 如何扩展 API
要扩展 API，您可以创建 `ApiPluginBase` 的子类，并使用 `@register_api("api 名称")` 进行注解。

通过 model、prompt 和 query_template 来构建请求的 build_request。您可以参考 opanai_api.py。

`parse_responses` 方法将返回 `prompt_tokens` 和 `completion_tokens` 的数量。

```python
class ApiPluginBase:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        
    @abstractmethod
    def build_request(self, messages: List[Dict], param: QueryParameters)->Dict:
        """Build a api request body.

        Args:
            messages (List[Dict]): The messages generated by dataset.
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
                        **kwargs:Any) -> Tuple[int, int]:
        """Parser responses and return number of request and response tokens.

        Args:
            responses (List[bytes]): List of http response body, for stream output,
                there are multiple responses, each is bytes, for general only one. 
            request (Any): The request body.

        Returns:
            Tuple: (Number of prompt_tokens and number of completion_tokens).
        """
        raise NotImplementedError  
```

#### 支持的数据集
目前支持逐行（line by line）、longalpaca 和 openqa 数据集。

- 逐行方法将每一行作为一个提示。

- longalpaca 将获取 item['instruction'] 作为提示。

- openqa 将获取 item['question'] 作为提示。

#### 如何扩展数据集
要扩展数据集，您可以创建 `DatasetPluginBase` 的子类，并使用 `@register_dataset('数据集名称')` 进行注解，

然后实现 `build_prompt` 方法以返回一个prompt。

```python
class DatasetPluginBase:
    def __init__(self, query_parameters: QueryParameters):
        """Build data set plugin

        Args:
            dataset_path (str, optional): The input dataset path. Defaults to None.
        """
        self.query_parameters = query_parameters

    def __next__(self):
        for item in self.build_messages():
            yield item
        raise StopIteration

    def __iter__(self):
        return self.build_messages()
    
    @abstractmethod
    def build_messages(self)->Iterator[List[Dict]]:
        """Build the request.

        Raises:
            NotImplementedError: The request is not impletion.

        Yields:
            Iterator[List[Dict]]: Yield request messages.
        """
        raise NotImplementedError
    
    def dataset_line_by_line(self, dataset: str)->Iterator[str]:
        """Get content line by line of dataset.

        Args:
            dataset (str): The dataset path.

        Yields:
            Iterator[str]: Each line of file.
        """
        with open(dataset, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
    
    def dataset_json_list(self, dataset: str)->Iterator[Dict]:
        """Read data from file which is list of requests.
           Sample: https://huggingface.co/datasets/Yukang/LongAlpaca-12k

        Args:
            dataset (str): The dataset path.

        Yields:
            Iterator[Dict]: The each request object.
        """
        with open(dataset, 'r', encoding='utf-8') as f:
            content = f.read()
        data = json.loads(content)
        for item in data:
            yield item      
```