# 模型推理性能压测
一个大语言模型的压力测试工具，可以自定义以支持各种数据集格式和不同的API协议格式，默认支持OpenAI API格式。

## 快速使用
可以使用以下两种方式启动模型推理性能压测工具：

::::{tab-set}
:::{tab-item} 命令行启动
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

:::{tab-item} Python脚本启动
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
参数说明：

- `url`: 请求的URL地址
- `parallel`: 并行处理的任务数量
- `model`: 使用的模型名称
- `number`: 请求数量
- `api`: 使用的API服务
- `dataset`: 数据集名称
- `stream`: 是否启用流式处理


### 输出结果
```text
Benchmarking summary: 
+----------------------------------------------+------------------------------------------------+
| key                                          | Value                                          |
+==============================================+================================================+
| Time taken for tests (senconds)              | 7.539                                          |
+----------------------------------------------+------------------------------------------------+
| Number of concurrency                        | 1                                              |
+----------------------------------------------+------------------------------------------------+
| Total requests                               | 15                                             |
+----------------------------------------------+------------------------------------------------+
| Succeed requests                             | 15                                             |
+----------------------------------------------+------------------------------------------------+
| Failed requests                              | 0                                              |
+----------------------------------------------+------------------------------------------------+
| Average QPS                                  | 1.99                                           |
+----------------------------------------------+------------------------------------------------+
| Average latency                              | 0.492                                          |
+----------------------------------------------+------------------------------------------------+
| Average time to first token                  | 0.026                                          |
+----------------------------------------------+------------------------------------------------+
| Throughput(average output tokens per second) | 334.006                                        |
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

### 指标说明

| **指标**                                     | **说明**                       | **数值**        |
|------------------------------------------|-----------------------------|-----------------|
| Total requests                          | 总请求数                     | 15              |
| Succeeded requests                      | 成功请求数                     | 15              |
| Failed requests                         | 失败请求数                      | 0               |
| Average QPS                            | 每秒平均请求数                 | 1.99          |
| Average latency                         | 所有请求的平均延迟          | 0.492          |
| Throughput(average output tokens per second)  | 每秒输出token数量              | 334.006          |
| Average time to first token            | 首token的平均延时       |    0.026         |
| Average input tokens per request        | 每个请求的平均输入token数量    | 40.133         |
| Average output tokens per request       | 每个请求的平均输出token数量    | 167.867            |
| Average time per output token           | 输出每个token的平均时间     | 0.00299         |
| Average package per request             | 每个请求的平均包数          | 167.867           |
| Average package latency                  | 每个包的平均延迟            | 0.003            |
| Percentile of time to first token (p10, ..., p99) | 首token延时百分位     |            |
| Percentile of request latency (p10, ..., p99)          | 请求延迟的百分位          |          |


## 全部参数说明
执行 `evalscope perf --help` 可获取全部参数说明：


### 基本设置
- `--model` 测试模型名称。
- `--url` 指定API地址。
- `--name` wandb数据库结果名称和结果数据库名称，默认为: `{model_name}_{current_time}`，可选。
- `--api` 指定服务API，目前支持[openai|dashscope]，您可以使用python定义自定义解析器，并指定python文件路径，参考`api_plugin_base.py`。
- `--api-key` API密钥，可选。
- `--debug` 输出调试信息。

### 网络配置
- `--connect-timeout` 网络连接超时，默认为120s。
- `--read-timeout` 网络读取超时，默认为120s。
- `--headers` 额外的HTTP头，格式为`key1=value1 key2=value2`。该头将用于每个查询。

### 请求控制
- `--number` 发出的请求数量，默认为None，表示基于数据集数量发送请求。
- `--parallel` 设置并发请求的数量，默认为1。
- `--rate` 每秒请求的数量，默认为None。如果设置为-1，则所有请求将在时间0发送。否则，我们使用泊松过程合成请求到达时间。与parallel互斥。
- `--log-every-n-query` 每n个查询记录日志。

### Prompt设置
- `--max-prompt-length` 最大输入prompt长度，默认为`sys.maxsize`。
- `--min-prompt-length` 最小输入prompt长度，默认为0。
- `--prompt` 指定请求prompt，一个字符串或本地文件，使用优先级高于`dataset`。使用本地文件时，通过`@{path/to/file}`指定本地文件。
- `--query-template` 指定查询模板，一个`JSON`字符串或本地文件，使用本地文件时，通过`{path/to/file}`指定，将在模板中替换模型和prompt。默认为`[{'role': 'user', 'content': prompt}]`

### 数据集配置
- `--dataset` 指定数据集[openqa|longalpaca|line_by_line]，您可以使用python定义自定义数据集解析器，并指定python文件路径，参考`dataset_plugin_base.py`。
- `--dataset-path` 数据集文件的路径，与数据集结合使用。openqa与longalpaca可不指定数据集，将自动下载；line_by_line必须指定本地数据集文件，将一行一行加载。

### 模型设置
- `--tokenizer-path` 指定分词器权重路径，用于计算输入和输出的token数量，通常与模型权重在同一目录下。
- `--frequency-penalty` frequency_penalty值。
- `--logprobs` 对数概率。
- `--max-tokens` 可以生成的最大token数量。
- `--n-choices` 生成的补全选择数量。
- `--seed` 随机种子。
- `--stop` 停止生成的tokens。
- `--stop-token-ids` 设置停止生成的token的ID。
- `--stream` 使用SSE流输出。
- `--temperature` 采样温度。
- `--top-p` top_p采样。

### 数据存储
- `--wandb-api-key` wandb API密钥，如果设置，则度量将保存到wandb。


## 示例

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