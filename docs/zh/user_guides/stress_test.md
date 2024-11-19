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
- `parallel`: 并行请求的任务数量
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
- `--api` 指定服务API，目前支持[openai|dashscope|local|local_vllm]。
  - 指定为`openai`，则使用支持OpenAI的API，需要提供`--url`参数。
  - 指定为`dashscope`，则使用支持DashScope的API，需要提供`--url`参数。
  - 指定为`local`，则使用本地文件作为模型，并使用transformers进行推理。`--model`为模型文件路径，也可为model_id，将自动从modelscope下载模型，例如`Qwen/Qwen2.5-0.5B-Instruct`。
  - 指定为`local_vllm`，则使用本地文件作为模型，并启动vllm推理服务。`--model`为模型文件路径，也可为model_id，将自动从modelscope下载模型，例如`Qwen/Qwen2.5-0.5B-Instruct`。
- `--attn-implementation` Attention实现方式，默认为None，可选[flash_attention_2|eager|sdpa]，仅在`api`为`local`时有效。
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
- `--log-every-n-query` 每n个查询记录日志，默认为10。
- `--stream` 使用SSE流输出，默认为False。

### Prompt设置
- `--max-prompt-length` 最大输入prompt长度，默认为`sys.maxsize`，大于该值时，将丢弃prompt。
- `--min-prompt-length` 最小输入prompt长度，默认为0，小于该值时，将丢弃prompt。
- `--prompt` 指定请求prompt，一个字符串或本地文件，使用优先级高于`dataset`。使用本地文件时，通过`@/path/to/file`指定文件路径，例如`@./prompt.txt`。
- `--query-template` 指定查询模板，一个`JSON`字符串或本地文件，使用本地文件时，通过`@/path/to/file`指定文件路径，例如`@./query_template.json`。

### 数据集配置
- `--dataset` 指定数据集[openqa|longalpaca|line_by_line]，您可以使用python定义自定义数据集解析器，并指定python文件路径，参考`dataset_plugin_base.py`。
- `--dataset-path` 数据集文件的路径，与数据集结合使用。openqa与longalpaca可不指定数据集路径，将自动下载；line_by_line必须指定本地数据集文件，将一行一行加载。

### 模型设置
- `--tokenizer-path` 可选，指定分词器权重路径，用于计算输入和输出的token数量，通常与模型权重在同一目录下。
- `--frequency-penalty` frequency_penalty值。
- `--logprobs` 对数概率。
- `--max-tokens` 可以生成的最大token数量。
- `--min-tokens` 生成的最少token数量。
- `--n-choices` 生成的补全选择数量。
- `--seed` 随机种子，默认为42。
- `--stop` 停止生成的tokens。
- `--stop-token-ids` 设置停止生成的token的ID。
- `--temperature` 采样温度。
- `--top-p` top_p采样。

### 数据存储
- `--wandb-api-key` wandb API密钥，如果设置，则度量将保存到wandb。


## 示例

### 使用本地模型推理

本项目支持本地transformers进行推理和vllm推理（需先安装vllm）， `--model`可以填入modelscope模型名称，例如`Qwen/Qwen2.5-0.5B-Instruct`；也可以直接指定模型权重路径，例如`/path/to/model_weights`，无需指定`--url`参数。

**使用transformers进行推理**

```bash
evalscope perf \
 --model 'Qwen/Qwen2.5-0.5B-Instruct' \
 --attn-implementation flash_attention_2 \  # 可不填，或选[flash_attention_2|eager|sdpa]
 --number 20 \
 --rate 2 \
 --api local \
 --dataset openqa
```

**使用vllm进行推理**
```bash
evalscope perf \
 --model 'Qwen/Qwen2.5-0.5B-Instruct' \
 --number 20 \
 --rate 2 \
 --api local_vllm \
 --dataset openqa
```

### 使用`prompt`
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
 --prompt '写一个科幻小说，请开始你的表演'
```
也可以使用本地文件作为prompt：
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

### 复杂请求
使用`stop`，`stream`，`temperature`等：

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

### 使用`query-template`

您可以在`query-template`中设置请求参数：

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
其中`%m`和`%p`会被替换为模型名称和prompt。

您也可以使用本地`query-template.json`文件：

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


### 使用wandb记录测试结果

请安装wandb：
```bash
pip install wandb
```

在启动时，添加以下参数，即可：
```bash
--wandb-api-key 'wandb_api_key'
--name 'name_of_wandb_log'
```  

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)


### 调试请求
使用 `--debug` 选项，我们将输出请求和响应。


### 分析结果
该工具在测试期间会将所有数据保存到 sqlite3 数据库中，包括请求和响应。您可以在测试后分析测试数据。

```python
import sqlite3
import base64
import pickle
import json
result_db_path = 'db_name.db'
con = sqlite3.connect(result_db_path)
query_sql = "SELECT request, response_messages, prompt_tokens, completion_tokens \
                FROM result WHERE success='1'"
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
                if not response['choices']:
                   continue
                response_content += response['choices'][0]['delta']['content']
            print('prompt: %s, tokens: %s, completion: %s, tokens: %s' %
                  (request['messages'][0]['content'], row[2], response_content,
                   row[3]))
```

## Speed Benchmark
若想进行速度测试，得到[Qwen官方](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)报告的速度基准，请使用 `--dataset speed_benchmark`，包括：
- `speed_benchmark`: 测试[96, 2048, 6144, 14336, 30720]长度的prompt，固定输出2048个token。
- `speed_benchmark_long`: 测试[63488, 129024]长度的prompt，固定输出2048个token。

### 基于Transformer推理
```bash
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --rate 1 \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --attn-implementation flash_attention_2 \
 --log-every-n-query 5 \
 --connect-timeout 6000 \
 --read-timeout 6000\
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local \
 --dataset speed_benchmark \
 --debug
```

输出示例：
```
Speed Benchmark Results:
+---------------+-----------------+----------------+
| Prompt Tokens | Speed(tokens/s) | GPU Memory(GB) |
+---------------+-----------------+----------------+
|      95       |      49.37      |      0.97      |
|     2048      |      51.19      |      1.03      |
|     6144      |      51.41      |      1.23      |
|     14336     |      50.99      |      1.59      |
|     30720     |      50.06      |      2.34      |
+---------------+-----------------+----------------+
```

### 基于vLLM推理
```bash
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --rate 1 \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --log-every-n-query 5 \
 --connect-timeout 6000 \
 --read-timeout 6000\
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local_vllm \
 --dataset speed_benchmark 
```
输出示例（注意vllm预留显存，此处不显示GPU使用情况）：
```
Speed Benchmark Results:
+---------------+-----------------+----------------+
| Prompt Tokens | Speed(tokens/s) | GPU Memory(GB) |
+---------------+-----------------+----------------+
|      95       |     340.43      |      0.0       |
|     2048      |     338.91      |      0.0       |
|     6144      |     333.12      |      0.0       |
|     14336     |     318.41      |      0.0       |
|     30720     |     291.39      |      0.0       |
+---------------+-----------------+----------------+
```


## 自定义请求 API
目前支持的 API 请求格式有 openai、dashscope。要扩展 API，您可以创建 `ApiPluginBase` 的子类，并使用 `@register_api("api 名称")` 进行注解。

通过 model、prompt 和 query_template 来构建请求的 build_request。您可以参考 opanai_api.py。

`parse_responses` 方法将返回 `prompt_tokens` 和 `completion_tokens` 的数量。

```python
@register_api('custom')
class CustomPlugin(ApiPluginBase):
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

## 自定义数据集
目前支持如下三种格式数据集：

- `line_by_line`逐行将每一行作为一个提示，需提供`dataset_path`。

- `longalpaca` 将获取 item['instruction'] 作为提示，不指定`dataset_path`将从modelscope自动下载。

- `openqa` 将获取 item['question'] 作为提示，不指定`dataset_path`将从modelscope自动下载。

### 扩展数据集
要扩展数据集，您可以创建 `DatasetPluginBase` 的子类，并使用 `@register_dataset('数据集名称')` 进行注解，

然后实现 `build_messages` 方法以返回一个prompt。

```python
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
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