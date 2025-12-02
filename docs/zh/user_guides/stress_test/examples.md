# 使用示例

## 使用本地模型推理

本项目支持本地transformers进行推理和vllm推理（需先安装vllm）， `--model`可以填入modelscope模型名称，例如`Qwen/Qwen2.5-0.5B-Instruct`；也可以直接指定模型权重路径，例如`/path/to/model_weights`，无需指定`--url`参数。

**1. 使用transformers进行推理**

指定`--api local`：
```bash
evalscope perf \
 --model 'Qwen/Qwen2.5-0.5B-Instruct' \
 --attn-implementation flash_attention_2 \  # 可不填，或选[flash_attention_2|eager|sdpa]
 --number 20 \
 --parallel 2 \
 --api local \
 --dataset openqa
```

**2. 使用vllm进行推理**

指定`--api local_vllm`：
```bash
evalscope perf \
 --model 'Qwen/Qwen2.5-0.5B-Instruct' \
 --number 20 \
 --parallel 2 \
 --api local_vllm \
 --dataset openqa
```

## 使用`prompt`
```bash
evalscope perf \
 --url 'http://127.0.0.1:8000/v1/chat/completions' \
 --parallel 2 \
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
 --parallel 2 \
 --model 'qwen2.5' \
 --log-every-n-query 10 \
 --number 20 \
 --api openai \
 --temperature 0.9 \
 --max-tokens 1024 \
 --prompt @prompt.txt
```

## 复杂请求
使用`stop`，`stream`，`temperature`等：

```bash
evalscope perf \
 --url 'http://127.0.0.1:8000/v1/chat/completions' \
 --parallel 2 \
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

## 使用`query-template`

您可以在`query-template`中设置请求参数：

```bash
evalscope perf \
 --url 'http://127.0.0.1:8000/v1/chat/completions' \
 --parallel 2 \
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
 --parallel 2 \
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

## 使用random数据集

根据`prefix-length`，`max-prompt-length`和`min-prompt-length`随机生成prompt，必需指定`tokenizer-path`。生成prompt的token数量在`prefix_length + min-prompt-length`和`prefix_length + max-prompt-length`之间均匀分布，在一次测试中所有请求prefix部分相同。

```{note}
由于chat_template以及tokenize算法的影响，生成的prompt的token数量可能有些误差，不是精确的指定token数量。
```

执行以下命令即可：

```bash
evalscope perf \
  --parallel 20 \
  --model Qwen2.5-0.5B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
  --api openai \
  --dataset random \
  --min-tokens 128 \
  --max-tokens 128 \
  --prefix-length 64 \
  --min-prompt-length 1024 \
  --max-prompt-length 2048 \
  --number 100 \
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
  --debug
```

## 使用random图文数据集
使用`random_vl`数据集，随机生成图像和文本输入，在`random`基础上增加了图像相关参数（`image-width`，`image-height`，`image-format`，`image-num`）。

```bash
evalscope perf \
  --parallel 20 \
  --model Qwen2.5-VL-3B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
  --api openai \
  --dataset random_vl \
  --min-tokens 128 \
  --max-tokens 128 \
  --prefix-length 0 \
  --min-prompt-length 100 \
  --max-prompt-length 100 \
  --image-width 512 \
  --image-height 512 \
  --image-format RGB \
  --image-num 1 \
  --number 100 \
  --tokenizer-path Qwen/Qwen2.5-VL-3B-Instruct \
  --debug
```

## 可视化测试结果

### 使用WandB
请使用如下命令安装wandb:
```bash
pip install wandb
```
启动测试前添加如下参数:
```bash
--visualizer wandb
--name 'name_of_wandb_log'
```

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)


### 使用SwanLab

请使用如下命令安装SwanLab:
```bash
pip install swanlab
```

启动测试前添加如下参数:
```bash
# 可使用 SWANLAB_PROJ_NAME 环境变量指定项目名称
--visualizer swanlab
--name 'name_of_swanlab_log'
```  

![swanlab sample](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/swanlab.png)


### 使用ClearML
请使用如下命令安装ClearML:
```bash
pip install clearml
```

初始化ClearML服务器:
```bash
clearml-init
```

启动测试前添加如下参数:
```bash
# 可使用 CLEARML_PROJECT_NAME 环境变量指定项目名称
--visualizer clearml
--name 'name_of_clearml_task'
```

![clearml sample](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/clearml_vis.jpg)


## 调试请求
使用 `--debug` 选项，我们将输出请求和响应，输出示例如下：

**非`stream`模式输出示例**

```text
2024-11-27 11:25:34,161 - evalscope - http_client.py - on_request_start - 116 - DEBUG - Starting request: <TraceRequestStartParams(method='POST', url=URL('http://127.0.0.1:8000/v1/completions'), headers=<CIMultiDict('Content-Type': 'application/json', 'user-agent': 'modelscope_bench', 'Authorization': 'Bearer EMPTY')>)>
2024-11-27 11:25:34,163 - evalscope - http_client.py - on_request_chunk_sent - 128 - DEBUG - Request sent: <method='POST',  url=URL('http://127.0.0.1:8000/v1/completions'), truncated_chunk='{"prompt": "hello", "model": "qwen2.5"}'>
2024-11-27 11:25:38,172 - evalscope - http_client.py - on_response_chunk_received - 140 - DEBUG - Request received: <method='POST',  url=URL('http://127.0.0.1:8000/v1/completions'), truncated_chunk='{"id":"cmpl-a4565eb4fc6b4a5697f38c0adaf9b70b","object":"text_completion","created":1732677934,"model":"qwen2.5","choices":[{"index":0,"text":"，everyone！今天我给您撒个谎哦。 ))\\n\\n今天开心的事。","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":1,"total_tokens":17,"completion_tokens":16}}'>
```

**`stream`模式输出示例**

```text
2024-11-27 20:02:24,760 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"重要的"},"finish_reason":null}],"usage":null}
2024-11-27 20:02:24,803 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}],"usage":null}
2024-11-27 20:02:24,847 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"，以便"},"finish_reason":null}],"usage":null}
2024-11-27 20:02:24,890 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"及时"},"finish_reason":null}],"usage":null}
2024-11-27 20:02:24,933 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"得到"},"finish_reason":null}],"usage":null}
2024-11-27 20:02:24,976 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"帮助"},"finish_reason":null}],"usage":null}
2024-11-27 20:02:25,023 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"和支持"},"finish_reason":null}],"usage":null}
2024-11-27 20:02:25,066 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}],"usage":null}
2024-11-27 20:02:25,109 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}],"usage":null}
2024-11-27 20:02:25,111 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"。<|im_end|>"},"finish_reason":null}],"usage":null}
2024-11-27 20:02:25,113 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: {"model":"Qwen2.5-0.5B-Instruct","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":260,"total_tokens":310}}
2024-11-27 20:02:25,113 - evalscope - http_client.py - _handle_stream - 57 - DEBUG - Response recevied: data: [DONE]
```
