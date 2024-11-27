# Examples

## Using Local Model Inference

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

## Using `prompt`
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

## Complex Requests
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

## Using `query-template`

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

## Using wandb to Record Test Results

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

## Debugging Requests
Use the `--debug` option to output the requests and responses.

```text
2024-11-27 11:25:34,161 - evalscope - http_client.py - on_request_start - 116 - DEBUG - Starting request: <TraceRequestStartParams(method='POST', url=URL('http://127.0.0.1:8000/v1/completions'), headers=<CIMultiDict('Content-Type': 'application/json', 'user-agent': 'modelscope_bench', 'Authorization': 'Bearer EMPTY')>)>
2024-11-27 11:25:34,163 - evalscope - http_client.py - on_request_chunk_sent - 128 - DEBUG - Request sent: <method='POST',  url=URL('http://127.0.0.1:8000/v1/completions'), truncated_chunk='{"prompt": "hello", "model": "qwen2.5"}'>
2024-11-27 11:25:38,172 - evalscope - http_client.py - on_response_chunk_received - 140 - DEBUG - Response received: <method='POST',  url=URL('http://127.0.0.1:8000/v1/completions'), truncated_chunk='{"id":"cmpl-a4565eb4fc6b4a5697f38c0adaf9b70b","object":"text_completion","created":1732677934,"model":"qwen2.5","choices":[{"index":0,"text":"Hello, everyone! Today I will tell you a lie. ))\\n\\nToday is a happy day.","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":1,"total_tokens":17,"completion_tokens":16}}'>
```