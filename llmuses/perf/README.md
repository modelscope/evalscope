## Quick Start

### Installation

```bash
cd llmuses/perf
pip install -r requirements.txt
```

### Usage

#### Start the server
```bash
# Pull the docker image
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.11.0-benchmark

# Start the server
docker run --rm --name perf_bench --gpus='"device=0"' --shm-size=384gb -e MODELSCOPE_CACHE=/data/modelscope_cache -v /root/perf/data:/data -v /root/perf/code:/code --net host registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.11.0-benchmark llmuses server --server-command 'python -m vllm.entrypoints.openai.api_server --host "0.0.0.0" --port 8000 --trust-remote-code --model=qwen/Qwen-7B-Chat --revision=v1.1.9 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 2048' --host '0.0.0.0' --logdir '/code/perf'

# Args
--envs: The environment variables for the server
    - MODELSCOPE_CACHE: the cache dir for modelscope
--server-command: the command to start the server
    - model: the model name on modelscope
    - revision: the model revision on modelscope
    - tensor-parallel-size: the tensor parallel size, align with the number of gpus
```

#### Start the client
```bash
# Local mode
python http_client.py --url 'http://0.0.0.0:8000/v1/chat/completions' --parallel 20 --rate 30 --model 'qwen/Qwen-7B-Chat' --prompt "hello" --parameters top_p=0.8 temperature=0.8 max_tokens=256 -n 1000 --log-every-n-query 10 --read-timeout=10 --format 'vllm_qwen_openai_completion'
# APIs 
python http_client.py --url 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation' --parallel 1 --rate 1 --headers 'Authorization=your_api_key' 'X-DashScope-SSE=enable' --model 'qwen/Qwen-7B-Chat'  --prompt "hello" --parameters top_p=0.8 temperature=0.8 max_tokens=256 --format dashscope_message -n 1 --log-every-n-query 1 

```

#### log metrics to wandb
--wandb-api-key 'your_wandb_api_key'  --wandb-name 'openonal_benchmark_name_default_{model_name}_{current_time}'

#### how to debug
--debug 
with --debug option, we will output the request and response.
