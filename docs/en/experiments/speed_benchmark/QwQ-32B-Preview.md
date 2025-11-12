# QwQ-32B-Preview

> QwQ-32B-Preview is an experimental research model developed by the Qwen team, aimed at enhancing the reasoning capabilities of artificial intelligence. [Model Link](https://modelscope.cn/models/Qwen/QwQ-32B-Preview/summary)

The Speed Benchmark tool was used to test the GPU memory usage and inference speed of the QwQ-32B-Preview model under different configurations. The following tests measure the speed and memory usage when generating 2048 tokens, with input lengths of 1, 6144, 14336, and 30720:

## Local Transformers Inference Speed

### Test Environment

- NVIDIA A100 80GB * 1
- CUDA 12.1
- Pytorch 2.3.1
- Flash Attention 2.5.8
- Transformers 4.46.0
- EvalScope 0.7.0


### Stress Testing Command
```shell
pip install evalscope[perf] -U
```
```shell
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --parallel 1 \
 --model Qwen/QwQ-32B-Preview \
 --attn-implementation flash_attention_2 \
 --log-every-n-query 1 \
 --connect-timeout 60000 \
 --read-timeout 60000\
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local \
 --dataset speed_benchmark
```

### Test Results
```text
+---------------+-----------------+----------------+
| Prompt Tokens | Speed(tokens/s) | GPU Memory(GB) |
+---------------+-----------------+----------------+
|       1       |      17.92      |     61.58      |
|     6144      |      12.61      |     63.72      |
|     14336     |      9.01       |     67.31      |
|     30720     |      5.61       |     74.47      |
+---------------+-----------------+----------------+
```

## vLLM Inference Speed

### Test Environment
- NVIDIA A100 80GB * 2
- CUDA 12.1
- vLLM 0.6.3
- Pytorch 2.4.0
- Flash Attention 2.6.3
- Transformers 4.46.0

### Test Command
```shell
CUDA_VISIBLE_DEVICES=0,1 evalscope perf \
 --parallel 1 \
 --model Qwen/QwQ-32B-Preview \
 --log-every-n-query 1 \
 --connect-timeout 60000 \
 --read-timeout 60000\
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local_vllm \
 --dataset speed_benchmark
```

### Test Results
```text
+---------------+-----------------+
| Prompt Tokens | Speed(tokens/s) |
+---------------+-----------------+
|       1       |      38.17      |
|     6144      |      36.63      |
|     14336     |      35.01      |
|     30720     |      31.68      |
+---------------+-----------------+
```
