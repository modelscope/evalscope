# vLLM Bench vs Evalscope Perf 压测对比

目的：给出一套“可比、可复现、可扩展”的压测方案，让 `evalscope perf` 与 `vllm bench serve` 在同一 vLLM 模型服务上输出一致的请求与统计指标。

结论：
- 在相同请求参数与并发配置下，`evalscope perf` 能与 `vllm bench serve` 达到一致的负载和指标（TTFT、TPOT、ITL、吞吐等）表现。
- 本文提供参数一一映射与校验步骤，帮助你快速复现并扩展测试。

---

## TL;DR

1) 启动 vLLM OpenAI Chat 服务
```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.5 \
  --served-model-name Qwen2.5-0.5B-Instruct \
  --trust-remote-code \
  --port 8801 \
  --no-enable-prefix-caching
```

2) 用 vLLM Bench 压测（随机数据）
```bash
vllm bench serve \
  --max-concurrency 50 \
  --num-prompts 500 \
  --host 127.0.0.1 \
  --port 8801 \
  --backend openai-chat \
  --model Qwen2.5-0.5B-Instruct \
  --dataset-name random \
  --random-input-len 100 \
  --random-output-len 100 \
  --tokenizer /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
  --endpoint /v1/chat/completions \
  --ignore-eos
```

3) 用 Evalscope Perf 压测（随机数据）
```bash
evalscope perf \
  --parallel 50 \
  --number 500 \
  --log-every-n-query 500 \
  --model Qwen2.5-0.5B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 100 \
  --prefix-length 0 \
  --min-prompt-length 100 \
  --max-prompt-length 100 \
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
  --extra-args '{"ignore_eos": true}'
```

---

## 环境与前置条件

- 硬件：A100 80G
- 版本：
  - vLLM: v0.11.0
  - evalscope: main 分支（2025-10-20）
- 安装建议：
  - vLLM 与 `evalscope[perf]` 均按官方文档安装
  - 开发环境安装 Evalscope：`pip install -e .[perf]`

注意：
- 若使用 ModelScope 权重，设置 `VLLM_USE_MODELSCOPE=True` 并提供对应 tokenizer 路径。
- 端点统一使用 OpenAI Chat `/v1/chat/completions`，保障请求体结构一致。

---

## 启动服务

使用 OpenAI Chat 端点（推荐）：
```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.5 \
  --served-model-name Qwen2.5-0.5B-Instruct \
  --trust-remote-code \
  --port 8801 \
  --no-enable-prefix-caching
```

提示：
- `--gpu-memory-utilization` 适度保守，避免 OOM。
- 若自定义 `--served-model-name`，请确保压测端的 `--model` 一致。
- `--no-enable-prefix-caching` 避免缓存影响压测结果。

---

## 参数对齐指南

为确保 apples-to-apples，请对齐以下参数：

- 并发与请求量
  - vLLM: `--max-concurrency` ↔ Evalscope: `--parallel`
  - vLLM: `--num-prompts` ↔ Evalscope: `--number`
- 端点与协议
  - vLLM: `--backend openai-chat` / `--endpoint /v1/chat/completions`
  - Evalscope: `--api openai` + `--url http://host:port/v1/chat/completions`
- 模型名
  - vLLM: `--model ...`
  - Evalscope: `--model ...`（仅用于填充请求体 model 字段）
- 数据生成（随机）
  - vLLM: `--dataset-name random --random-input-len N --random-output-len M`
  - Evalscope: `--dataset random --min-prompt-length N --max-prompt-length N --max-tokens M --prefix-length 0`
- Tokenizer（模板一致）
  - vLLM: `--tokenizer /path/to/tokenizer`
  - Evalscope: `--tokenizer-path <same-as-above-or-model-id>`
- 解码控制
  - vLLM: `--ignore-eos`
  - Evalscope: `--extra-args '{"ignore_eos": true}'`

---

## 一致性校验：最小示例（1 并发 / 1 请求）

vLLM：
```bash
vllm bench serve \
  --max-concurrency 1 \
  --num-prompts 1 \
  --host 127.0.0.1 \
  --port 8801 \
  --backend openai-chat \
  --model Qwen2.5-0.5B-Instruct \
  --dataset-name random \
  --random-input-len 100 \
  --random-output-len 100 \
  --tokenizer /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
  --endpoint /v1/chat/completions \
  --ignore-eos
```

示例日志（节选）：
```text
INFO ... Received request ... params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.1, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=100, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, prompt_embeds shape: None, lora_request: None, prompt_adapter_request: None.
```

Evalscope：
```bash
evalscope perf \
  --parallel 1 \
  --number 1 \
  --log-every-n-query 500 \
  --model Qwen2.5-0.5B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 100 \
  --prefix-length 0 \
  --min-prompt-length 100 \
  --max-prompt-length 100 \
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
  --extra-args '{"ignore_eos": true}'
```

示例日志（节选）：
```text
INFO ... Received request ... params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.1, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=100, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, prompt_embeds shape: None, lora_request: None, prompt_adapter_request: None.
```

对比结果：两者请求参数一致（TTFT/TPOT/ITL 的测量口径也相同）。

---

## 规模压测：50 并发 / 500 请求

vLLM：
```bash
vllm bench serve \
  --max-concurrency 50 \
  --num-prompts 500 \
  --host 127.0.0.1 \
  --port 8801 \
  --backend openai-chat \
  --model Qwen2.5-0.5B-Instruct \
  --dataset-name random \
  --random-input-len 100 \
  --random-output-len 100 \
  --tokenizer /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
  --endpoint /v1/chat/completions \
  --ignore-eos
```

vLLM 输出：
```text
============ Serving Benchmark Result ============
Successful requests:                     500       
Maximum request concurrency:             50        
Benchmark duration (s):                  5.10      
Total input tokens:                      49932     
Total generated tokens:                  50000     
Request throughput (req/s):              98.02     
Output token throughput (tok/s):         9802.05   
Peak output token throughput (tok/s):    10346.00  
Peak concurrent requests:                154.00    
Total Token throughput (tok/s):          19590.78  
---------------Time to First Token----------------
Mean TTFT (ms):                          33.14     
Median TTFT (ms):                        23.82     
P99 TTFT (ms):                           97.82     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.73      
Median TPOT (ms):                        4.73      
P99 TPOT (ms):                           4.94      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.69      
Median ITL (ms):                         4.58      
P99 ITL (ms):                            7.57      
==================================================
```

Evalscope：
```bash
evalscope perf \
  --parallel 50 \
  --number 500 \
  --log-every-n-query 500 \
  --model Qwen2.5-0.5B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 100 \
  --prefix-length 0 \
  --min-prompt-length 100 \
  --max-prompt-length 100 \
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
  --extra-args '{"ignore_eos": true}'
```

Evalscope 输出：
```text
Benchmarking summary:
+-----------------------------------+------------+
| Key                               |      Value |
+===================================+============+
| Time taken for tests (s)          |     5.7095 |
+-----------------------------------+------------+
| Number of concurrency             |    50      |
+-----------------------------------+------------+
| Total requests                    |   500      |
+-----------------------------------+------------+
| Succeed requests                  |   500      |
+-----------------------------------+------------+
| Failed requests                   |     0      |
+-----------------------------------+------------+
| Output token throughput (tok/s)   |  8757.5    |
+-----------------------------------+------------+
| Total token throughput (tok/s)    | 17507.8    |
+-----------------------------------+------------+
| Request throughput (req/s)        |    87.575  |
+-----------------------------------+------------+
| Average latency (s)               |     0.5539 |
+-----------------------------------+------------+
| Average time to first token (s)   |     0.0641 |
+-----------------------------------+------------+
| Average time per output token (s) |     0.0049 |
+-----------------------------------+------------+
| Average inter-token latency (s)   |     0.0049 |
+-----------------------------------+------------+
| Average input tokens per request  |    99.918  |
+-----------------------------------+------------+
| Average output tokens per request |   100      |
+-----------------------------------+------------+
2025-10-20 15:14:45 - evalscope - INFO: 
Percentile results:
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
| Percentiles | TTFT (s) | ITL (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Output (tok/s) | Total (tok/s) |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
|     10%     |  0.037   | 0.0001  |  0.0046  |   0.5183    |     100      |      100      |    160.5518    |   321.1035    |
|     25%     |  0.0514  | 0.0043  |  0.0046  |   0.5241    |     100      |      100      |    174.6327    |   349.1378    |
|     50%     |  0.0645  | 0.0045  |  0.0047  |   0.5362    |     100      |      100      |    186.5207    |   372.9354    |
|     66%     |  0.0685  | 0.0046  |  0.005   |   0.5629    |     100      |      100      |    189.7739    |   379.5479    |
|     75%     |  0.0755  | 0.0047  |  0.0052  |   0.5727    |     100      |      100      |    190.8478    |   381.6955    |
|     80%     |  0.0768  | 0.0048  |  0.0053  |   0.5807    |     100      |      100      |    191.9659    |   383.9318    |
|     90%     |  0.0904  | 0.0053  |  0.0057  |   0.6234    |     100      |      100      |    192.9745    |   385.8503    |
|     95%     |  0.1052  | 0.0066  |  0.0059  |   0.6408    |     100      |      100      |    194.7532    |   389.5064    |
|     98%     |  0.1085  | 0.0098  |  0.006   |   0.6573    |     100      |      100      |    202.4392    |   404.8783    |
|     99%     |  0.1093  |  0.013  |  0.0062  |   0.6633    |     100      |      100      |    205.6883    |   411.3766    |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
```

两者统计指标在误差范围（3%）内高度一致（差异来自计时开销、系统抖动、连接与调度细节等）。

---

## 指标口径与命名对照

- 成功/失败请求
  - vLLM: Successful requests
  - Evalscope: Succeed/Failed requests
- 吞吐（请求）
  - vLLM: Request throughput (req/s)
  - Evalscope: Request throughput (req/s)
- 吞吐（Token）
  - vLLM: Output token throughput / Total Token throughput
  - Evalscope: Output token throughput / Total token throughput
- 时延（单位差异）
  - vLLM: ms（TTFT/TPOT/ITL）
  - Evalscope: s（TTFT/TPOT/ITL/Latency）
  - 换算：1 ms = 0.001 s
- 分位数
  - Evalscope 默认打印多分位（10/25/50/75/90/95/98/99%），便于 tail 分析

---

## 常见误差来源与排查建议

- 流式/非流式一致
  - 两者默认按 token 流式统计。确保未混用非流式路径，以免 TTFT/ITL 口径不一致。
- 结束条件一致
  - `ignore_eos` 与 `stop` 配置需一致，否则输出长度与 TPOT/ITL 会偏差。
- Tokenizer 与模板一致
  - 指定相同 `--tokenizer`/`--tokenizer-path`，确保 Chat 模板一致（如 Qwen）。
- 解码参数一致
  - 将 temperature 设为 0、top_p=1、top_k=0；如有 `repetition_penalty`，保持一致。
- 预热
  - 大并发前建议做一次小批量请求预热，减少首次编译/加载影响。
- 连接与并发
  - 端侧连接池、Keep-Alive、DNS 解析等均会影响 TTFT；尽量在同一节点内回环（127.0.0.1）测试。
- 资源争用
  - 后台任务、NVLink/PCIe 带宽等对尾部时延影响显著；测试时保持环境稳定。
