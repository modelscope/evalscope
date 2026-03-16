# vLLM Bench vs Evalscope Perf Load Testing Comparison

Goal: To present a reproducible, comparable, and extensible load testing methodology, enabling `evalscope perf` and `vllm bench serve` to output consistent request metrics and statistics for the same vLLM model service.

Conclusion:
- With identical request parameters and concurrency configurations, `evalscope perf` achieves consistent metrics (TTFT, TPOT, ITL, throughput, etc.) with `vllm bench serve`.
- This guide provides parameter mappings and validation steps to help you quickly reproduce and extend tests.

---

## TL;DR: Quick Comparison Recipe

1) Start vLLM OpenAI Chat Service
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

2) Load Test with vLLM Bench (Random Input)
```bash
vllm bench serve \
  --max-concurrency 50 \
  --num-prompts 1000 \
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

3) Load Test with Evalscope Perf (Random Input)
```bash
evalscope perf \
  --parallel 50 \
  --number 1000 \
  --log-every-n-query 1000 \
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

## Environment and Prerequisites

- Hardware: A100 80GB GPU
- Versions:
  - vLLM: v0.17.0
  - evalscope: v1.5.0

Notes:
- When using ModelScope weights, set `VLLM_USE_MODELSCOPE=True` and provide the corresponding tokenizer path.
- Ensure the endpoint uses OpenAI Chat `/v1/chat/completions` to maintain consistent request structures.

---

## Unified Server Configuration

Using the OpenAI Chat endpoint (recommended):
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

Tips:
- Set `--gpu-memory-utilization` conservatively to prevent OOM issues.
- If customizing `--served-model-name`, ensure the corresponding `--model` value matches on the benchmarking tools.
- Use `--no-enable-prefix-caching` to avoid caching effects on load test results.

---

## Parameter Alignment Guide (Key Mappings)

To maintain an apples-to-apples comparison, align the following parameters:

- **Concurrency and Requests**
  - vLLM: `--max-concurrency` ↔ Evalscope: `--parallel`
  - vLLM: `--num-prompts` ↔ Evalscope: `--number`
- **Endpoint and Protocol**
  - vLLM: `--backend openai-chat` / `--endpoint /v1/chat/completions`
  - Evalscope: `--api openai` + `--url http://host:port/v1/chat/completions`
- **Model Name**
  - vLLM: `--model ...`
  - Evalscope: `--model ...` (used for populating model field in request body)
- **Data Generation (Random)**
  - vLLM: `--dataset-name random --random-input-len N --random-output-len M`
  - Evalscope: `--dataset random --min-prompt-length N --max-prompt-length N --max-tokens M --prefix-length 0`
- **Tokenizer (Consistent Template)**
  - vLLM: `--tokenizer /path/to/tokenizer`
  - Evalscope: `--tokenizer-path <same-as-above-or-model-id>`
- **Decoding Control**
  - vLLM: `--ignore-eos`
  - Evalscope: `--extra-args '{"ignore_eos": true}'`

---

## Consistency Validation: Minimum Example (1 Concurrent / 1 Request)

vLLM:
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

Sample Logs:
```text
INFO ... Received request ... params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.1, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=100, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, prompt_embeds shape: None, lora_request: None, prompt_adapter_request: None.
```

Evalscope:
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

Sample Logs:
```text
INFO ... Received request ... params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.1, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=100, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, prompt_embeds shape: None, lora_request: None, prompt_adapter_request: None.
```

Comparison Results: Both tools produce consistent request parameters (metric methodologies for TTFT, TPOT, and ITL also align).

---

## Full Load Test: 50 Concurrency / 1000 Requests

**vLLM:**
```bash
vllm bench serve \
  --max-concurrency 50 \
  --num-prompts 1000 \
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

**Evalscope:**
```bash
evalscope perf \
  --parallel 50 \
  --number 1000 \
  --log-every-n-query 1000 \
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

vLLM Output:
```text
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             50        
Benchmark duration (s):                  9.25      
Total input tokens:                      100000    
Total generated tokens:                  100000    
Request throughput (req/s):              108.08    
Output token throughput (tok/s):         10808.22  
Peak output token throughput (tok/s):    11399.00  
Peak concurrent requests:                176.00    
Total token throughput (tok/s):          21616.43  
---------------Time to First Token----------------
Mean TTFT (ms):                          73.18     
Median TTFT (ms):                        74.81     
P99 TTFT (ms):                           144.48    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          3.85      
Median TPOT (ms):                        3.85      
P99 TPOT (ms):                           4.14      
---------------Inter-token Latency----------------
Mean ITL (ms):                           3.86      
Median ITL (ms):                         3.63      
P99 ITL (ms):                            12.26     
==================================================
```

Evalscope Output:
```text
Benchmarking summary:
+-----------------------------------+------------+
| Key                               |      Value |
+===================================+============+
| Time taken for tests (s)          |     9.4961 |
+-----------------------------------+------------+
| Number of concurrency             |    50      |
+-----------------------------------+------------+
| Request rate (req/s)              |    -1      |
+-----------------------------------+------------+
| Total requests                    |  1000      |
+-----------------------------------+------------+
| Succeed requests                  |  1000      |
+-----------------------------------+------------+
| Failed requests                   |     0      |
+-----------------------------------+------------+
| Output token throughput (tok/s)   | 10530.7    |
+-----------------------------------+------------+
| Total token throughput (tok/s)    | 21061.2    |
+-----------------------------------+------------+
| Request throughput (req/s)        |   105.307  |
+-----------------------------------+------------+
| Average latency (s)               |     0.4663 |
+-----------------------------------+------------+
| Average time to first token (s)   |     0.1131 |
+-----------------------------------+------------+
| Average time per output token (s) |     0.0036 |
+-----------------------------------+------------+
| Average inter-token latency (s)   |     0.0037 |
+-----------------------------------+------------+
| Average input tokens per request  |    99.999  |
+-----------------------------------+------------+
| Average output tokens per request |   100      |
+-----------------------------------+------------+
Percentile results:
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
| Percentiles | TTFT (s) | ITL (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Output (tok/s) | Total (tok/s) |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
|     10%     |  0.0532  |   0.0   |  0.003   |   0.3699    |     100      |      100      |    167.0738    |   334.1475    |
|     25%     |  0.0802  | 0.0025  |  0.003   |   0.3836    |     100      |      100      |    190.1434    |   380.2868    |
|     50%     |  0.0949  | 0.0029  |  0.0032  |   0.4225    |     100      |      100      |    236.7219    |   473.4438    |
|     66%     |  0.1054  | 0.0031  |  0.0039  |   0.4846    |     100      |      100      |    253.8581    |   507.7162    |
|     75%     |  0.1136  | 0.0033  |  0.004   |    0.526    |     100      |      100      |    260.8281    |   521.6561    |
|     80%     |  0.1398  | 0.0036  |  0.0041  |   0.5509    |     100      |      100      |    264.4042    |   528.8084    |
|     90%     |  0.163   | 0.0052  |  0.0043  |   0.5985    |     100      |      100      |    270.4783    |   540.9567    |
|     95%     |  0.4063  | 0.0067  |  0.005   |    0.738    |     100      |      100      |    276.9653    |   553.9306    |
|     98%     |  0.4287  | 0.0108  |  0.0055  |   0.8134    |     100      |      100      |    287.3055    |   574.6111    |
|     99%     |  0.4302  | 0.0141  |  0.0059  |   0.8161    |     100      |      100      |    293.9073    |   587.8146    |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
```


---

## Metric Definitions and Naming Correspondence

To better understand the outputs from both tools, here's a comparison of key metric definitions and the naming conventions used by `vLLM` and `Evalscope`:

- **Successful/Failed Requests**
  - **vLLM:** `Successful requests`
  - **Evalscope:** `Succeed requests` / `Failed requests`
  
- **Request Throughput**
  - **vLLM:** `Request throughput (req/s)`
  - **Evalscope:** `Request throughput (req/s)`

- **Token Throughput**
  - **vLLM:** `Output token throughput / Total Token throughput`
  - **Evalscope:** `Output token throughput / Total token throughput`

- **Latency Metrics (Unit Differences)**
  - **vLLM:** Reports in milliseconds (`ms`) for TTFT (Time to First Token), TPOT (Time Per Output Token), and ITL (Inter-Token Latency).
  - **Evalscope:** Reports in seconds (`s`) for TTFT, TPOT, and ITL, as well as for overall latency.
  - Conversion between the two: `1 ms = 0.001 s`

- **Percentiles**
  - **vLLM:** Reports percentile metrics such as `mean`, `median`, and `P99`.
  - **Evalscope:** Provides a more comprehensive percentile breakdown (e.g., `10%`, `25%`, `50%` (median), `75%`, `90%`, `95%`, `98%`, `99%`). This is helpful for analyzing tail latencies and variances.

---

## Common Sources of Discrepancies and Troubleshooting Suggestions

The following are common sources of discrepancies between `vLLM` and `Evalscope` performance results, along with tips for resolving them:

1. **Streaming vs. Non-Streaming**
   - Both tools use token streaming by default to measure token-level outputs.
   - Ensure that no non-streaming paths are mistakenly mixed in, as this could affect metrics like TTFT, ITL, and throughput.

2. **Consistent Termination Criteria**
   - Both tools must use identical settings for ending conditions. For example, ensure that `ignore_eos` and `stop_tokens` are consistently configured; otherwise, output lengths and metrics (e.g., TPOT/ITL) can diverge.

3. **Tokenizer and Prompt Template Consistency**
   - Use the same tokenizer and implement a consistent prompt formatting template (e.g., for models like `Qwen`).
   - Specify the correct `--tokenizer` or `--tokenizer-path` in both tools.

4. **Decoding Parameters**
   - Match decoding parameters such as `temperature=0`, `top_p=1`, `top_k=0`, and `repetition_penalty`.
   - Any inconsistency here can significantly affect both output length and latency metrics.

5. **Warm-Up**
   - Perform a small batch of warm-up queries before large-scale benchmarking to eliminate the impacts of initial loading or model compilation overheads.

6. **Connection and Concurrency Settings**
   - Ensure that both client-side connection pooling and server-side Keep-Alive settings are properly configured if running tests over a network.
   - Avoid DNS resolution delays or network jitter by running benchmarks locally (e.g., using `127.0.0.1` as the host).

7. **Resource Contention**
   - Background processes, NVIDIA NVLink bandwidth, and PCIe throughput may introduce non-deterministic delays.
   - For consistent measurements, perform benchmarks in a stable and isolated environment.

