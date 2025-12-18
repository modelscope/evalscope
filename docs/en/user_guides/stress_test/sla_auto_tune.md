# SLA Auto-Tuning

The SLA (Service Level Agreement) auto-tuning feature allows users to define service quality metrics (such as latency and throughput), and the tool will automatically adjust request pressure (concurrency or request rate) to find the maximum pressure value that the service can sustain while meeting these metrics.

## Features

- **Automatic Detection**: Uses binary search algorithm to automatically find the maximum concurrency (`parallel`) or request rate (`rate`) that satisfies SLA constraints.
- **Multi-Metric Support**: Supports end-to-end latency (Latency), time to first token (TTFT), time per output token (TPOT), as well as request throughput (RPS) and token throughput (TPS).
- **Flexible Constraints**: Supports setting upper limits (e.g., `p99_latency <= 2s`) or finding extremes (e.g., `tps: max`).
- **Stable Results**: Each test point runs multiple times by default and takes the average to reduce network fluctuation interference.

## Parameter Description

See [Parameter Description](./parameters.md#sla-settings) for details.

Main parameters:
- `--sla-auto-tune`: Enable auto-tuning.
- `--sla-variable`: Adjustment variable, `parallel` or `rate`.
- `--sla-params`: Define SLA rules.

## Supported Metrics and Operators

| Metric Category | Metric Name | Description | Supported Operators |
|----------|----------|------|------------|
| **Latency** | `avg_latency` | Average request latency | `<=`, `<`, `min` |
| | `p99_latency` | 99th percentile request latency | `<=`, `<`, `min` |
| | `avg_ttft` | Average time to first token | `<=`, `<`, `min` |
| | `p99_ttft` | 99th percentile time to first token | `<=`, `<`, `min` |
| | `avg_tpot` | Average time per output token | `<=`, `<`, `min` |
| | `p99_tpot` | 99th percentile time per output token | `<=`, `<`, `min` |
| **Throughput** | `rps` | Requests per second | `>=`, `>`, `max` |
| | `tps` | Tokens per second | `>=`, `>`, `max` |

## Workflow

1. **Baseline Test**: Start testing with the user-specified initial `parallel` or `rate` (recommended to set a small value, such as 1 or 2).
2. **Boundary Detection**:
   - If current metrics meet SLA, double the pressure until SLA is first violated or `--sla-max-concurrency` is reached.
   - If initial metrics violate SLA, halve the pressure to find a lower bound that satisfies conditions.
3. **Binary Search**: Perform binary search within the determined boundary window to precisely lock in the maximum pressure value that "just doesn't violate" SLA.
4. **Result Confirmation**: Each test point runs `--sla-num-runs` times (default 3), taking the average for judgment.
5. **Report Output**: After testing, output a summary of the tuning process and final results.

> **Note**: If the request success rate during testing is below 100%, that test point will be considered failed (violating SLA).

## Usage Examples

### 1. Find Maximum Concurrency Meeting P99 Latency <= 2s

```bash
evalscope perf \
 --model Qwen2.5-0.5B-Instruct \
 --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
 --url http://127.0.0.1:8801/v1/chat/completions \
 --api openai \
 --dataset random \
 --max-tokens 1024 \
 --prefix-length 0 \
 --min-prompt-length 1024 \
 --max-prompt-length 1024 \
 --sla-auto-tune \
 --sla-variable parallel \
 --sla-params '[{"p99_latency": "<=2"}]' \
 --parallel 2 \
 --sla-upper-bound 64
```

```text
                                    Detailed Performance Metrics                                    
┏━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃      ┃      ┃      Avg ┃      P99 ┃    Gen. ┃      Avg ┃     P99 ┃      Avg ┃     P99 ┃   Success┃
┃Conc. ┃  RPS ┃  Lat.(s) ┃  Lat.(s) ┃  toks/s ┃  TTFT(s) ┃ TTFT(s) ┃  TPOT(s) ┃ TPOT(s) ┃      Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│  1.0 │ 0.36 │    2.765 │    2.781 │  370.27 │    0.027 │   0.029 │    0.003 │   0.003 │    100.0%│
│  2.0 │ 0.67 │    2.964 │    2.991 │  689.13 │    0.033 │   0.038 │    0.003 │   0.003 │    100.0%│
└──────┴──────┴──────────┴──────────┴─────────┴──────────┴─────────┴──────────┴─────────┴──────────┘


               Best Performance Configuration               
 Highest RPS         Concurrency 2.0 (0.67 req/sec)         
 Lowest Latency      Concurrency 1.0 (2.765 seconds)        

Performance Recommendations:
• The system seems not to have reached its performance bottleneck, try higher concurrency
2025-12-18 15:11:02 - evalscope - INFO: Performance summary saved to: outputs/20251218_150933/Qwen2.5-0.5B-Instruct/performance_summary.txt
2025-12-18 15:11:02 - evalscope - INFO: SLA Auto-tune Summary:
+--------------------+------------+-----------------+---------------------+
| Criteria           | Variable   | Max Satisfied   | Note                |
+====================+============+=================+=====================+
| p99_latency <= 2.0 | parallel   | None            | Failed at min value |
+--------------------+------------+-----------------+---------------------+
```

### 2. Find Concurrency with Maximum TPS

```bash
evalscope perf \
 --model Qwen2.5-0.5B-Instruct \
 --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
 --url http://127.0.0.1:8801/v1/chat/completions \
 --api openai \
 --dataset random \
 --max-tokens 1024 \
 --prefix-length 0 \
 --min-prompt-length 1024 \
 --max-prompt-length 1024 \
 --sla-auto-tune \
 --sla-variable parallel \
 --sla-params '[{"tps": "max"}]' \
 --parallel 4
```

Example output:
```text
                                    Detailed Performance Metrics                                    
┏━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃      ┃      ┃      Avg ┃      P99 ┃    Gen. ┃      Avg ┃     P99 ┃      Avg ┃     P99 ┃   Success┃
┃Conc. ┃  RPS ┃  Lat.(s) ┃  Lat.(s) ┃  toks/s ┃  TTFT(s) ┃ TTFT(s) ┃  TPOT(s) ┃ TPOT(s) ┃      Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ 32.0 │ 5.68 │    5.590 │    6.250 │ 5813.67 │    0.194 │   0.294 │    0.005 │   0.006 │    100.0%│
│ 64.0 │ 5.76 │   11.040 │   11.875 │ 5902.57 │    0.220 │   0.580 │    0.011 │   0.011 │    100.0%│
│128.0 │ 6.96 │   18.212 │   19.624 │ 7124.25 │    0.429 │   1.379 │    0.017 │   0.019 │    100.0%│
│256.0 │ 7.81 │   32.062 │   37.111 │ 8000.89 │    1.567 │   4.752 │    0.030 │   0.032 │    100.0%│
│384.0 │ 7.87 │   43.487 │   65.909 │ 8057.14 │   12.210 │  32.809 │    0.031 │   0.033 │    100.0%│
│385.0 │ 7.58 │   43.956 │   66.302 │ 7766.79 │   12.487 │  32.461 │    0.031 │   0.033 │    100.0%│
│386.0 │ 7.69 │   43.658 │   66.308 │ 7869.97 │   12.541 │  32.859 │    0.030 │   0.033 │    100.0%│
│388.0 │ 7.60 │   44.322 │   66.909 │ 7784.20 │   12.873 │  33.074 │    0.031 │   0.033 │    100.0%│
│392.0 │ 7.57 │   45.006 │   67.572 │ 7748.62 │   13.501 │  33.293 │    0.031 │   0.034 │    100.0%│
│400.0 │ 7.76 │   44.831 │   66.181 │ 7945.06 │   14.184 │  33.387 │    0.030 │   0.033 │    100.0%│
│416.0 │ 7.56 │   46.748 │   67.288 │ 7738.68 │   16.175 │  33.624 │    0.030 │   0.033 │    100.0%│
│448.0 │ 7.61 │   50.028 │   68.255 │ 7790.59 │   19.657 │  35.208 │    0.030 │   0.032 │    100.0%│
│512.0 │ 7.76 │   57.843 │   70.814 │ 7941.28 │   25.967 │  37.136 │    0.031 │   0.033 │    100.0%│
└──────┴──────┴──────────┴──────────┴─────────┴──────────┴─────────┴──────────┴─────────┴──────────┘


               Best Performance Configuration               
 Highest RPS         Concurrency 384.0 (7.87 req/sec)       
 Lowest Latency      Concurrency 32.0 (5.590 seconds)       

Performance Recommendations:
• Optimal concurrency range is around 384.0
2025-12-18 15:06:49 - evalscope - INFO: Performance summary saved to: ./outputs/20251218_144530/Qwen2.5-0.5B-Instruct/performance_summary.txt
2025-12-18 15:06:49 - evalscope - INFO: SLA Auto-tune Summary:
+------------+------------+-----------------+---------------------+
| Criteria   | Variable   |   Max Satisfied | Note                |
+============+============+=================+=====================+
| tps -> max | parallel   |             384 | Best tps: 8057.1438 |
+------------+------------+-----------------+---------------------+
```

### 3. Find Maximum Request Rate Meeting TTFT < 0.05s and TTFT < 0.01s in Specific Range

```bash
evalscope perf \
 --model Qwen2.5-0.5B-Instruct \
 --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
 --url http://127.0.0.1:8801/v1/chat/completions \
 --api openai \
 --dataset random \
 --max-tokens 512 \
 --prefix-length 0 \
 --min-prompt-length 512 \
 --max-prompt-length 512 \
 --sla-auto-tune \
 --sla-variable rate \
 --sla-params '[{"p99_ttft": "<0.05"}, {"p99_ttft": "<0.01"}]' \
 --rate 2 \
 --sla-num-runs 1 \
 --sla-lower-bound 10 \
 --sla-upper-bound 40
```

Example output:
```text
                                    Detailed Performance Metrics                                    
┏━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃      ┃      ┃       ┃     Avg ┃     P99 ┃     Avg ┃    P99 ┃     Avg ┃    P99 ┃    Gen. ┃ Success┃
┃Conc. ┃ Rate ┃   RPS ┃ Lat.(s) ┃ Lat.(s) ┃ TTFT(s) ┃ TTFT(… ┃ TPOT(s) ┃ TPOT(… ┃  toks/s ┃    Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│   40 │   10 │  5.16 │   0.570 │   1.530 │   0.021 │  0.029 │   0.003 │  0.004 │  948.33 │  100.0%│
│   40 │   15 │  9.78 │   0.793 │   1.743 │   0.024 │  0.034 │   0.003 │  0.004 │ 2249.29 │  100.0%│
│   40 │   17 │  8.17 │   0.606 │   1.623 │   0.022 │  0.031 │   0.003 │  0.004 │ 1530.79 │  100.0%│
│   40 │   18 │ 10.30 │   0.799 │   1.712 │   0.023 │  0.042 │   0.003 │  0.004 │ 2466.09 │  100.0%│
│   40 │   19 │ 12.83 │   0.611 │   1.682 │   0.021 │  0.027 │   0.004 │  0.005 │ 2296.22 │  100.0%│
│   40 │   20 │ 11.81 │   0.744 │   1.861 │   0.023 │  0.054 │   0.004 │  0.006 │ 2435.94 │  100.0%│
└──────┴──────┴───────┴─────────┴─────────┴─────────┴────────┴─────────┴────────┴─────────┴────────┘


               Best Performance Configuration               
 Highest RPS         Concurrency 40 (20 req/sec)            
 Lowest Latency      Concurrency 40 (5.16 seconds)          

Performance Recommendations:
• The system seems not to have reached its performance bottleneck, try higher concurrency
• Success rate is low at high concurrency, check system resources or reduce concurrency
2025-12-18 16:19:48 - evalscope - INFO: Performance summary saved to: outputs/20251218_161909/Qwen2.5-0.5B-Instruct/performance_summary.txt
2025-12-18 16:19:48 - evalscope - INFO: SLA Auto-tune Summary:
+-----------------+------------+-----------------+----------------------------+
| Criteria        | Variable   | Max Satisfied   | Note                       |
+=================+============+=================+============================+
| p99_ttft < 0.05 | rate       | 19              | Satisfied                  |
+-----------------+------------+-----------------+----------------------------+
| p99_ttft < 0.01 | rate       | None            | Failed at lower bound (10) |
+-----------------+------------+-----------------+----------------------------+
```