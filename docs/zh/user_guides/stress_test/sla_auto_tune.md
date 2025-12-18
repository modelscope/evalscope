# SLA 自动调优

SLA (Service Level Agreement) 自动调优功能允许用户定义服务质量指标（如延迟、吞吐量），工具将自动调整请求压力（并发数或请求速率），寻找服务能够满足这些指标的最大压力值。

## 功能特性

- **自动探测**：通过二分查找算法，自动寻找满足 SLA 约束的最大并发数 (`parallel`) 或请求速率 (`rate`)。
- **多指标支持**：支持端到端延迟（Latency）、首字延迟（TTFT）、单字输出延迟（TPOT）以及请求吞吐量（RPS）、token吞吐量（TPS）等指标。
- **灵活约束**：支持设置上限（如 `p99_latency <= 2s`）或寻找极值（如 `tps: max`）。
- **结果稳定**：每个测试点默认运行多次取平均值，减少网络波动干扰。

## 参数说明

详见 [参数说明](./parameters.md#sla设置)。

主要参数：
- `--sla-auto-tune`: 开启自动调优。
- `--sla-variable`: 调整变量，`parallel` 或 `rate`。
- `--sla-params`: 定义 SLA 规则。

## 支持的指标与操作符

| 指标类别 | 指标名称 | 说明 | 支持操作符 |
|----------|----------|------|------------|
| **延迟类** | `avg_latency` | 平均请求延迟 | `<=`, `<`, `min` |
| | `p99_latency` | 99% 分位请求延迟 | `<=`, `<`, `min` |
| | `avg_ttft` | 平均首字延迟 (Time To First Token) | `<=`, `<`, `min` |
| | `p99_ttft` | 99% 分位首字延迟 | `<=`, `<`, `min` |
| | `avg_tpot` | 平均单字生成延迟 (Time Per Output Token) | `<=`, `<`, `min` |
| | `p99_tpot` | 99% 分位单字生成延迟 | `<=`, `<`, `min` |
| **吞吐类** | `rps` | 请求吞吐量 (Requests Per Second) | `>=`, `>`, `max` |
| | `tps` | Token 吞吐量 (Tokens Per Second) | `>=`, `>`, `max` |

## 工作流程

1. **基准测试**：以用户指定的初始 `parallel` 或 `rate` 开始测试（建议设置为较小值，如 1 或 2）。
2. **边界探测**：
   - 如果当前指标满足 SLA，将压力翻倍，直到首次违反 SLA 或达到 `--sla-upper-bound`。
   - 如果初始指标即违反 SLA，将压力减半，寻找满足条件的下界。
3. **二分查找**：在确定的边界窗口内进行二分查找，精确锁定“刚好不违约”的最大压力值。
4. **结果确认**：每个测试点会运行 `--sla-num-runs` 次（默认 3 次），取平均值进行判断。
5. **报告输出**：测试结束后，输出调优过程摘要及最终结果。

> **注意**：如果测试过程中请求成功率（Success Rate）低于 100%，该测试点将被视为失败（违反 SLA）。

## 使用示例

### 1. 寻找满足 P99 Latency <= 2s 的最大并发数

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
┏━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃      ┃      ┃      ┃     Avg ┃     P99 ┃     Avg ┃     P99 ┃     Avg ┃    P99 ┃    Gen. ┃ Success┃
┃Conc. ┃ Rate ┃  RPS ┃ Lat.(s) ┃ Lat.(s) ┃ TTFT(s) ┃ TTFT(s) ┃ TPOT(s) ┃ TPOT(… ┃  toks/s ┃    Rate┃
┡━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│    2 │  INF │ 2.19 │   0.928 │   1.413 │   0.030 │   0.038 │   0.003 │  0.003 │  640.20 │  100.0%│
│    4 │  INF │ 7.18 │   0.783 │   1.635 │   0.033 │   0.050 │   0.003 │  0.004 │ 1013.67 │  100.0%│
│    5 │  INF │ 6.39 │   0.743 │   1.657 │   0.038 │   0.061 │   0.003 │  0.004 │ 1210.93 │  100.0%│
│    6 │  INF │ 3.86 │   0.893 │   3.001 │   0.039 │   0.064 │   0.003 │  0.004 │ 1095.79 │  100.0%│
│    8 │  INF │ 4.03 │   1.286 │   3.181 │   0.044 │   0.081 │   0.003 │  0.004 │ 1615.33 │  100.0%│
└──────┴──────┴──────┴─────────┴─────────┴─────────┴─────────┴─────────┴────────┴─────────┴────────┘


               Best Performance Configuration               
 Highest RPS         Concurrency 2 (INF req/sec)            
 Lowest Latency      Concurrency 2 (2.19 seconds)           

Performance Recommendations:
• Consider lowering concurrency, current load may be too high
• Success rate is low at high concurrency, check system resources or reduce concurrency
2025-12-18 16:32:39 - evalscope - INFO: Performance summary saved to: outputs/20251218_163037/Qwen2.5-0.5B-Instruct/performance_summary.txt
2025-12-18 16:32:39 - evalscope - INFO: SLA Auto-tune Summary:
+--------------------+------------+-----------------+-----------+
| Criteria           | Variable   |   Max Satisfied | Note      |
+====================+============+=================+===========+
| p99_latency <= 2.0 | parallel   |               5 | Satisfied |
+--------------------+------------+-----------------+-----------+
```

### 2. 寻找 TPS 最大的并发数

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

输出示例：
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

### 3. 寻找特定范围满足 TTFT < 0.05s, TTFT < 0.01s 的最大请求速率

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

输出示例：
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
