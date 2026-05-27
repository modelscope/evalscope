# SLA Auto-Tuning

The SLA (Service Level Agreement) auto-tuning feature allows users to define service quality metrics (such as latency and throughput), and the tool will automatically adjust request pressure (concurrency or request rate) to find the maximum pressure value that the service can sustain while meeting these metrics.

## Features

- **Automatic Detection**: Uses binary search algorithm to automatically find the maximum concurrency (`parallel`) or request rate (`rate`) that satisfies SLA constraints.
- **Multi-Metric Support**: Supports end-to-end latency (Latency), time to first token (TTFT), time per output token (TPOT), as well as request throughput (RPS) and token throughput (TPS).
- **Flexible Constraints**: Supports setting upper limits (e.g., `p99_latency <= 2s`) or finding extremes (e.g., `tps: max`).
- **Stable Results**: Each test point runs multiple times by default and takes the average to reduce network fluctuation interference.

## Parameter Description

| Parameter | Type | Description | Default |
|------|------|------|--------|
| `--sla-auto-tune` | `bool` | Whether to enable SLA auto-tuning mode | `False` |
| `--sla-variable` | `str` | Variable for auto-tuning<br>Options: `parallel` (concurrency), `rate` (request rate) | `parallel` |
| `--sla-params` | `str` | SLA constraint conditions, JSON string, supports multiple constraint groups (AND/OR logic), see [description below](#sla-params-logic) | `None` |
| `--sla-upper-bound` | `int` | Upper bound of the tuned SLA variable search range | `65536` |
| `--sla-lower-bound` | `int` | Lower bound of the tuned SLA variable search range | `1` |
| `--sla-fixed-parallel` | `int` | Fixed parallel workers used when `--sla-variable=rate`; defaults to `--sla-upper-bound` for backward compatibility | `None` |
| `--sla-num-runs` | `int` | Number of repeated runs per test point (average taken to reduce fluctuation) | `3` |
| `--sla-number-multiplier` | `float` | Multiplier of total requests relative to the tuned variable (concurrency or rate), i.e. `number = round(variable × N)`; defaults to `2` when not set | `None` |

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

(sla-params-logic)=
## `--sla-params` Logic

`--sla-params` accepts a **JSON array**, where each element is an **object (group)**. Logic rules are as follows:

- **Multiple metrics within the same object**: **AND** (must all be satisfied simultaneously)
- **Between different objects**: **OR** (any one group satisfied is sufficient)

The overall semantics are: `(Group1 ConditionA AND Group1 ConditionB) OR (Group2 ConditionC AND Group2 ConditionD) OR ...`

### AND Example: Satisfy TTFT and TPOT Simultaneously

Write multiple metrics in the **same object** to indicate they must **all** be satisfied:

```bash
--sla-params '[{"avg_ttft": "<=2", "avg_tpot": "<=0.05"}]'
```

Meaning: Find the maximum concurrency satisfying **`avg_ttft <= 2s` AND `avg_tpot <= 0.05s`**. Only when both metrics are met does that concurrency level pass.

### OR Example: Independently Evaluate Multiple TTFT Thresholds

Write each metric in a **different object** so each group of conditions is evaluated **independently**:

```bash
--sla-params '[{"p99_ttft": "<0.05"}, {"p99_ttft": "<0.01"}]'
```

Meaning: Find the maximum request rate satisfying **`p99_ttft < 0.05s`** and satisfying **`p99_ttft < 0.01s`** separately, each outputting results independently.

### AND + OR Combined Example

```bash
--sla-params '[{"avg_ttft": "<=1", "avg_tpot": "<=0.05"}, {"p99_latency": "<=5"}]'
```

Meaning:
- **Group 1**: `avg_ttft <= 1s` **AND** `avg_tpot <= 0.05s` (both satisfied simultaneously)
- **Group 2**: `p99_latency <= 5s`
- Each group independently completes a binary search and outputs its maximum concurrency value separately.

### Extremum Optimization Mode

When the array has **only one object with only one metric**, and the operator is `max` or `min`, the tool enters extremum optimization mode and directly finds the pressure value corresponding to the optimal metric:

```bash
--sla-params '[{"tps": "max"}]'
```

Meaning: Find the concurrency corresponding to maximum TPS (token throughput).

## Workflow

1. **Baseline Test**: Start testing with the user-specified initial `parallel` or `rate` (recommended to set a small value, such as 1 or 2).
2. **Boundary Detection**:
   - If current metrics meet SLA, double the pressure until SLA is first violated or `--sla-upper-bound` is reached.
   - If initial metrics violate SLA, halve the pressure to find a lower bound that satisfies conditions.
3. **Binary Search**: Perform binary search within the determined boundary window to precisely lock in the maximum pressure value that "just doesn't violate" SLA.
4. **Result Confirmation**: Each test point runs `--sla-num-runs` times (default 3), taking the average for judgment.
5. **Report Output**: After testing, output a summary of the tuning process and final results.

> **Note**: If the request success rate during testing is below 100%, that test point will be considered failed (violating SLA).
>
> When `--sla-variable=rate`, use `--sla-fixed-parallel` to explicitly control the fixed concurrency. If not set, the implementation falls back to `--sla-upper-bound` for backward compatibility.

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
                Performance Overview
┏━━━━━━┳━━━━━━┳━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃Conc. ┃ Rate ┃ Num ┃  RPS ┃   Gen/s ┃ Success ┃
┡━━━━━━╇━━━━━━╇━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│    2 │    - │  20 │ 2.19 │  640.20 │  100.0% │
│    4 │    - │  20 │ 7.18 │ 1013.67 │  100.0% │
│    5 │    - │  20 │ 6.39 │ 1210.93 │  100.0% │
│    6 │    - │  20 │ 3.86 │ 1095.79 │  100.0% │
│    8 │    - │  20 │ 4.03 │ 1615.33 │  100.0% │
└──────┴──────┴─────┴──────┴─────────┴─────────┘
2025-12-18 16:32:39 - evalscope - INFO: SLA Auto-tune Summary:
+--------------------+------------+-----------------+-----------+
| Criteria           | Variable   |   Max Satisfied | Note      |
+====================+============+=================+===========+
| p99_latency <= 2.0 | parallel   |               5 | Satisfied |
+--------------------+------------+-----------------+-----------+
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
                  Performance Overview
┏━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Conc. ┃ Rate ┃  Num ┃  RPS ┃   Gen/s ┃ Success ┃
┡━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│    32 │    - │  ... │ 5.68 │ 5813.67 │  100.0% │
│    64 │    - │  ... │ 5.76 │ 5902.57 │  100.0% │
│   128 │    - │  ... │ 6.96 │ 7124.25 │  100.0% │
│   256 │    - │  ... │ 7.81 │ 8000.89 │  100.0% │
│   384 │    - │  ... │ 7.87 │ 8057.14 │  100.0% │
│   ... │    - │  ... │  ... │     ... │  100.0% │
│   512 │    - │  ... │ 7.76 │ 7941.28 │  100.0% │
└───────┴──────┴──────┴──────┴─────────┴─────────┘
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
 --sla-fixed-parallel 40 \
 --sla-lower-bound 10 \
 --sla-upper-bound 40
```

Example output:
```text
              Performance Overview
┏━━━━━━┳━━━━━━┳━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃Conc. ┃ Rate ┃ Num ┃   RPS ┃   Gen/s ┃ Success ┃
┡━━━━━━╇━━━━━━╇━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│   40 │   10 │ ... │  5.16 │  948.33 │  100.0% │
│   40 │   15 │ ... │  9.78 │ 2249.29 │  100.0% │
│   40 │   17 │ ... │  8.17 │ 1530.79 │  100.0% │
│   40 │   18 │ ... │ 10.30 │ 2466.09 │  100.0% │
│   40 │   19 │ ... │ 12.83 │ 2296.22 │  100.0% │
│   40 │   20 │ ... │ 11.81 │ 2435.94 │  100.0% │
└──────┴──────┴─────┴───────┴─────────┴─────────┘
2025-12-18 16:19:48 - evalscope - INFO: SLA Auto-tune Summary:
+-----------------+------------+-----------------+----------------------------+
| Criteria        | Variable   | Max Satisfied   | Note                       |
+=================+============+=================+============================+
| p99_ttft < 0.05 | rate       | 19              | Satisfied                  |
+-----------------+------------+-----------------+----------------------------+
| p99_ttft < 0.01 | rate       | None            | Failed at lower bound (10) |
+-----------------+------------+-----------------+----------------------------+
```
