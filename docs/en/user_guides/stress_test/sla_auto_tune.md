# SLA Auto-Tuning

SLA (Service Level Agreement) auto-tuning tests request pressure (concurrency or request rate) against service quality metrics. It reports the best tested value for each constraint group or optimization objective.

## Features

- **Automatic Detection**: Uses exponential bracketing and binary search to locate a qualifying concurrency (`parallel`) or request rate (`rate`).
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

> **Note**: Latency thresholds accept an `s` or `ms` suffix, e.g. `{"avg_ttft": "<=2s"}` or `{"avg_tpot": "<=50ms"}`. A bare number is read in the unit the reports use for that metric (`Avg TTFT (ms)` is milliseconds, `Avg Latency (s)` is seconds), so a value can be copied off a report as-is. Every check logs the unit it resolved, e.g. `avg_ttft = 40 ms | Expect <= 2000 ms | PASSED`.

Embedding and rerank APIs support only `avg_latency`, `p99_latency`, and `rps` in SLA mode. Other metric/API combinations fail before benchmarking. A metric missing or non-finite in any repeated run makes that tested pressure ineligible; a real numeric zero remains valid.

If you copied an earlier example such as `{"avg_ttft": "<=2", "avg_tpot": "<=0.05"}` intending **seconds**, add the unit suffixes: `{"avg_ttft": "<=2s", "avg_tpot": "<=0.05s"}`. Without suffixes, those TTFT/TPOT thresholds are read as milliseconds. The same applies to earlier `p99_ttft` examples such as `<0.05`: write `<0.05s` or `<50ms`.

(sla-params-logic)=
## `--sla-params` Logic

`--sla-params` accepts a **JSON array**, where each element is an **object (group)**. Logic rules are as follows:

- **Multiple metrics within the same object**: **AND** (must all be satisfied simultaneously)
- **Between different objects**: **OR** (any one group satisfied is sufficient)

The overall semantics are: `(Group1 ConditionA AND Group1 ConditionB) OR (Group2 ConditionC AND Group2 ConditionD) OR ...`

### AND Example: Satisfy TTFT and TPOT Simultaneously

Write multiple metrics in the **same object** to indicate they must **all** be satisfied:

```bash
--sla-params '[{"avg_ttft": "<=2s", "avg_tpot": "<=50ms"}]'
```

Meaning: Find the maximum concurrency satisfying **`avg_ttft <= 2s` AND `avg_tpot <= 50ms`**. Only when both metrics are met does that concurrency level pass.

### OR Example: Independently Evaluate Multiple TTFT Thresholds

Write each metric in a **different object** so each group of conditions is evaluated **independently**:

```bash
--sla-params '[{"p99_ttft": "<50ms"}, {"p99_ttft": "<10ms"}]'
```

Meaning: Find the maximum request rate satisfying **`p99_ttft < 50ms`** and satisfying **`p99_ttft < 10ms`** separately, each outputting results independently.

### AND + OR Combined Example

```bash
--sla-params '[{"avg_ttft": "<=1s", "avg_tpot": "<=50ms"}, {"p99_latency": "<=5s"}]'
```

Meaning:
- **Group 1**: `avg_ttft <= 1s` **AND** `avg_tpot <= 50ms` (both satisfied simultaneously)
- **Group 2**: `p99_latency <= 5s`
- Each group independently completes a binary search and outputs its maximum concurrency value separately.

### Extremum Optimization Mode

When the array has **only one object with only one metric**, and the operator is `max` or `min`, the tool enters extremum optimization mode and directly finds the pressure value corresponding to the optimal metric:

```bash
--sla-params '[{"tps": "max"}]'
```

Meaning: Find the concurrency corresponding to maximum TPS (token throughput).

`max` and `min` cannot be combined with thresholds or another group. Empty groups are also rejected.

## Workflow

1. **Baseline Test**: Start testing with the user-specified initial `parallel` or `rate` (recommended to set a small value, such as 1 or 2).
2. **Boundary Detection**:
   - If current metrics meet SLA, double the pressure until SLA is first violated or `--sla-upper-bound` is reached.
   - If initial metrics violate SLA, halve the pressure to find a lower bound that satisfies conditions.
3. **Binary Search**: Constraint mode searches the pass/fail boundary. Extremum mode compares adjacent values to locate a peak inside the bracket; if the initial value is ineligible, it first searches below that value.
4. **Run Validation**: Each pressure is tested `--sla-num-runs` times (default 3). Every run must have successful requests and every requested metric; metric values are then averaged. Success counts are summed exactly across runs.
5. **Report Output**: The result contains tested pressure points and one final selection per group.

> **Note**: If the request success rate during testing is below 100%, that test point will be considered failed (violating SLA).
>
> When `--sla-variable=rate`, use `--sla-fixed-parallel` to explicitly control the fixed concurrency. If not set, the implementation falls back to `--sla-upper-bound` for backward compatibility.

The search assumes SLA satisfaction decreases monotonically as pressure rises, or a single peak for `max`/`min` within a valid low-pressure prefix. Non-monotonic or multi-peak behavior can be missed. Selections are labelled **best observed**, not guaranteed global optima. A high `--sla-upper-bound` can still be expensive: at the default bound of 65536, the default multiplier requests 131072 responses per run at that point, repeated three times by default. Set a practical upper bound for the service under test.

## Result Contract

The Python return value and `sla_summary.json` keep the existing pressure-keyed dictionary shape. No version field or top-level metadata is added. Each `parallel_N` (or `rate_N`) maps to a result containing `metrics` and `percentiles`, so existing pressure-keyed readers continue to work.

```json
{
  "parallel_2": {
    "metrics": {"Total Requests": 12, "Success Requests": 12},
    "percentiles": {"Percentiles": ["99%"]}
  }
}
```

The returned Python dictionary has `probes` and `selections` attributes for execution-order details and the final choice for each criterion group. These attributes are neither written to JSON nor added as dictionary keys. The service `result` field also keeps the existing shape; its `table` field displays the selections.

Console and service tables display the exact `succeeded_requests/total_requests` count. A pressure with failed requests or a missing metric is marked `FAILED`, even if a rounded percentage would appear to be 100%.
`valid` means all requested metrics were present across all groups. Groups are still evaluated independently: a missing metric in one group does not prevent another group from selecting the same pressure when its own metrics are present.

## Usage Examples

The log excerpts below show representative probe rows; an actual run may test additional pressure values.

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
 --sla-params '[{"p99_latency": "<=2s"}]' \
 --parallel 2 \
 --sla-upper-bound 64
```

```text
SLA Auto-tune Summary:
Probes:
┌────────────┬───────────────────┬──────────────┬──────────┬──────────┐
│   Pressure │ Succeeded/Total   │ Run status   │ Groups   │ Reason   │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│          2 │ 12/12             │ VALID        │ 1:PASS   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│          4 │ 24/24             │ VALID        │ 1:PASS   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│          5 │ 30/30             │ VALID        │ 1:PASS   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│          6 │ 36/36             │ VALID        │ 1:FAIL   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│          8 │ 48/48             │ VALID        │ 1:FAIL   │ -        │
└────────────┴───────────────────┴──────────────┴──────────┴──────────┘

Selections:
┌──────────────────┬────────────┬───────────────┬───────────────────────────────────────────┐
│ Criteria         │   Selected │ Status        │ Reason                                    │
├──────────────────┼────────────┼───────────────┼───────────────────────────────────────────┤
│ p99_latency <=2s │          5 │ best_observed │ Satisfied at the selected tested pressure │
└──────────────────┴────────────┴───────────────┴───────────────────────────────────────────┘
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
SLA Auto-tune Summary:
Probes:
┌────────────┬───────────────────┬──────────────┬──────────┬──────────┐
│   Pressure │ Succeeded/Total   │ Run status   │ Groups   │ Reason   │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│         32 │ 192/192           │ VALID        │ 1:PASS   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│         64 │ 384/384           │ VALID        │ 1:PASS   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│        128 │ 768/768           │ VALID        │ 1:PASS   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│        256 │ 1536/1536         │ VALID        │ 1:PASS   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│        384 │ 2304/2304         │ VALID        │ 1:PASS   │ -        │
├────────────┼───────────────────┼──────────────┼──────────┼──────────┤
│        512 │ 3072/3072         │ VALID        │ 1:PASS   │ -        │
└────────────┴───────────────────┴──────────────┴──────────┴──────────┘

Selections:
┌────────────┬────────────┬───────────────┬──────────────────────────────────┐
│ Criteria   │   Selected │ Status        │ Reason                           │
├────────────┼────────────┼───────────────┼──────────────────────────────────┤
│ tps max    │        384 │ best_observed │ Best observed tps: 8057.14 tok/s │
└────────────┴────────────┴───────────────┴──────────────────────────────────┘
```

### 3. Find Maximum Request Rate Meeting TTFT < 50ms and TTFT < 10ms in Specific Range

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
 --sla-params '[{"p99_ttft": "<50ms"}, {"p99_ttft": "<10ms"}]' \
 --rate 2 \
 --sla-num-runs 1 \
 --sla-fixed-parallel 40 \
 --sla-lower-bound 10 \
 --sla-upper-bound 40
```

Example output:
```text
SLA Auto-tune Summary:
Probes:
┌────────────┬───────────────────┬──────────────┬────────────────┬──────────┐
│   Pressure │ Succeeded/Total   │ Run status   │ Groups         │ Reason   │
├────────────┼───────────────────┼──────────────┼────────────────┼──────────┤
│         10 │ 20/20             │ VALID        │ 1:PASS, 2:FAIL │ -        │
├────────────┼───────────────────┼──────────────┼────────────────┼──────────┤
│         15 │ 30/30             │ VALID        │ 1:PASS, 2:FAIL │ -        │
├────────────┼───────────────────┼──────────────┼────────────────┼──────────┤
│         17 │ 34/34             │ VALID        │ 1:PASS, 2:FAIL │ -        │
├────────────┼───────────────────┼──────────────┼────────────────┼──────────┤
│         18 │ 36/36             │ VALID        │ 1:PASS, 2:FAIL │ -        │
├────────────┼───────────────────┼──────────────┼────────────────┼──────────┤
│         19 │ 38/38             │ VALID        │ 1:PASS, 2:FAIL │ -        │
├────────────┼───────────────────┼──────────────┼────────────────┼──────────┤
│         20 │ 40/40             │ VALID        │ 1:FAIL, 2:FAIL │ -        │
└────────────┴───────────────────┴──────────────┴────────────────┴──────────┘

Selections:
┌────────────────┬────────────┬───────────────┬───────────────────────────────────────────┐
│ Criteria       │ Selected   │ Status        │ Reason                                    │
├────────────────┼────────────┼───────────────┼───────────────────────────────────────────┤
│ p99_ttft <50ms │ 19         │ best_observed │ Satisfied at the selected tested pressure │
├────────────────┼────────────┼───────────────┼───────────────────────────────────────────┤
│ p99_ttft <10ms │ None       │ none          │ Failed at lower bound (10)                │
└────────────────┴────────────┴───────────────┴───────────────────────────────────────────┘
```
