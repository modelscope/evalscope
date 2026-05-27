# SLA 自动调优

SLA (Service Level Agreement) 自动调优功能允许用户定义服务质量指标（如延迟、吞吐量），工具将自动调整请求压力（并发数或请求速率），寻找服务能够满足这些指标的最大压力值。

## 功能特性

- **自动探测**：通过二分查找算法，自动寻找满足 SLA 约束的最大并发数 (`parallel`) 或请求速率 (`rate`)。
- **多指标支持**：支持端到端延迟（Latency）、首字延迟（TTFT）、单字输出延迟（TPOT）以及请求吞吐量（RPS）、token吞吐量（TPS）等指标。
- **灵活约束**：支持设置上限（如 `p99_latency <= 2s`）或寻找极值（如 `tps: max`）。
- **结果稳定**：每个测试点默认运行多次取平均值，减少网络波动干扰。

## 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--sla-auto-tune` | `bool` | 是否启用 SLA 自动调优模式 | `False` |
| `--sla-variable` | `str` | 自动调优的变量<br>可选：`parallel`（并发数）、`rate`（请求速率） | `parallel` |
| `--sla-params` | `str` | SLA 约束条件，JSON 字符串，支持多组约束（AND/OR 逻辑），详见[下方说明](#sla-params-逻辑说明) | `None` |
| `--sla-upper-bound` | `int` | 被调优变量的搜索上界 | `65536` |
| `--sla-lower-bound` | `int` | 被调优变量的搜索下界 | `1` |
| `--sla-fixed-parallel` | `int` | 在 `--sla-variable=rate` 时使用的固定并发数；未设置时默认回退到 `--sla-upper-bound` 以兼容旧行为 | `None` |
| `--sla-num-runs` | `int` | 每个测试点的重复运行次数（取平均值，减少波动） | `3` |
| `--sla-number-multiplier` | `float` | 每次测试时请求总数相对于被调优变量（并发数或速率）的倍数，即 `number = round(variable × N)`；未设置时默认为 `2` | `None` |

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

(sla-params-逻辑说明)=
## `--sla-params` 逻辑说明

`--sla-params` 接受一个 **JSON 数组**，数组中每个元素为一个**对象（组）**。逻辑规则如下：

- **同一对象内的多个指标**：**AND**（必须同时满足）
- **不同对象之间**：**OR**（任意一组满足即可）

即整体语义为：`(组1条件A AND 组1条件B) OR (组2条件C AND 组2条件D) OR ...`

### AND 示例：同时满足 TTFT 和 TPOT

将多个指标写在**同一个对象**中，表示这些指标必须**同时**满足：

```bash
--sla-params '[{"avg_ttft": "<=2", "avg_tpot": "<=0.05"}]'
```

含义：寻找满足 **`avg_ttft <= 2s` AND `avg_tpot <= 0.05s`** 的最大并发数。只有两个指标都达标，该并发级别才算通过。

### OR 示例：独立评估多个 TTFT 阈值

将每个指标写在**不同对象**中，各组条件**独立**进行二分搜索：

```bash
--sla-params '[{"p99_ttft": "<0.05"}, {"p99_ttft": "<0.01"}]'
```

含义：分别寻找满足 **`p99_ttft < 0.05s`** 以及满足 **`p99_ttft < 0.01s`** 的最大请求速率，各自独立输出结果。

### AND + OR 组合示例

```bash
--sla-params '[{"avg_ttft": "<=1", "avg_tpot": "<=0.05"}, {"p99_latency": "<=5"}]'
```

含义：
- **组1**：`avg_ttft <= 1s` **AND** `avg_tpot <= 0.05s`（同时满足）
- **组2**：`p99_latency <= 5s`
- 两组各自独立完成二分搜索，分别输出最大并发值。

### 极值优化模式

当数组只有**一个对象且只有一个指标**，且操作符为 `max` 或 `min` 时，进入极值优化模式，直接寻找指标最优时对应的压力值：

```bash
--sla-params '[{"tps": "max"}]'
```

含义：寻找 TPS（token 吞吐量）最大时对应的并发数。

## 工作流程

1. **基准测试**：以用户指定的初始 `parallel` 或 `rate` 开始测试（建议设置为较小值，如 1 或 2）。
2. **边界探测**：
   - 如果当前指标满足 SLA，将压力翻倍，直到首次违反 SLA 或达到 `--sla-upper-bound`。
   - 如果初始指标即违反 SLA，将压力减半，寻找满足条件的下界。
3. **二分查找**：在确定的边界窗口内进行二分查找，精确锁定“刚好不违约”的最大压力值。
4. **结果确认**：每个测试点会运行 `--sla-num-runs` 次（默认 3 次），取平均值进行判断。
5. **报告输出**：测试结束后，输出调优过程摘要及最终结果。

> **注意**：如果测试过程中请求成功率（Success Rate）低于 100%，该测试点将被视为失败（违反 SLA）。
>
> 当 `--sla-variable=rate` 时，可以通过 `--sla-fixed-parallel` 显式指定固定并发数；未设置时默认沿用 `--sla-upper-bound`，以兼容旧版本行为。

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
 --sla-fixed-parallel 40 \
 --sla-lower-bound 10 \
 --sla-upper-bound 40
```

输出示例：
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
