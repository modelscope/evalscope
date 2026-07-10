# 参数说明

Perf 配置按职责拆分，不再使用一个全局参数对象。

| 分组 | CLI 示例 | Python 模型 |
| --- | --- | --- |
| 目标服务 | `--model`、`--protocol`、`--base-url`、超时 | `TargetConfig` |
| Workload | `--workload`、`--workload-path`、`--data-source` | `WorkloadConfig` |
| 生成参数 | `--max-tokens`、`--temperature`、`--top-p` | `GenerationConfig` |
| 负载 | `--mode`、`--requests`、`--concurrency`、`--request-rate` | 三种 Load 类型 |
| 运行时 | `--seed`、`--dataset-workers`、`--queue-size` | `RuntimeConfig` |
| 输出 | `--output-root`、`--run-id`、`--overwrite` | `OutputConfig` |

负载模式：

- `closed_loop`：需要并发数，以及请求数或时长。
- `open_loop`：需要请求到达率、请求数或时长，并显式设置有界的 `max_outstanding`。
- `conversation`：需要多轮 workload、并发会话数，以及会话数或时长。

多个负载点使用可重复的 `--load '<JSON>'`：

```bash
--load '{"mode":"closed_loop","concurrency":1,"request_count":100}' \
--load '{"mode":"closed_loop","concurrency":8,"request_count":400}'
```

`--warmup-count` 与 `--warmup-ratio` 互斥。不兼容的 workload、protocol 和 load 组合会在发送网络请求前报错。

## 旧 CLI 参数兼容

已有的 `evalscope perf` 命令会通过一层轻量翻译继续工作。常见映射包括：`--url` 到 `--base-url`、`--api` 到 `--protocol`、`--dataset` 到 `--workload`、`--number` 到 `--requests`、`--parallel` 到 `--concurrency`，以及 `--rate --open-loop` 到 `--request-rate --mode open_loop`。旧的列表 sweep 会被转换为显式 load 配置。

CLI 会输出弃用提示并给出对应的新参数。该兼容只覆盖命令行解析，不恢复已删除的 Python `Arguments` API 或旧结果结构。
