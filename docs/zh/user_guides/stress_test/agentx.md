# AgentX 服务性能基准

AgentX 用于衡量推理服务处理长编码 Agent 会话时的性能，不评估模型能力或任务正确率。

`agentx` 保留原有 MVP 口径。[AgentX Harness 场景](#agentx-harness) 使用独立 runner，不改变已有命令或默认配置。

## 安装

AgentX 要求 Python 3.11--3.13，并且不包含在默认 Perf extra 中：

```bash
pip install 'evalscope[agentx]'
```

## 服务前置条件

默认 256K workload 需要支持流式 OpenAI Chat Completions 的服务端，并且实际上下文窗口至少为 256K tokens。32K 服务无法运行默认 trace。使用 vLLM 时，请设置 `--max-model-len 262144`，并在运行 AgentX 前通过 `/v1/models` 返回的 `max_model_len` 核验。

## 快速开始

建议先执行短 smoke 测试。它使用一个活跃会话，结果不能与公开成绩比较：

```bash
evalscope perf \
  --scenario '{"name":"agentx","mode":"smoke"}' \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 1
```

## 正式运行

默认运行使用已校验的 ModelScope 256K 镜像、固定 seed 和 30 分钟时长：

```bash
evalscope perf \
  --scenario agentx \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 8
```

`parallel` 是同时运行的 AgentX 会话数。一个会话可能发出多个请求，因此实际请求并发可能更高。

仅在需要与 AIPerf 上游结果兼容时使用 `--data-source huggingface`：

```bash
evalscope perf --scenario agentx --data-source huggingface \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

## 等待在途请求完成

在场景 JSON 中设置 `benchmark_grace_period`，可以控制发送时长结束后 AIPerf 额外等待在途响应的秒数。例如，将 `--scenario agentx` 替换为：

```bash
--scenario '{"name":"agentx","benchmark_grace_period":600}'
```

取值须为有限的非负数；零也会显式透传。省略时保留 AIPerf 默认值（0.12.0 中为 30 秒）。该参数不会改变 `--duration` 或单个请求的 `request_timeout_seconds`。在此期间完成的响应会计入指标，阶段总耗时可能超过发送时长。对比不同引擎时请使用相同设置，并检查阶段日志中的取消请求数。

## 高级配置

JSON 形式可用于选择其他 workload 变体、配置请求时间参数或记录部署元数据。`full` 变体需要服务端支持更大的上下文窗口：

```bash
evalscope perf \
  --scenario '{"name":"agentx","variant":"full","num_gpus":8,"engine":"vllm"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

## 结果

每个并发点会生成 `agentx_summary.json`，整体汇总为 `agentx_sweep_summary.json`。原始 benchmark 文件保存在相邻的 `aiperf/` 目录，便于排查问题。

smoke、取消、失败、缩短或哈希不符的运行，`submission_valid` 均为 false。需要可比较结果时，请使用符合要求的数据集和时长执行正式运行。

## MVP 兼容性

原有 `agentx` 场景固定 `aiperf==0.12.0` 及其 `inferencex-agentx-mvp` 场景。它不等同于当前 InferenceX AgentX v1.0 榜单口径，后者的 profiling 和 warmup 规则已变化。AgentX 指标不能作为模型质量或 Agent 任务得分。

## AgentX Harness

新增场景固定使用 `agentx-harness==1.0.13`，均调用其原生 `agentx` 场景：

| 场景 | 语料 | 默认正式运行时长 | 默认 seed |
| --- | --- | --- | --- |
| `agentx_harness` | 完整 Weka 语料 | 3600 秒 | 42 |
| `agentx_harness_256k` | 256K Weka 子集 | 3600 秒 | 42 |

要求 POSIX 和 Python 3.11--3.13。**不要在同一环境安装 `evalscope[agentx]` 和 `evalscope[agentx_harness]`**：两个 runner 均安装 `aiperf` Python 包。检测到混装会拒绝运行；应重建干净环境，而不是在混装环境直接卸载其中一个包。

```bash
python3 -m venv .venv/agentx_harness
.venv/agentx_harness/bin/python -m pip install -e '.[agentx_harness]'

# Smoke check: four traces, 60 seconds, not a valid submission.
.venv/agentx_harness/bin/evalscope perf \
  --scenario '{"name":"agentx_harness_256k","mode":"smoke"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 1

# Benchmark: use agentx_harness for the full corpus.
.venv/agentx_harness/bin/evalscope perf \
  --scenario agentx_harness_256k \
  --model YOUR_MODEL --tokenizer-path YOUR_TOKENIZER_PATH_OR_ID \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

以上 editable 安装命令用于源码仓库；发行包可在新环境安装 `evalscope[agentx_harness]`。已有 MVP 环境保持独立。

256K 场景要求服务至少支持 256K 上下文；完整语料可能需要更大的上下文窗口。均要求流式 OpenAI Chat Completions 和 `ignore_eos=true` 支持。`--parallel` 表示活跃会话树数，并非单次请求并发。Harness 自行管理 warmup 和 drain，因此墙钟耗时可能超过 `--duration`。正式运行时长须为有限值且至少 900 秒；smoke 默认 60 秒、4 条 trace，只有 smoke 允许 `trace_limit`。

JSON 支持前文的时间、tokenizer 和部署元数据选项。`tokenizer_trust_remote_code` 默认 false，执行远程 tokenizer 代码须显式开启，建议同时固定 `tokenizer_revision`。与场景名称冲突的 `variant` 会被拒绝。不支持 `--number`、open-loop/rate 控制、multi-turn、SLA 调优、Perf warmup 或自定义 dataset arguments。

Python 使用相同的类型化 `Arguments` 接口：

```python
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

results = run_perf_benchmark(Arguments(
    scenario={'name': 'agentx_harness_256k', 'mode': 'smoke'},
    model='YOUR_MODEL',
    tokenizer_path='YOUR_TOKENIZER_PATH_OR_ID',
    url='http://localhost:8000/v1/chat/completions',
    parallel=[1],
))
```

默认使用固定身份的上游公开数据集，并校验原生别名实际读取的当前仓库 head 和预期校验和。`--data-source modelscope` 或 `--dataset-path` 可选择镜像/本地语料。本地 JSONL 目录直接交给 harness loader，不进行 MVP 的逐 trace JSON 转换。正式运行拒绝哈希不匹配的数据；未校验数据仅允许 smoke。

每个并发点生成 `<scenario>/parallel_<n>/agentx_summary.json`（schema `2.0`）、`harness/` 原始产物和 `harness.log`；场景目录下另有 `agentx_sweep_summary.json`。汇总保留 harness/原生引擎版本、数据来源、原生无效原因和指标。失败或取消会保留诊断并终止 sweep。已有原生产物目录不可复用，以免误读旧结果。

原生 submission validity 为最终依据：镜像和 smoke 使用 `unsafe_override`，不会被提升为有效提交。不要混合比较 harness 与 MVP 数值。本次提供 CLI/Python 接口，不提供 Web dashboard 专用结果表或归档展示。
