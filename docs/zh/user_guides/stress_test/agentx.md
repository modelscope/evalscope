# AgentX MVP 服务性能基准

AgentX 是推理服务性能 workload，不是 Agent 能力或任务正确率评测。它通过 AIPerf 回放长上下文 coding-agent session tree，保留前缀复用、思考间隔和子 agent 依赖。

## 安装

AgentX 要求 Python 3.11--3.13，并且不包含在默认 Perf extra 中：

```bash
pip install 'evalscope[agentx]'
```

## 运行

简写默认使用已校验的 ModelScope 256K 镜像、固定 seed `20260707` 和 AIPerf AgentX MVP 的 1800 秒默认时长：

```bash
evalscope perf \
  --scenario agentx \
  --model YOUR_MODEL \
  --tokenizer-path YOUR_HF_TOKENIZER \
  --url http://localhost:8000/v1/chat/completions \
  --parallel 8 16
```

若需要 AIPerf 上游的官方数据源状态，显式选择 Hugging Face：

```bash
evalscope perf --scenario agentx --data-source huggingface \
  --model YOUR_MODEL --tokenizer-path YOUR_HF_TOKENIZER \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

JSON 形式只放 AgentX 专属元数据，不重复连接参数：

```bash
evalscope perf \
  --scenario '{"name":"agentx","variant":"full","max_context_length":1000000,"num_gpus":8,"engine":"vllm"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_HF_TOKENIZER \
  --url http://localhost:8000/v1/chat/completions --parallel 8
```

`parallel` 表示活跃 agent session tree 数，不是 HTTP 请求上限；子 agent fan-out 后实际在途请求可能更高。

## 烟测与产物

使用 `mode=smoke` 执行短运行；它默认四条轨迹、60 秒，结果明确不可比较：

```bash
evalscope perf \
  --scenario '{"name":"agentx","mode":"smoke"}' \
  --model YOUR_MODEL --tokenizer-path YOUR_HF_TOKENIZER \
  --url http://localhost:8000/v1/chat/completions --parallel 1
```

每个并发点的 AIPerf 原始文件保持不变，位于 `agentx_<variant>/parallel_<N>/aiperf/`；同目录的 `agentx_summary.json` 是 EvalScope 归一化摘要，根目录 `agentx_sweep_summary.json` 汇总 sweep。

烟测、取消、失败、缩短或哈希不符的运行，`submission_valid` 一律为 false。对于逐字节校验的 ModelScope 镜像，若 AIPerf 唯一无效原因是本地镜像所需的 `unsafe_override`，EvalScope 会在归一化摘要中重新认证；AIPerf 原始有效性字段始终原样保留。

## 兼容性

首版固定 `aiperf==0.12.0` 及其 `inferencex-agentx-mvp` 场景。它不等同于当前 InferenceX AgentX v1.0 榜单口径，后者的 profiling 和 warmup 规则已变化。AgentX 指标不能作为模型质量或 Agent 任务得分。
