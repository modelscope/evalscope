# DeepSearchQA

## 简介

[DeepSearchQA](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf) 用于评估
Deep Research Agent 处理复杂联网问答的能力。ModelScope 数据集
[`google/deepsearchqa`](https://modelscope.cn/datasets/google/deepsearchqa/summary) 在 `eval` split 中包含 900 道题。
每条样本包含问题、类别、标准答案和答案类型（`Single Answer` 或 `Set Answer`）。

EvalScope 接入 ModelScope 数据集，并对齐公开 Kaggle starter code 的判分协议。官方 starter 使用 Gemini 2.5 Flash
和 Google Search 生成答案，再用 Gemini 评分；EvalScope 不绑定模型或搜索服务，可以使用任意 OpenAI-compatible 模型，
并通过 AgentLoop MCP server 接入搜索和网页抓取工具。

## 安装

如果运行时需要 MCP server，安装 MCP 依赖：

```bash
pip install 'evalscope[mcp]'
```

`evalscope[mcp]` 自带官方 `mcp` Python SDK 和 `mcp-server-fetch`。Fetch 适合读取已知 URL，但不是搜索引擎。
正式 DeepSearchQA agent 评测还应配置真实搜索 MCP server。

## 搜索 MCP 要求

DeepSearchQA 通常需要发现多个网页、解析别名，并交叉验证多个来源。建议至少提供两类工具：

- `search` 或 `web_search`：输入 query，返回带 title、snippet、URL 的排序结果。
- `fetch` 或 `fetch_url`：输入 URL，返回可读网页正文。

`mcp_server_fetch` 只提供第二类能力。如果只配置 fetch，模型只能打开已经知道的 URL，无法可靠发现新来源。

## Native AgentLoop 示例

下面示例同时配置 Fetch MCP 和一个远端搜索 MCP endpoint。搜索 endpoint 可替换为 ModelScope MCP、Brave/Tavily
封装 MCP，或自建 MCP server。

```python
import os
import sys

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigHTTP, MCPServerConfigStdio

task_config = TaskConfig(
    model='YOUR_AGENT_MODEL',
    api_url='OPENAI_COMPATIBLE_URL',
    api_key=os.getenv('MODEL_API_KEY'),
    eval_type='openai_api',
    datasets=['deepsearchqa'],
    eval_batch_size=1,
    limit=10,
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'YOUR_JUDGE_MODEL',
        'api_url': 'OPENAI_COMPATIBLE_JUDGE_URL',
        'api_key': os.getenv('JUDGE_API_KEY'),
        'generation_config': {'temperature': 0.0},
    },
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=30,
        mcp_servers=[
            MCPServerConfigStdio(
                name='fetch',
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
            ),
            MCPServerConfigHTTP(
                name='search',
                url='https://mcp.api-inference.modelscope.net/<search-server-id>/mcp',
                headers={'Authorization': 'Bearer <MCP_TOKEN>'},
            ),
        ],
        kwargs={
            'system_prompt': (
                'Use the available search and fetch tools to research the question. '
                'Cross-check important facts before answering. When finished, call submit(answer=...) with only the '
                'final answer or complete answer set.'
            ),
        },
    ),
)

run_task(task_config)
```

## CLI 示例

通过 CLI JSON 配置 MCP server 时，每个 server 都要写 `type` 字段。

```bash
evalscope eval \
  --model YOUR_AGENT_MODEL \
  --api-url OPENAI_COMPATIBLE_URL \
  --api-key "$MODEL_API_KEY" \
  --datasets deepsearchqa \
  --limit 10 \
  --eval-batch-size 1 \
  --judge-strategy llm \
  --judge-model-args '{"model_id":"YOUR_JUDGE_MODEL","api_url":"OPENAI_COMPATIBLE_JUDGE_URL","api_key":"'"$JUDGE_API_KEY"'","generation_config":{"temperature":0.0}}' \
  --agent-config '{"mode":"native","strategy":"function_calling","max_steps":30,"mcp_servers":[{"type":"stdio","name":"fetch","command":"python","args":["-m","mcp_server_fetch","--ignore-robots-txt"]},{"type":"http","name":"search","url":"https://mcp.api-inference.modelscope.net/<search-server-id>/mcp","headers":{"Authorization":"Bearer <MCP_TOKEN>"}}],"kwargs":{"system_prompt":"Use search and fetch tools to research the question. Cross-check important facts before answering. When finished, call submit(answer=...) with only the final answer."}}'
```

## 评分

主指标是 `f1_score`，同时报告 `precision` 和 `recall`。EvalScope 还会按照官方 starter 逻辑输出评测流程健康度指标：

- `empty_model_response`
- `empty_auto_rater_response`
- `invalid_auto_rater_response`

空模型回答、空 judge 回复和非法 judge 回复不会进入有效样本 mean precision/recall/F1 的分母，而是单独以 rate 报告。
每条样本的 review 仍会保留 judge 细节，包括哪些标准答案被命中，以及检测到了哪些额外答案。

## 注意事项

- `function_calling` 模式要求模型完成后调用 `submit(answer=...)`。
- DeepSearchQA 增加了 max-step finalization：如果到达 `max_steps` 时 completion 为空，会要求模型基于已收集信息输出最终答案。
- 为了比较不同模型，建议固定同一个 judge model，并设置 judge temperature 为 `0.0`。

## 相关资源

- [论文](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf)
- [ModelScope 数据集](https://modelscope.cn/datasets/google/deepsearchqa/summary)
- [Agent MCP 配置指南](../user_guides/agent/native.md#mcp-server-tools)
