# DeepSearchQA

## Introduction

[DeepSearchQA](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf) evaluates
deep research agents on difficult web information-seeking questions. The ModelScope dataset
[`google/deepsearchqa`](https://modelscope.cn/datasets/google/deepsearchqa/summary) contains 900 examples in the
`eval` split. Each row contains a question, category, gold answer, and answer type (`Single Answer` or `Set Answer`).

EvalScope implements the ModelScope dataset adapter and the public Kaggle starter-code judging protocol. The official
starter uses Gemini 2.5 Flash with Google Search for answer generation and a Gemini judge. EvalScope does not hard-code
either provider: you can run any OpenAI-compatible model and attach search/fetch tools through AgentLoop MCP servers.

## Installation

Install MCP support if the run uses MCP servers:

```bash
pip install 'evalscope[mcp]'
```

`evalscope[mcp]` includes the official `mcp` Python SDK and `mcp-server-fetch`. Fetch is useful for reading known URLs,
but it is not a search engine. A formal DeepSearchQA agent run should also provide a real search MCP server.

## Search MCP Requirements

DeepSearchQA questions usually require discovering multiple web pages, resolving aliases, and checking several sources.
The agent should have at least two capabilities:

- `search` or `web_search`: accepts a query string and returns ranked results with title, snippet, and URL.
- `fetch` or `fetch_url`: accepts a URL and returns readable page text.

`mcp_server_fetch` only provides the second capability. If it is the only MCP server configured, the model can open URLs
that it already knows, but it cannot reliably discover new sources.

## Native AgentLoop Example

The example below uses Fetch MCP plus a remote search MCP endpoint. Replace the search endpoint with your own
ModelScope MCP, Brave/Tavily-backed MCP, or self-hosted MCP server.

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

## CLI Example

When configuring MCP servers through CLI JSON, include each server's `type` field.

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

## Scoring

The primary metric is `f1_score`; `precision` and `recall` are also reported. EvalScope additionally emits pipeline
health rates from the official starter logic:

- `empty_model_response`
- `empty_auto_rater_response`
- `invalid_auto_rater_response`

Empty model responses and empty/invalid judge responses are excluded from the valid-sample mean precision/recall/F1
denominator and reported separately as rates. Per-sample reviews keep the judge details, including which expected
answers were found and which excessive answers were detected.

## Notes

- `function_calling` expects the model to call `submit(answer=...)` when it is done.
- DeepSearchQA adds a max-step finalization turn: if the loop reaches `max_steps` with an empty completion, the model is
  asked to answer from the information already gathered.
- For comparable runs, keep the judge temperature at `0.0` and use the same judge model across all systems.

## Resources

- [Paper](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf)
- [ModelScope dataset](https://modelscope.cn/datasets/google/deepsearchqa/summary)
- [Agent MCP guide](../user_guides/agent/native.md#mcp-server-tools)
