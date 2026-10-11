"""Run one DeepSearchQA sample with anonymous Parallel search and fetch tools.

Install with ``pip install -e '.[mcp]'`` from the repository root. Set
MODEL_NAME, MODEL_API_URL and, if required by your model service, MODEL_API_KEY.
Optional JUDGE_MODEL_NAME, JUDGE_API_URL and JUDGE_API_KEY select a separate judge.
Then run ``python examples/agent/parallel_search.py``.
"""

import os

from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigHTTP


def build_task_config() -> TaskConfig:
    """Build an opt-in research evaluation using the existing native MCP loader."""
    model = os.environ['MODEL_NAME']
    api_url = os.environ['MODEL_API_URL']
    api_key = os.getenv('MODEL_API_KEY')
    return TaskConfig(
        model=model,
        api_url=api_url,
        api_key=api_key,
        eval_type='openai_api',
        datasets=['deepsearchqa'],
        eval_batch_size=1,
        limit=1,
        judge={
            'strategy': 'llm',
            'models': {
                'model_id': os.getenv('JUDGE_MODEL_NAME', model),
                'api_url': os.getenv('JUDGE_API_URL', api_url),
                'api_key': os.getenv('JUDGE_API_KEY', api_key),
                'generation_config': {'temperature': 0.0},
            },
        },
        agent_config=NativeAgentConfig(
            strategy='function_calling',
            max_steps=30,
            mcp_servers=[
                MCPServerConfigHTTP(
                    name='parallel_search',
                    url='https://search.parallel.ai/mcp',
                    headers={
                        'User-Agent': 'EvalScope-Parallel-Example/1.0 (https://github.com/modelscope/evalscope)',
                    },
                    tools=['web_search', 'web_fetch'],
                ),
            ],
            kwargs={
                'system_prompt': (
                    'Use web_search to research the question. Read web_fetch excerpts when search results '
                    'are insufficient, and cross-check important facts. Generate one UUID session_id for this '
                    'question and reuse it for all search and fetch calls. '
                    'When finished, call submit(answer=...) with only the final answer or complete answer set.'
                ),
            },
        ),
    )


if __name__ == '__main__':
    run_task(build_task_config())
