"""One-shot: GSM8K × 1 sample × codex (external agent) × qwen3-max.

Triggers the codex CLI inside a local environment, pointed at the
evalscope bridge's Responses API route, with DashScope qwen3-max as the
downstream model. The bridge translates Responses input[] →
ChatMessage[] → DashScope chat completions and synthesizes the
Responses SSE event stream back.

Lighter than the SWE-bench Pro variant: runs in ``local`` environment
(no container pull), tens of seconds wall, ¥0.01-0.1 in DashScope
tokens. Requires ``codex`` CLI installed on PATH and
``DASHSCOPE_API_KEY`` in the environment (or ``.env``).
"""

import os
import shutil
import sys
import tempfile
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    if shutil.which('codex') is None:
        print('FAIL: codex CLI not on PATH (install via `npm install -g @openai/codex`)', file=sys.stderr)
        return 2
    base_url = os.environ.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print('FAIL: DASHSCOPE_API_KEY missing', file=sys.stderr)
        return 2
    target_model = os.environ.get('EVALSCOPE_QWEN_MODEL', 'qwen3-max')
    dataset = os.environ.get('EVALSCOPE_CODEX_DATASET', 'gsm8k')

    from evalscope.config import TaskConfig
    from evalscope.run import run_task

    work_dir = tempfile.mkdtemp(prefix='codex_qwen_')
    print(f'work_dir = {work_dir}', flush=True)
    print(f'model    = {target_model}', flush=True)
    print(f'dataset  = {dataset}', flush=True)
    print(f'base_url = {base_url}', flush=True)

    task_cfg = TaskConfig(
        model=target_model,
        api_url=base_url,
        api_key=api_key,
        eval_type='openai_api',
        datasets=[dataset],
        agent_config={
            'mode': 'external',
            'framework': 'codex',
            'environment': 'local',
            'timeout': 300.0,
            'kwargs': {},
        },
        eval_batch_size=1,
        limit=1,
        analysis_report=False,
        work_dir=work_dir,
    )

    result = run_task(task_cfg=task_cfg)
    print(f'\n=== RESULT ===\n{result!r}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
