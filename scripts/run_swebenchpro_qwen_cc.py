"""One-shot: SWE-bench_Pro × 1 sample × claude-code (external agent) × qwen3-max.

Drives the same path as ``test_swe_bench_pro_real_e2e`` but swaps the
backend model from claude/idealab to qwen3-max via DashScope's
OpenAI-compatible chat-completions endpoint. The bridge translates
claude-code's Anthropic protocol to ChatMessage and dispatches to
``OpenAICompatibleAPI(qwen3-max)``.

Heavy: pulls per-instance ``jefzda/sweap-images:*`` (5-15 GB), installs
node + claude-code inside that container, runs a multi-turn agent loop
against real DashScope, then runs the per-instance test suite. Budget:
10-30 min walltime, $0.10-$1 in DashScope tokens for one sample.
"""

import os
import sys
import tempfile
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    base_url = os.environ.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print('FAIL: DASHSCOPE_API_KEY missing', file=sys.stderr)
        return 2
    target_model = os.environ.get('EVALSCOPE_QWEN_MODEL', 'qwen3-max')

    from evalscope.config import TaskConfig
    from evalscope.run import run_task

    work_dir = tempfile.mkdtemp(prefix='swebpro_qwen_cc_')
    print(f'work_dir = {work_dir}', flush=True)
    print(f'model    = {target_model}', flush=True)
    print(f'base_url = {base_url}', flush=True)

    task_cfg = TaskConfig(
        model=target_model,
        api_url=base_url,
        api_key=api_key,
        eval_type='openai_api',
        datasets=['swe_bench_pro'],
        agent_config={
            'mode': 'external',
            'framework': 'claude-code',
            'environment': 'enclave',
            'timeout': 1800.0,
            'kwargs': {
                'model_name': target_model,
                'install_node': True,
            },
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
