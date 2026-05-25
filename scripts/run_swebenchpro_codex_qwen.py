"""One-shot: SWE-bench_Pro × 1 sample × codex (external agent) × qwen3-max.

Mirrors ``run_swebenchpro_qwen_cc.py`` (PR1) but swaps the agent
framework from ``claude-code`` to ``codex``. Both share the same
SWE-bench Pro container pull / colima-enclave plumbing — only the
runner + bridge route differ:

* claude-code → ``POST /anthropic/v1/messages``
* codex      → ``POST /openai/v1/responses``  (codex v0.133+ forces it)

Heavy: pulls per-instance ``jefzda/sweap-images:*`` (5-15 GB), installs
node + codex inside that container, drives a multi-turn agent loop
against real DashScope, then runs the per-instance test suite. Budget:
30-60 min walltime; ¥0.5-5 in DashScope tokens for one sample.

Known risk: ``npm install -g @openai/codex`` may stall inside the
sandbox if the container's egress goes through the same slow CDN path
that hung the host install. If you see ``CodexRunner.setup`` time out
after 300s, set ``install_codex=False`` and bake codex into the
sweap-image yourself, or wait for the GH-binary fallback to land in
``CodexRunner._install_codex_cli``.
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

    work_dir = tempfile.mkdtemp(prefix='swebpro_codex_qwen_')
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
            'framework': 'codex',
            'environment': 'enclave',
            'timeout': 1800.0,
            'kwargs': {
                'model_name': target_model,
                'install_codex': True,
                # codex sandbox is its own thing — workspace-write lets the
                # agent edit the SWE-bench Pro repo files in /app.
                'sandbox': 'workspace-write',
                'yolo': True,
                # codex writes its final assistant message here; we read it
                # back to populate AgentRunResult.output. The path must be
                # writable in codex's workspace-write sandbox.
                'output_last_message_path': '/tmp/evalscope-codex-last.txt',
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
