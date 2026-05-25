"""Docker-backed end-to-end tests for the external-agent path.

These exercise the parts of the bridge / runner stack that
``LocalAgentEnvironment`` cannot validate:

* ``host.docker.internal`` URL rewrite + ``extra_hosts`` injection on
  Linux — the bridge must be reachable from inside the container.
* The bridge actually serving across the host/container boundary on
  ``0.0.0.0`` rather than only on host loopback.
* Runtime install of Node + ``@anthropic-ai/claude-code`` in
  ``ClaudeCodeRunner.setup`` against a real apt-based image.
* ``extract_patch`` against a real container working tree.

All tests are opt-in via ``EVALSCOPE_DOCKER_E2E=1`` because they require
a running Docker daemon and pull external images. Tier C
(``test_swe_bench_pro_real_e2e``) additionally requires
``EVALSCOPE_REAL_CC=1`` + idealab credentials because it spends real
LLM tokens.
"""

import os
import pytest
import shutil

from evalscope.agent.external import ExternalAgentConfig
from evalscope.agent.external.adapter import run_external_agent
from evalscope.agent.external.helpers import extract_patch
from evalscope.api.agent import EventType
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.models.mockllm import MockLLM
from evalscope.utils.function_utils import AsyncioLoopRunner

# ---------------------------------------------------------------------------
# Opt-in gating
# ---------------------------------------------------------------------------

_REQUIRES_DOCKER = pytest.mark.skipif(
    os.environ.get('EVALSCOPE_DOCKER_E2E') != '1',
    reason='docker e2e test; set EVALSCOPE_DOCKER_E2E=1 to enable (requires Docker daemon)',
)


def _docker_available() -> bool:
    if shutil.which('docker') is None:
        return False
    import subprocess
    return subprocess.run(['docker', 'info'], capture_output=True).returncode == 0


_REQUIRES_DOCKER_DAEMON = pytest.mark.skipif(
    not _docker_available(),
    reason='docker daemon not reachable',
)


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _build_mock_model(text: str) -> Model:
    api = MockLLM(
        model_name='mock-model',
        # Many outputs so the same fixture handles retries / multi-turn calls.
        custom_outputs=[ModelOutput.from_content(model='mock-model', content=text) for _ in range(8)],
    )
    return Model(api=api, config=GenerateConfig())


# ---------------------------------------------------------------------------
# Tier A: extract_patch inside a real container
# ---------------------------------------------------------------------------


@_REQUIRES_DOCKER
@_REQUIRES_DOCKER_DAEMON
def test_extract_patch_inside_enclave(tmp_path):
    """Run ``extract_patch`` against a working tree inside a real Docker
    container. Pins the helper's ``cwd=`` plumbing through ms_enclave's
    shell_executor (different code path from LocalAgentEnvironment)."""
    from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

    async def _go() -> str:
        # No ``platform`` pin: both python:3.11-slim and ubuntu:22.04 are
        # multi-arch, so docker picks the host-native architecture
        # (matters for Apple Silicon hosts where arm64 is cached).
        env = EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config={
                'image': 'python:3.11-slim',
                'working_dir': '/workspace',
            },
            timeout=180.0,
        )
        async with env:
            # python:3.11-slim ships without git; install it once.
            install = await env.exec(
                ['bash', '-c', 'apt-get update -qq && apt-get install -y -qq git'],
                timeout=180.0,
            )
            assert install.returncode == 0, f'apt install git failed: {install.stderr!r}'

            setup = await env.exec(
                [
                    'bash', '-c',
                    'set -e; mkdir -p /workspace/repo && cd /workspace/repo && '
                    'git init -q -b main && '
                    'git config user.email t@e.com && git config user.name t && '
                    'printf "old\\n" > file.txt && '
                    'git add file.txt && git commit -q -m init && '
                    'printf "new\\n" > file.txt && '
                    'printf "fresh\\n" > untracked.txt'
                ],
                timeout=60.0,
            )
            assert setup.returncode == 0, f'repo setup failed: {setup.stderr!r}'

            return await extract_patch(env, cwd='/workspace/repo')

    patch = AsyncioLoopRunner.run(_go())
    assert 'diff --git' in patch, f'no diff returned: {patch!r}'
    assert '-old' in patch
    assert '+new' in patch
    # ``git add -A`` is the load-bearing part: untracked files must be in.
    assert 'untracked.txt' in patch
    assert '+fresh' in patch


# ---------------------------------------------------------------------------
# Tier B: claude-code → bridge → MockLLM, all crossing the host/container
# boundary. No real LLM cost.
# ---------------------------------------------------------------------------


@_REQUIRES_DOCKER
@_REQUIRES_DOCKER_DAEMON
def test_claude_code_through_docker_bridge_with_mock_llm(tmp_path):
    """Container → bridge → MockLLM round-trip.

    Validates: Node + claude-code runtime install via apt (covers the
    SWE-bench / SWE-bench_Pro install path), bridge reachable on
    ``host.docker.internal``, ``ANTHROPIC_BASE_URL`` end-to-end.

    This pins the wiring without spending LLM tokens — the MockLLM
    returns a fixed sentinel and we assert claude-code surfaces that
    same sentinel as its stdout.
    """
    from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

    expected = 'BRIDGE_OK'
    model = _build_mock_model(expected)
    sample = Sample(input='Reply with exactly: BRIDGE_OK', id=1)

    # ``node:20-slim`` already ships Node + npm, so the runner's
    # apt+nodesource detour is skipped (only ``npm install -g claude-code``
    # runs). Makes the test robust against partial in-container DNS
    # (apt mirrors are flaky on colima / VPN'd machines).
    env = EnclaveAgentEnvironment(
        engine='docker',
        sandbox_config={
            'image': 'node:20-slim',
            'working_dir': '/workspace',
            'network_enabled': True,
        },
        timeout=180.0,
    )

    config = ExternalAgentConfig(
        framework='claude-code',
        kwargs={
            'model_name': 'mock-model',
            'auto_install': True,
            # NOTE: do NOT pass ``allowed_tools=''`` — ``--allowedTools``
            # is variadic and would swallow the trailing prompt
            # (landmine #3 in the agent_bridge_design plan).
            # Inherit the inside-container HOME (no host keychain to evade).
            'home_override': '',
        },
        # AgentLoopAdapter normally builds the env; here we wire it
        # directly because we are not exercising a benchmark adapter.
        environment='enclave',
        timeout=300.0,
    )

    result: InferenceResult = run_external_agent(
        config=config,
        model=model,
        sample=sample,
        environment_override=env,
    )

    text = (result.output.message.text or '').strip()
    assert expected in text, (
        f'expected mock-LLM sentinel {expected!r} to surface as agent output; '
        f'got {text!r}'
    )
    assert result.trace is not None
    assert result.trace.framework == 'claude-code'
    types = [ev.type for ev in result.trace.events]
    assert EventType.RUN_START in types
    assert EventType.RUN_END in types
    assert EventType.MODEL_GENERATE in types


# ---------------------------------------------------------------------------
# Tier C: full SWE-bench_Pro 1-sample real-LLM round-trip
# ---------------------------------------------------------------------------


def _load_env_file() -> None:
    """Best-effort ``.env`` load so opt-in tests pick up local secrets."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


_load_env_file()


_REQUIRES_REAL_LLM = pytest.mark.skipif(
    os.environ.get('EVALSCOPE_REAL_CC') != '1' or not os.environ.get('EVALSCOPE_IDEALAB_TOKEN'),
    reason='real-LLM SWE-bench_Pro test; set EVALSCOPE_REAL_CC=1 + EVALSCOPE_IDEALAB_TOKEN',
)


@_REQUIRES_DOCKER
@_REQUIRES_DOCKER_DAEMON
@_REQUIRES_REAL_LLM
def test_swe_bench_pro_real_e2e(tmp_path, monkeypatch):
    """1-sample SWE-bench_Pro through external claude-code.

    This is the slowest / costliest test in the suite: pulls the
    per-instance ``sweap-images:*`` (~5–15 GB), installs claude-code,
    runs claude against a real LLM, then runs the SWE-bench Pro
    evaluation container. Counts as manual validation only — not
    intended for CI.

    We assert the path *executes* end-to-end (no exception, scored
    report flowing back); we deliberately do NOT assert ``resolved=True``
    because that is model-quality dependent.
    """
    from evalscope.config import TaskConfig
    from evalscope.run import run_task

    idealab_url = os.environ.get('EVALSCOPE_IDEALAB_BASE_URL', 'https://idealab.alibaba-inc.com/api/anthropic')
    idealab_token = os.environ['EVALSCOPE_IDEALAB_TOKEN']
    target_model = os.environ.get('EVALSCOPE_IDEALAB_MODEL', 'claude-opus-4-6')

    # Idealab rejects requests carrying both ``x-api-key`` and
    # ``Authorization``; the Anthropic SDK auto-adds Authorization from
    # ``ANTHROPIC_AUTH_TOKEN`` whenever it is present. Strip both so the
    # SDK only sends ``x-api-key`` (set via our explicit ``api_key`` kwarg).
    monkeypatch.delenv('ANTHROPIC_AUTH_TOKEN', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)

    task_cfg = TaskConfig(
        model=target_model,
        api_url=idealab_url,
        api_key=idealab_token,
        eval_type='anthropic_api',
        # Idealab User-Agent whitelist: only claude-cli traffic passes.
        model_args={'default_headers': {'User-Agent': 'claude-cli/2.1.146 (external, cli)'}},
        datasets=['swe_bench_pro'],
        agent_config={
            'mode': 'external',
            'framework': 'claude-code',
            'environment': 'enclave',
            'timeout': 1200.0,
            'kwargs': {
                'model_name': target_model,
                'auto_install': True,
            },
        },
        eval_batch_size=1,
        limit=1,
        analysis_report=False,
        work_dir=str(tmp_path),
    )

    result = run_task(task_cfg=task_cfg)
    assert isinstance(result, dict) and 'swe_bench_pro' in result, (
        f'expected swe_bench_pro report in {result!r}'
    )
    from evalscope.report.report import Report
    report = result['swe_bench_pro']
    assert isinstance(report, Report)
