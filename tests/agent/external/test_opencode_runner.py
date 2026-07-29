"""End-to-end: opencode run → bridge (OpenAI Responses API) → DashScope qwen-plus.

Skipped by default — opt-in with ``EVALSCOPE_REAL_QWEN=1`` plus a valid
``DASHSCOPE_API_KEY`` so CI never depends on the network.

Covers two tiers:

* **Tier A (local)**: requires ``opencode`` CLI installed on the host
  (``npm install -g opencode-ai``). Exercises ``LocalAgentEnvironment``
  + bridge round-trip without Docker overhead.
* **Tier B (Docker)**: requires Docker daemon + pre-built
  ``evalscope-opencode:latest`` image. Exercises the full
  ``EnclaveAgentEnvironment`` path — container → bridge →
  DashScope → trajectory capture.

OpenCode speaks the OpenAI **Responses API** (``/openai/v1/responses``),
not Chat Completions.  The bridge's Responses route translates the request
to ``ChatMessage[]`` for the upstream model, so this test proves the
complete cross-protocol chain:
``opencode (Responses API) → bridge → DashScope (OpenAI compat) → qwen-plus``.
"""

import os
import pytest
import shutil

from evalscope.agent.external import ExternalAgentConfig
from evalscope.agent.external.adapter import run_external_agent
from evalscope.api.agent import EventType
from evalscope.api.dataset import Sample
from evalscope.api.model import GenerateConfig, Model
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


_load_env_file()

DASHSCOPE_BASE_URL = os.environ.get(
    'DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'
)
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
TARGET_MODEL = os.environ.get('EVALSCOPE_QWEN_MODEL', 'qwen-plus')


def _has_opencode_cli() -> bool:
    return shutil.which('opencode') is not None


def _docker_available() -> bool:
    if shutil.which('docker') is None:
        return False
    import subprocess
    return subprocess.run(['docker', 'info'], capture_output=True).returncode == 0


def _image_exists(name: str) -> bool:
    import subprocess
    result = subprocess.run(
        ['docker', 'images', '-q', name],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


_REQUIRES_REAL = pytest.mark.skipif(
    os.environ.get('EVALSCOPE_REAL_QWEN') != '1' or not DASHSCOPE_API_KEY,
    reason='real-network test; set EVALSCOPE_REAL_QWEN=1 and DASHSCOPE_API_KEY (e.g. in .env)',
)

_REQUIRES_DOCKER = pytest.mark.skipif(
    os.environ.get('EVALSCOPE_DOCKER_E2E') != '1',
    reason='docker e2e test; set EVALSCOPE_DOCKER_E2E=1 to enable',
)

_REQUIRES_DOCKER_DAEMON = pytest.mark.skipif(
    not _docker_available(),
    reason='docker daemon not reachable',
)


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _build_qwen_model() -> Model:
    api = OpenAICompatibleAPI(
        model_name=TARGET_MODEL,
        base_url=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
    )
    return Model(api=api, config=GenerateConfig(max_tokens=256))


# ---------------------------------------------------------------------------
# Tier A: local opencode CLI → bridge → DashScope
# ---------------------------------------------------------------------------


@_REQUIRES_REAL
@pytest.mark.skipif(not _has_opencode_cli(), reason='opencode CLI not installed')
def test_opencode_local_through_bridge_to_qwen():
    """End-to-end: opencode run → bridge /openai/v1/chat/completions → qwen-plus."""
    model = _build_qwen_model()
    sample = Sample(input='What is 6 * 7? Reply with just the number.', target='42', id=1)
    config = ExternalAgentConfig(
        framework='opencode',
        kwargs={
            'model_name': TARGET_MODEL,
            'auto_install': False,
            'home_override': '',
        },
        environment='local',
        timeout=180.0,
    )
    result = run_external_agent(config=config, model=model, sample=sample)
    text = (result.output.message.text or '').strip()
    assert text, f'empty agent output; trace_events={[e.type for e in result.trace.events]}'
    assert '42' in text, f'unexpected agent output: {text!r}'
    trace = result.trace
    assert trace.framework == 'opencode'
    assert any(ev.type == EventType.MODEL_GENERATE for ev in trace.events)


# ---------------------------------------------------------------------------
# Tier B: Docker (evalscope-opencode:latest) → bridge → DashScope
# ---------------------------------------------------------------------------


@_REQUIRES_REAL
@_REQUIRES_DOCKER
@_REQUIRES_DOCKER_DAEMON
@pytest.mark.skipif(
    not _image_exists('evalscope-opencode:latest'),
    reason='evalscope-opencode:latest image not built; run: '
           'docker build -f evalscope/agent/external/dockerfiles/Dockerfile.opencode '
           '-t evalscope-opencode:latest .',
)
def test_opencode_docker_through_bridge_to_qwen():
    """Docker container (pre-built image) → bridge → qwen-plus via DashScope.

    Validates the full production path: container sandbox → bridge →
    DashScope → trajectory capture, with a real LLM response.
    """
    from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

    model = _build_qwen_model()
    sample = Sample(input='What is 6 * 7? Reply with just the number.', target='42', id=1)

    env = EnclaveAgentEnvironment(
        engine='docker',
        sandbox_config={
            'image': 'evalscope-opencode:latest',
            'working_dir': '/workspace',
            'network_enabled': True,
        },
        timeout=300.0,
    )

    config = ExternalAgentConfig(
        framework='opencode',
        kwargs={
            'model_name': TARGET_MODEL,
            'auto_install': False,
            'home_override': '',
        },
        environment='enclave',
        timeout=180.0,
    )

    result = run_external_agent(
        config=config,
        model=model,
        sample=sample,
        environment_override=env,
    )

    text = (result.output.message.text or '').strip()
    assert text, f'empty agent output; trace_events={[e.type for e in result.trace.events]}'
    assert '42' in text, f'unexpected agent output: {text!r}'

    # Trace validation
    trace = result.trace
    assert trace.framework == 'opencode'
    assert trace.environment == 'enclave'
    assert any(ev.type == EventType.MODEL_GENERATE for ev in trace.events)
    assert any(ev.type == EventType.RUN_START for ev in trace.events)
    assert any(ev.type == EventType.RUN_END for ev in trace.events)

    # Messages validation: bridge should have captured the conversation
    assert result.messages, 'no messages captured by bridge'
    roles = [m.role for m in result.messages]
    assert 'assistant' in roles, f'no assistant message in {roles}'
