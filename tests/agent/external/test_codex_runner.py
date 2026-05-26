"""End-to-end: codex exec → bridge (Responses API) → bridge translation
→ DashScope qwen3-max chat completions.

Skipped by default — opt-in with ``EVALSCOPE_REAL_QWEN=1`` plus a valid
``DASHSCOPE_API_KEY`` and a locally installed ``codex`` CLI so CI never
depends on the network or external binaries.

This is the PR2 acceptance test that proves the full chain works:
codex speaks Responses to the bridge; the bridge translates to
``ChatMessage[]``, calls :class:`OpenAICompatibleAPI` against DashScope's
chat-completions endpoint, then synthesizes a Responses SSE stream back
to codex. Verifies model output reaches the answer file and the trace
records ``framework='codex'`` with ≥1 MODEL_GENERATE event.

Note: qwen3-max does not return ``reasoning_content`` by default, so
this test does not verify reasoning end-to-end. The mock suite already
covers reasoning translation.
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
from evalscope.utils.function_utils import AsyncioLoopRunner


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
TARGET_MODEL = os.environ.get('EVALSCOPE_QWEN_MODEL', 'qwen3-max')


def _has_codex_cli() -> bool:
    return shutil.which('codex') is not None


_REQUIRES_REAL = pytest.mark.skipif(
    os.environ.get('EVALSCOPE_REAL_QWEN') != '1' or not DASHSCOPE_API_KEY,
    reason='real-network test; set EVALSCOPE_REAL_QWEN=1 and DASHSCOPE_API_KEY (e.g. in .env)',
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


@_REQUIRES_REAL
@pytest.mark.skipif(not _has_codex_cli(), reason='codex CLI not installed')
def test_codex_exec_through_bridge_responses_to_qwen3_max():
    """End-to-end: codex exec → bridge /openai/v1/responses → qwen3-max."""
    model = _build_qwen_model()
    sample = Sample(input='What is 6 * 7? Reply with just the number.', target='42', id=1)
    config = ExternalAgentConfig(
        framework='codex',
        # Defaults cover everything: sandbox=workspace-write hardcoded,
        # non-interactive always on, model_name auto-inherited from
        # the Model passed to run_external_agent.
        kwargs={},
        environment='local',
        timeout=180.0,
    )
    result = run_external_agent(config=config, model=model, sample=sample)
    text = (result.output.message.text or '').strip()
    assert text, f'empty agent output; trace_events={[e.type for e in result.trace.events]}'
    assert '42' in text, f'unexpected agent output: {text!r}'
    trace = result.trace
    assert trace.framework == 'codex'
    assert any(ev.type == EventType.MODEL_GENERATE for ev in trace.events)
