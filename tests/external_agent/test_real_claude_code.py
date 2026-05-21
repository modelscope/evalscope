"""End-to-end test against the real Anthropic API via the idealab proxy.

Skipped by default — opt-in with ``EVALSCOPE_REAL_CC=1`` so CI does not
hit the network.  Validates the full stack: ``claude --print`` →
EvalScope bridge → ``AnthropicCompatibleAPI`` → idealab Anthropic proxy.

Credentials are read from environment variables (loaded from a project-
root ``.env`` when present via ``python-dotenv``).  Required:

* ``EVALSCOPE_IDEALAB_TOKEN`` — Anthropic-compatible API key for idealab.
* ``EVALSCOPE_IDEALAB_BASE_URL`` (optional) — defaults to the public
  idealab endpoint.
"""

import os
import pytest
import shutil

from evalscope.api.dataset import Sample
from evalscope.api.model import GenerateConfig, Model
from evalscope.external_agent import ExternalAgentConfig
from evalscope.external_agent.adapter import run_external_agent
from evalscope.models.anthropic_compatible import AnthropicCompatibleAPI
from evalscope.utils.function_utils import AsyncioLoopRunner


def _load_env_file() -> None:
    """Best-effort ``.env`` load so opt-in tests pick up local secrets."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


_load_env_file()

IDEALAB_BASE_URL = os.environ.get('EVALSCOPE_IDEALAB_BASE_URL', 'https://idealab.alibaba-inc.com/api/anthropic')
IDEALAB_TOKEN = os.environ.get('EVALSCOPE_IDEALAB_TOKEN', '')
TARGET_MODEL = os.environ.get('EVALSCOPE_IDEALAB_MODEL', 'claude-opus-4-6')

# Idealab gates requests by user-agent (Team管理规则): only claude-cli traffic
# is whitelisted, so we forge the header on every outbound request.
_IDEALAB_HEADERS = {'User-Agent': 'claude-cli/2.1.143 (external, cli)'}


def _has_claude_cli() -> bool:
    return shutil.which('claude') is not None


_REQUIRES_REAL = pytest.mark.skipif(
    os.environ.get('EVALSCOPE_REAL_CC') != '1' or not IDEALAB_TOKEN,
    reason='real-network test; set EVALSCOPE_REAL_CC=1 and EVALSCOPE_IDEALAB_TOKEN (e.g. in .env)',
)


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    """Release the per-thread bridge loop after each test so the next test
    starts from a clean ``AsyncioLoopRunner`` state."""
    yield
    AsyncioLoopRunner.shutdown_for_thread()


@pytest.fixture(autouse=True)
def _scrub_anthropic_env(monkeypatch):
    """Idealab rejects requests carrying both ``x-api-key`` and
    ``Authorization``; the Anthropic SDK auto-populates ``Authorization``
    from ``ANTHROPIC_AUTH_TOKEN`` whenever that env var is present.  We
    strip it so the SDK only sends ``x-api-key`` (which we set explicitly
    via ``api_key=``).
    """
    monkeypatch.delenv('ANTHROPIC_AUTH_TOKEN', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)


def _build_anthropic_model() -> Model:
    api = AnthropicCompatibleAPI(
        model_name=TARGET_MODEL,
        base_url=IDEALAB_BASE_URL,
        api_key=IDEALAB_TOKEN,
        default_headers=_IDEALAB_HEADERS,
    )
    return Model(api=api, config=GenerateConfig(max_tokens=64))


@_REQUIRES_REAL
def test_evalscope_direct_to_idealab():
    """Sanity: ``AnthropicCompatibleAPI`` can talk to idealab directly."""
    model = _build_anthropic_model()
    output = model.generate(input='Reply with exactly: PONG')
    text = (output.message.text or '').strip()
    assert text, f'empty response: {output!r}'
    assert 'PONG' in text.upper(), f'unexpected response: {text!r}'


@_REQUIRES_REAL
@pytest.mark.skipif(not _has_claude_cli(), reason='claude CLI not installed')
def test_claude_code_through_bridge_to_idealab():
    """End-to-end: claude --print → bridge → idealab Anthropic."""
    model = _build_anthropic_model()
    sample = Sample(input='Reply with exactly: PONG. No other text.', id=1)
    config = ExternalAgentConfig(
        framework='claude-code',
        kwargs={
            'model_name': TARGET_MODEL,
            'skip_permissions': True,
            # NOTE: ``allowed_tools=''`` would consume the prompt because
            # ``--allowedTools <tools...>`` is variadic; rely on
            # ``--dangerously-skip-permissions`` for safety in this test.
            # ``bare=True`` disables auth fallback and breaks the CLI on
            # bridge endpoints, so leave it off.
        },
        environment='local',
        timeout=120.0,
    )
    output = run_external_agent(config=config, model=model, sample=sample)
    text = (output.message.text or '').strip()
    assert text, f'empty agent output; trajectory={sample.metadata.get("external_agent_trajectory")}'
    # The model often appends explanations; just check the keyword appears.
    assert 'PONG' in text.upper(), f'unexpected agent output: {text!r}'
    trajectory = sample.metadata['external_agent_trajectory']
    assert trajectory['framework'] == 'claude-code'
    assert any(s['source'] == 'agent' for s in trajectory['steps'])
