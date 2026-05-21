"""End-to-end walking-skeleton test for the external-agent bridge.

Builds a real aiohttp bridge on a random port, points the embedded
:class:`MockAgentRunner` at it via ``LocalAgentEnvironment``, and asserts
the round-trip (agent → bridge → MockLLM → bridge → agent → adapter)
produces the expected ``InferenceResult`` and a populated
:class:`AgentTrace`.
"""

import asyncio
import pytest

from evalscope.agent.external import ExternalAgentConfig
from evalscope.agent.external.adapter import run_external_agent
from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.agent import AgentTrace, EventType
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.models.mockllm import MockLLM
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    """Tear down the calling thread's bridge loop after each test.

    The bridge's :meth:`shutdown` is registered as an ``AsyncioLoopRunner``
    close-callback, so shutting the loop also releases the aiohttp port.
    """
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _build_mock_model(text: str) -> Model:
    api = MockLLM(
        model_name='mock-claude',
        custom_outputs=[ModelOutput.from_content(model='mock-claude', content=text)],
    )
    return Model(api=api, config=GenerateConfig())


def test_walking_skeleton_round_trip():
    expected = 'the answer is 42'
    model = _build_mock_model(expected)
    sample = Sample(input='what is 6 * 7?', target='42', id=1)

    config = ExternalAgentConfig(framework='mock', environment='local')
    result = run_external_agent(config=config, model=model, sample=sample)

    assert isinstance(result, InferenceResult)
    assert isinstance(result.output, ModelOutput)
    assert result.output.choices, 'expected at least one choice'
    assert result.output.choices[0].message.text == expected

    trace = result.trace
    assert isinstance(trace, AgentTrace)
    assert trace.framework == 'mock'
    assert trace.trial_id, 'trial_id should be set by the bridge'
    assert trace.environment == 'local'
    # The mock makes exactly one LLM call → exactly one MODEL_GENERATE event
    # plus RUN_START / RUN_END brackets.
    types = [ev.type for ev in trace.events]
    assert EventType.RUN_START in types
    assert EventType.MODEL_GENERATE in types
    assert EventType.RUN_END in types
    assert types.count(EventType.MODEL_GENERATE) == 1
    # Messages reconstructed from the bridge transcript: user prompt + 1 assistant.
    roles = [m.role for m in (result.messages or [])]
    assert roles == ['user', 'assistant']
    assert result.messages[1].text == expected


def test_unknown_framework_raises():
    """Validator rejects unregistered framework names at construction time
    (fail-fast, before any bridge or env setup runs)."""
    with pytest.raises(Exception, match='does-not-exist'):
        ExternalAgentConfig(framework='does-not-exist')


def test_bridge_rejects_unknown_trial_token(tmp_path):
    """The bridge must 401 requests bearing an unknown trial token."""
    import urllib.error
    import urllib.request

    async def _go() -> int:
        proxy = await ModelProxyServer.get_or_start(host='127.0.0.1', port=None)
        url = f'{proxy.base_url}/anthropic/v1/messages'

        def _request() -> int:
            req = urllib.request.Request(
                url,
                data=b'{"model":"x","max_tokens":1,"messages":[]}',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer trial-bogus',
                },
                method='POST',
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status
            except urllib.error.HTTPError as exc:
                return exc.code

        return await asyncio.get_running_loop().run_in_executor(None, _request)

    status = AsyncioLoopRunner.run(_go())
    assert status == 401
