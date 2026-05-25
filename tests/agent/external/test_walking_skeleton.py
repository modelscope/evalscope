"""End-to-end walking-skeleton tests for the external-agent bridge.

Builds a real aiohttp bridge on a random port, points the embedded
:class:`MockAgentRunner` at it via ``LocalAgentEnvironment``, and asserts
the round-trip (agent → bridge → MockLLM → bridge → agent → adapter)
produces the expected ``InferenceResult`` and a populated
:class:`AgentTrace`.

Also covers:
* Config-level validation (unknown framework name → fail-fast)
* Auth rejection (unknown trial token → 401)
* Anthropic SDK streaming round-trip — verifies the synthesized SSE
  event sequence is parseable by the official ``anthropic`` client.
"""

import anthropic
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


def test_anthropic_sdk_streaming_round_trip():
    """Synthesized Anthropic SSE events must round-trip through the official
    ``anthropic.AsyncAnthropic`` client: full event sequence parses cleanly
    and text deltas reassemble to the original assistant content."""
    expected = 'streamed answer: 42' * 3  # spans multiple text_delta chunks

    async def _go() -> tuple[set, str]:
        model = _build_mock_model(expected)
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            client = anthropic.AsyncAnthropic(
                base_url=f'{proxy.base_url}/anthropic',
                auth_token=session.token,
            )
            event_types: set = set()
            collected_text = ''
            async with client.messages.stream(
                model='mock-claude',
                max_tokens=256,
                messages=[{'role': 'user', 'content': 'streaming?'}],
            ) as stream:
                async for event in stream:
                    etype = getattr(event, 'type', None)
                    if etype:
                        event_types.add(etype)
                    if etype == 'content_block_delta' and hasattr(event.delta, 'text'):
                        collected_text += event.delta.text
            await client.close()
            return event_types, collected_text

    event_types, collected_text = AsyncioLoopRunner.run(_go())
    for expected_event in (
        'message_start',
        'content_block_start',
        'content_block_delta',
        'content_block_stop',
        'message_delta',
        'message_stop',
    ):
        assert expected_event in event_types, f'missing SSE event: {expected_event}'
    assert collected_text == expected
