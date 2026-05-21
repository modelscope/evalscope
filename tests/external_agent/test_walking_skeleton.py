"""End-to-end walking-skeleton test for the external-agent bridge.

Builds a real aiohttp bridge on a random port, points the embedded
:class:`MockAgentRunner` at it via ``LocalAgentEnvironment``, and asserts
the round-trip (agent → bridge → MockLLM → bridge → agent → adapter)
produces the expected ``ModelOutput`` and a populated trajectory.
"""

import asyncio
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.external_agent import ExternalAgentConfig
from evalscope.external_agent.adapter import run_external_agent
from evalscope.external_agent.bridge import ModelProxyServer
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
    output = run_external_agent(config=config, model=model, sample=sample)

    assert isinstance(output, ModelOutput)
    assert output.choices, 'expected at least one choice'
    assert output.choices[0].message.text == expected

    trajectory = sample.metadata.get('external_agent_trajectory')
    assert trajectory is not None
    assert trajectory['framework'] == 'mock'
    assert trajectory['model'] == 'mock-claude'
    # The mock makes exactly one LLM call → one agent step (no tool step
    # because no tool_result observation precedes it).
    agent_steps = [s for s in trajectory['steps'] if s['source'] == 'agent']
    assert len(agent_steps) == 1
    assert agent_steps[0]['message'] == expected


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
