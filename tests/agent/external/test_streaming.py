"""SSE streaming round-trip test via the Anthropic SDK.

Points ``anthropic.AsyncAnthropic`` at the bridge with a trial token, asks
the bridge to stream a response from a :class:`MockLLM`-backed model, and
asserts the SDK parses the full event sequence (``message_start`` …
``message_stop``) and reassembles the original text.
"""

import anthropic
import pytest

from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.models.mockllm import MockLLM
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    """Tear down the calling thread's bridge loop after each test."""
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _build_mock_model(text: str) -> Model:
    api = MockLLM(
        model_name='mock-claude',
        custom_outputs=[ModelOutput.from_content(model='mock-claude', content=text)],
    )
    return Model(api=api, config=GenerateConfig())


def test_anthropic_streaming_round_trip():
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
