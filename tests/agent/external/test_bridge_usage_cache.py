"""Cached input tokens must be reported in the bridge client's own convention.

OpenAI-style providers count cached tokens inside ``input_tokens``; Anthropic
reports them separately. The bridge has to convert between the two.
"""

import asyncio
import json

import pytest

from evalscope.agent.external.bridge.sse_anthropic import stream_anthropic_response
from evalscope.agent.external.bridge.sse_openai import stream_openai_response
from evalscope.agent.external.bridge.translate_anthropic import anthropic_usage_payload
from evalscope.agent.external.bridge.translate_gemini import model_output_to_gemini_response
from evalscope.agent.external.bridge.translate_openai import model_output_to_openai_response
from evalscope.agent.external.bridge.translate_responses import model_output_to_responses_payload
from evalscope.api.model import ModelOutput, ModelUsage

# 1000 prompt tokens, 800 read from cache (and, for the last case, 100 written).
OPENAI_STYLE = ModelUsage(input_tokens=1000, output_tokens=50, total_tokens=1050, input_tokens_cache_read=800)
ANTHROPIC_STYLE = ModelUsage(input_tokens=200, output_tokens=50, total_tokens=1050, input_tokens_cache_read=800)
ANTHROPIC_STYLE_WRITE = ModelUsage(
    input_tokens=100, output_tokens=50, total_tokens=1050, input_tokens_cache_read=800, input_tokens_cache_write=100
)
NO_CACHE = ModelUsage(input_tokens=1000, output_tokens=50, total_tokens=1050)
# Some OpenAI-compatible providers omit completion_tokens.
NO_COMPLETION_COUNT = ModelUsage(input_tokens=1000, output_tokens=0, total_tokens=1200, input_tokens_cache_read=800)


@pytest.mark.parametrize(
    'usage, uncached',
    [(OPENAI_STYLE, 200), (ANTHROPIC_STYLE, 200), (ANTHROPIC_STYLE_WRITE, 100), (NO_CACHE, 1000)],
)
def test_anthropic_client_gets_uncached_input_tokens(usage: ModelUsage, uncached: int) -> None:
    payload = anthropic_usage_payload(usage)

    assert payload['input_tokens'] == uncached
    assert payload.get('cache_read_input_tokens') == usage.input_tokens_cache_read
    assert payload.get('cache_creation_input_tokens') == usage.input_tokens_cache_write


@pytest.mark.parametrize(
    'usage', [OPENAI_STYLE, ANTHROPIC_STYLE, ANTHROPIC_STYLE_WRITE, NO_CACHE, NO_COMPLETION_COUNT]
)
def test_openai_and_gemini_clients_get_full_prompt_tokens(usage: ModelUsage) -> None:
    output = ModelOutput.from_content('mock', 'hi')
    output.usage = usage

    chat = model_output_to_openai_response(output, request_model='mock')['usage']
    responses = model_output_to_responses_payload(output, request_model='mock')['usage']
    gemini = model_output_to_gemini_response(output, request_model='mock')['usageMetadata']

    assert chat['prompt_tokens'] == 1000
    assert responses['input_tokens'] == 1000
    assert gemini['promptTokenCount'] == 1000


def test_streaming_usage_follows_the_client_convention() -> None:
    async def collect(stream, usage: ModelUsage, **kwargs) -> list:
        output = ModelOutput.from_content('mock', 'hi')
        output.usage = usage

        async def generate() -> ModelOutput:
            return output

        task = asyncio.ensure_future(generate())
        frames = b''.join([frame async for frame in stream(task, request_model='mock', **kwargs)]).decode()
        data = [line[6:] for line in frames.splitlines() if line.startswith('data: {')]
        return [json.loads(item) for item in data]

    chat = asyncio.run(collect(stream_openai_response, ANTHROPIC_STYLE, include_usage=True))
    anthropic = asyncio.run(collect(stream_anthropic_response, OPENAI_STYLE))

    assert chat[-1]['usage']['prompt_tokens'] == 1000
    delta = next(event for event in anthropic if event.get('type') == 'message_delta')
    assert delta['usage']['input_tokens'] == 200
