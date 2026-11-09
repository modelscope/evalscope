"""Streaming and non-streaming replies retain the same token probabilities."""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.models.utils.openai import async_collect_stream_response, collect_stream_response


def _token(text: str, logprob: float, alternative: str) -> dict[str, Any]:
    return {
        'token': text,
        'logprob': logprob,
        'bytes': list(text.encode('utf-8')),
        'top_logprobs': [
            {'token': text, 'logprob': logprob, 'bytes': list(text.encode('utf-8'))},
            {'token': alternative, 'logprob': -3.0, 'bytes': list(alternative.encode('utf-8'))},
        ],
    }


TOKENS = [
    [_token('caf', -0.1, 'tea'), _token('é', -0.2, 'e')],
    [_token('blue', -0.3, 'red'), _token(' sky', -0.4, ' sea')],
]


@pytest.fixture
def provider_url() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            include_logprobs = request.get('logprobs', False)
            common = {'id': 'completion-id', 'created': 1, 'model': 'test-model'}
            usage = {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7}
            if request.get('stream'):
                chunks = []
                # Different arrival orders exercise both per-choice and per-token ordering.
                for position, indices in enumerate([(1, 0), (0, 1)]):
                    chunks.append({
                        **common,
                        'object': 'chat.completion.chunk',
                        'choices': [
                            {
                                'index': index,
                                'delta': {'content': TOKENS[index][position]['token']},
                                'finish_reason': None,
                                'logprobs': {'content': [TOKENS[index][position]]} if include_logprobs else None,
                            }
                            for index in indices
                        ],
                    })
                chunks.extend([
                    {
                        **common,
                        'object': 'chat.completion.chunk',
                        'choices': [
                            {'index': index, 'delta': {}, 'finish_reason': 'stop', 'logprobs': None}
                            for index in (1, 0)
                        ],
                    },
                    {**common, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage},
                ])
                body = (''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks) + 'data: [DONE]\n\n').encode()
                content_type = 'text/event-stream'
            else:
                body = json.dumps({
                    **common,
                    'object': 'chat.completion',
                    'choices': [
                        {
                            'index': index,
                            'finish_reason': 'stop',
                            'message': {'role': 'assistant', 'content': ''.join(t['token'] for t in TOKENS[index])},
                            'logprobs': {'content': TOKENS[index]} if include_logprobs else None,
                        }
                        for index in (1, 0)
                    ],
                    'usage': usage,
                }).encode()
                content_type = 'application/json'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('stream', [False, True], ids=['json', 'sse'])
@pytest.mark.parametrize('logprobs', [False, True], ids=['without-logprobs', 'with-logprobs'])
def test_token_probabilities(
    provider_url: tuple[str, list[dict[str, Any]]], asynchronous: bool, stream: bool, logprobs: bool,
) -> None:
    base_url, requests = provider_url
    api = OpenAICompatibleAPI(model_name='test-model', base_url=base_url, api_key='EMPTY', max_retries=0)
    config = GenerateConfig(stream=stream, logprobs=logprobs, top_logprobs=2 if logprobs else None, n=2, retries=0)
    messages = [ChatMessageUser(content='answer')]

    async def generate_async() -> ModelOutput:
        try:
            return await api.generate_async(messages, [], 'none', config)
        finally:
            await api.aclose()

    try:
        output = asyncio.run(generate_async()) if asynchronous else api.generate(messages, [], 'none', config)
    finally:
        api.client.close()

    assert len(requests) == 1
    assert requests[0]['logprobs'] is logprobs
    assert requests[0].get('top_logprobs') == (2 if logprobs else None)
    assert output.completions == ['café', 'blue sky']
    assert [choice.stop_reason for choice in output.choices] == ['stop', 'stop']
    assert output.usage is not None
    assert output.usage.total_tokens == 7
    for choice, tokens in zip(output.choices, TOKENS):
        if logprobs:
            assert choice.logprobs is not None
            assert choice.logprobs.model_dump() == {'content': tokens}
        else:
            assert choice.logprobs is None
    assert ModelOutput.model_validate_json(output.model_dump_json()).choices == output.choices


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('parts,expected', [
    ([None, None, None], None),
    ([{'content': []}, None, None], {'content': [], 'refusal': None}),
    (
        [{'refusal': [TOKENS[0][0]]}, {'refusal': [TOKENS[0][1]]}, None],
        {'content': None, 'refusal': TOKENS[0]},
    ),
    (
        [{'content': [], 'refusal': [TOKENS[0][0]]}, {'content': [TOKENS[0][1]], 'refusal': []}, None],
        {'content': [TOKENS[0][1]], 'refusal': [TOKENS[0][0]]},
    ),
], ids=['absent', 'empty-content', 'refusal', 'independent-fields'])
def test_optional_logprob_parts(
    asynchronous: bool, parts: list[dict[str, Any] | None], expected: dict[str, Any] | None,
) -> None:
    chunks = [
        ChatCompletionChunk.model_validate({
            'id': 'completion-id',
            'created': 1,
            'model': 'test-model',
            'object': 'chat.completion.chunk',
            'choices': [{
                'index': 0,
                'delta': {'content': 'reply'} if index == 0 else {},
                'finish_reason': 'stop' if index == 2 else None,
                'logprobs': part,
            }],
        })
        for index, part in enumerate(parts)
    ]
    original = [chunk.model_dump() for chunk in chunks]

    async def collect_async() -> Any:
        async def events() -> Any:
            for chunk in chunks:
                yield chunk
        return await async_collect_stream_response(events())

    completion, _ = asyncio.run(collect_async()) if asynchronous else collect_stream_response(iter(chunks))
    actual = completion.choices[0].logprobs
    assert (actual.model_dump() if actual is not None else None) == expected
    assert [chunk.model_dump() for chunk in chunks] == original
