import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.evaluator.cache import CacheManager
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessageUser, ContentReasoning
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.utils.io_utils import OutputsStructure


@pytest.fixture
def provider_url() -> Iterator[str]:
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            reason = request['model']
            choices = [
                {
                    'index': 1,
                    'finish_reason': 123 if reason == 'invalid-reason' else reason,
                    'message': {'role': 'assistant', 'content': 'second answer', 'reasoning_content': 'second thought'},
                },
                {
                    'index': 'invalid' if reason == 'invalid-index' else 0,
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'first answer', 'reasoning_content': 'first thought'},
                },
            ]
            response = {
                'id': 'completion-id', 'created': 1, 'model': reason,
                'object': 'chat.completion', 'choices': choices,
                'usage': {
                    'prompt_tokens': 7, 'completion_tokens': 9, 'total_tokens': 16,
                    'completion_tokens_details': {'reasoning_tokens': 3},
                },
            }
            if request.get('stream'):
                chunks = [
                    {
                        **response, 'object': 'chat.completion.chunk', 'usage': None,
                        'choices': [
                            {'index': choice['index'], 'delta': choice['message'], 'finish_reason': None}
                            for choice in choices
                        ],
                    },
                    {
                        **response, 'object': 'chat.completion.chunk', 'usage': None,
                        'choices': [
                            {'index': choice['index'], 'delta': {}, 'finish_reason': choice['finish_reason']}
                            for choice in choices
                        ],
                    },
                    {**response, 'object': 'chat.completion.chunk', 'choices': []},
                ]
                body = (''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks) + 'data: [DONE]\n\n').encode()
                content_type = 'text/event-stream'
            else:
                body = json.dumps(response).encode()
                content_type = 'application/json'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(server_address=('127.0.0.1', 0), RequestHandlerClass=Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def _generate(*, api: OpenAICompatibleAPI, asynchronous: bool, stream: bool) -> ModelOutput:
    config = GenerateConfig(stream=stream, retries=0)
    messages = [ChatMessageUser(content='answer')]

    async def generate_async() -> ModelOutput:
        try:
            return await api.generate_async(input=messages, tools=[], tool_choice='none', config=config)
        finally:
            await api.aclose()

    try:
        if asynchronous:
            return asyncio.run(generate_async())
        return api.generate(input=messages, tools=[], tool_choice='none', config=config)
    finally:
        api.client.close()


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('stream', [True, False], ids=['sse', 'json'])
@pytest.mark.parametrize('reason,stop_reason', [
    ('stop', 'stop'),
    ('length', 'max_tokens'),
    ('tool_calls', 'tool_calls'),
    ('content_filter', 'content_filter'),
    ('function_call', 'tool_calls'),
    ('repetition_truncation', 'unknown'),
    ('provider_specific', 'unknown'),
])
def test_provider_finish_reason(
    provider_url: str, tmp_path: Path, asynchronous: bool, stream: bool, reason: str, stop_reason: str,
) -> None:
    api = OpenAICompatibleAPI(model_name=reason, base_url=provider_url, api_key='EMPTY', max_retries=0)
    output = _generate(api=api, asynchronous=asynchronous, stream=stream)

    assert [choice.stop_reason for choice in output.choices] == ['stop', stop_reason]
    assert [choice.message.text for choice in output.choices] == ['first answer', 'second answer']
    for choice, reasoning in zip(output.choices, ['first thought', 'second thought']):
        assert isinstance(choice.message.content, list)
        assert choice.message.content[0] == ContentReasoning(reasoning=reasoning, reasoning_tokens=3)
    assert output.usage is not None
    assert (output.usage.input_tokens, output.usage.output_tokens, output.usage.total_tokens) == (7, 9, 16)
    assert output.usage.reasoning_tokens == 3
    assert output.message.perf_metrics is not None
    assert (output.message.perf_metrics.ttft is not None) == stream
    expected_metadata = {'finish_reasons': {'1': reason}} if stop_reason == 'unknown' else None
    assert output.metadata == expected_metadata

    sample = Sample(id=0, input='answer', target='second answer')
    cache = CacheManager(
        outputs=OutputsStructure(str(tmp_path)), model_name=reason, benchmark_name='fixture',
    )
    try:
        cache.save_prediction_cache(
            subset='default', task_state=TaskState(model=reason, sample=sample, output=output),
        )
        cached, remaining = cache.filter_prediction_cache(subset='default', dataset=MemoryDataset(samples=[sample]))
    finally:
        cache.close()
    assert len(cached) == 1
    assert len(remaining) == 0
    assert cached[0].output.model_dump() == output.model_dump()
    print(f'HTTP 200 {"SSE [DONE]" if stream else "JSON"} {"async" if asynchronous else "sync"}: '
          f'{reason} -> {stop_reason}; cached metadata={cached[0].output.metadata}')


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('reason,field', [('invalid-reason', 'finish_reason'), ('invalid-index', 'index')])
def test_stream_fields_remain_validated(provider_url: str, asynchronous: bool, reason: str, field: str) -> None:
    api = OpenAICompatibleAPI(model_name=reason, base_url=provider_url, api_key='EMPTY', max_retries=0)
    with pytest.raises(ValidationError, match=field):
        _generate(api=api, asynchronous=asynchronous, stream=True)
