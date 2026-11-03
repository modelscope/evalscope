"""Providers can omit token counts from usage; that must not fail a completed reply."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Tuple

import pytest

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.models.openai_compatible import OpenAICompatibleAPI

PARTIAL_USAGES = [
    # Vertex AI's Gemini omits completion_tokens when max_tokens is hit during reasoning (#1432).
    {'prompt_tokens': 5, 'total_tokens': 5},
    {'prompt_tokens': 5, 'completion_tokens': None, 'total_tokens': 5},
]
COMMON = {'id': 'completion-id', 'created': 1, 'model': 'test-model'}


@pytest.fixture
def generate() -> Iterator[Callable[[str, bytes, bool], ModelOutput]]:
    """Run the real OpenAI SDK against a local server that answers with one canned body."""
    reply: Tuple[str, bytes] = ('application/json', b'{}')

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers['Content-Length']))
            content_type, body = reply
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def run(content_type: str, body: bytes, stream: bool) -> ModelOutput:
        nonlocal reply
        reply = (content_type, body)
        api = OpenAICompatibleAPI(
            model_name='test-model', base_url=f'http://127.0.0.1:{server.server_port}/v1', api_key='test-key'
        )
        config = GenerateConfig(retries=0, retry_interval=0, stream=stream)
        return api.generate([ChatMessageUser(content='q')], [], 'none', config)

    try:
        yield run
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize('usage', PARTIAL_USAGES)
def test_json_reply_keeps_the_reported_counts(generate: Callable, usage: Dict[str, Any]) -> None:
    body = {
        **COMMON,
        'object': 'chat.completion',
        'choices': [{'index': 0, 'finish_reason': 'length', 'message': {'role': 'assistant', 'content': ''}}],
        'usage': usage,
    }

    output = generate('application/json', json.dumps(body).encode(), False)

    assert output.stop_reason == 'max_tokens'
    assert output.usage.input_tokens == 5
    assert output.usage.output_tokens == 0
    assert output.message.perf_metrics.input_tokens == 5


@pytest.mark.parametrize('usage', PARTIAL_USAGES)
def test_streamed_reply_keeps_the_reported_counts(generate: Callable, usage: Dict[str, Any]) -> None:
    chunk = {**COMMON, 'object': 'chat.completion.chunk'}
    chunks = [
        {**chunk, 'choices': [{'index': 0, 'delta': {'content': 'hi'}, 'finish_reason': 'length'}]},
        {**chunk, 'choices': [], 'usage': usage},
    ]
    body = ''.join(f'data: {json.dumps(item)}\n\n' for item in chunks) + 'data: [DONE]\n\n'

    output = generate('text/event-stream', body.encode(), True)

    assert output.completion == 'hi'
    assert output.usage.input_tokens == 5
    assert output.usage.output_tokens == 0
