"""Reaching max_tokens is a truncated reply, not an exceeded context window."""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip('anthropic')

import evalscope  # noqa: F401 - trigger strategy registration
from evalscope.api.agent import AgentContext, AgentLoop, EventType, ToolExecutor
from evalscope.api.agent.constants import LoopMessages
from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.registry import get_strategy
from evalscope.models.anthropic_compatible import AnthropicCompatibleAPI

TRUNCATED_REPLY = {
    'id': 'message-id',
    'type': 'message',
    'role': 'assistant',
    'model': 'test-model',
    'content': [{'type': 'text', 'text': 'partial'}],
    'stop_reason': 'max_tokens',
    'stop_sequence': None,
    'usage': {'input_tokens': 3, 'output_tokens': 5},
}
CONTEXT_WINDOW_EXCEEDED = {**TRUNCATED_REPLY, 'stop_reason': 'model_context_window_exceeded'}
PROMPT_TOO_LONG = {
    'type': 'error',
    'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 300000 tokens'},
}


def _stream_body(payload: Dict[str, Any]) -> bytes:
    message = {**payload, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 3, 'output_tokens': 0}}
    events = [
        {'type': 'message_start', 'message': message},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'partial'}},
        {'type': 'content_block_stop', 'index': 0},
        {
            'type': 'message_delta',
            'delta': {'stop_reason': payload['stop_reason'], 'stop_sequence': None},
            'usage': {'output_tokens': 5},
        },
        {'type': 'message_stop'},
    ]
    return ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events).encode()


@pytest.fixture
def generate() -> Iterator[Callable[..., ModelOutput]]:
    """Run the real Anthropic SDK against a local server that answers with one canned response."""
    reply: Tuple[int, Dict[str, Any]] = (200, TRUNCATED_REPLY)

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            status, payload = reply
            stream = request.get('stream', False)
            body = _stream_body(payload) if stream else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'text/event-stream' if stream else 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def run(status: int, payload: Dict[str, Any], stream: bool = False, asynchronous: bool = False) -> ModelOutput:
        nonlocal reply
        reply = (status, payload)
        api = AnthropicCompatibleAPI(
            model_name='test-model', base_url=f'http://127.0.0.1:{server.server_port}', api_key='test-key'
        )
        config = GenerateConfig(retries=0, retry_interval=0, max_tokens=5, stream=stream)
        if asynchronous:

            async def generate_async() -> ModelOutput:
                try:
                    return await api.generate_async([ChatMessageUser(content='q')], [], 'none', config)
                finally:
                    await api.aclose()

            return asyncio.run(generate_async())
        return api.generate([ChatMessageUser(content='q')], [], 'none', config)

    try:
        yield run
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_max_tokens_stop_reason_matches_the_openai_path(generate: Callable) -> None:
    assert generate(200, TRUNCATED_REPLY).stop_reason == 'max_tokens'


def test_agent_loop_does_not_report_a_truncated_reply_as_context_overflow(generate: Callable) -> None:
    model = MagicMock()
    model.generate_async = AsyncMock(return_value=generate(200, TRUNCATED_REPLY))
    loop = AgentLoop(
        model=model,
        strategy=get_strategy('function_calling')(),
        tool_executor=ToolExecutor(handlers={}, environment=None),
        max_steps=1,
    )

    result = asyncio.run(loop.run(AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])))

    errors = [event.payload.get('message') for event in result.trace.events if event.type == EventType.ERROR]
    assert LoopMessages.MODEL_CONTEXT_OVERFLOW not in errors


def test_prompt_too_long_still_reports_model_length(generate: Callable) -> None:
    assert generate(400, PROMPT_TOO_LONG).stop_reason == 'model_length'


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('stream', [False, True], ids=['json', 'sse'])
def test_context_window_exceeded_stop_reason_maps_to_model_length(
    generate: Callable, stream: bool, asynchronous: bool
) -> None:
    output = generate(200, CONTEXT_WINDOW_EXCEEDED, stream=stream, asynchronous=asynchronous)

    assert output.stop_reason == 'model_length'
    assert output.completions == ['partial']
