"""Upstream-failure paths for the Responses bridge route.

When ``Model.generate_async`` raises, the bridge must return:
- JSON mode: HTTP 502 with ``{"error": {"type": "api_error", "message": ...}}``
- SSE mode: a single ``event: error`` frame using the OpenAI SDK
  ``ResponseErrorEvent`` shape (flat ``type/code/message/param/sequence_number``),
  NOT a fabricated ``response.failed`` shape (that one would require a fully
  constructed Response object — overkill for an upstream failure).

Reference: ``openai/types/responses/response_error_event.py`` in the openai
Python SDK v2.x.
"""

import asyncio
import json
import pytest
import urllib.error
import urllib.request

from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.model import GenerateConfig, Model
from evalscope.models.mockllm import MockLLM
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _build_raising_model(monkeypatch, exc: Exception) -> Model:
    """A Model whose ``generate_async`` always raises ``exc``."""
    api = MockLLM(model_name='mock-responses', custom_outputs=[])
    model = Model(api=api, config=GenerateConfig())

    async def _raise(*args, **kwargs):
        raise exc

    monkeypatch.setattr(model, 'generate_async', _raise)
    return model


def _user_input(text: str) -> dict:
    return {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}


def test_responses_json_upstream_failure_returns_502_api_error(monkeypatch):
    """JSON-mode: upstream raise → 502 with shape {'error': {'type': 'api_error', 'message': ...}}."""
    exc = RuntimeError('upstream went boom')

    async def _go():
        model = _build_raising_model(monkeypatch, exc)
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                req = urllib.request.Request(
                    url,
                    data=json.dumps({'model': 'mock-responses', 'input': [_user_input('hi')]}).encode('utf-8'),
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {session.token}',
                    },
                    method='POST',
                )
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        return resp.status, json.loads(resp.read().decode('utf-8'))
                except urllib.error.HTTPError as he:
                    return he.code, json.loads(he.read().decode('utf-8'))

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    status, body = AsyncioLoopRunner.run(_go())
    assert status == 502
    assert body['error']['type'] == 'api_error'
    assert 'upstream went boom' in body['error']['message']


def test_responses_streaming_upstream_failure_emits_openai_error_event(monkeypatch):
    """SSE-mode: upstream raise → single ``event: error`` frame with flat shape per OpenAI SDK."""
    exc = RuntimeError('boom in stream')

    async def _go():
        model = _build_raising_model(monkeypatch, exc)
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                req = urllib.request.Request(
                    url,
                    data=json.dumps({
                        'model': 'mock-responses',
                        'input': [_user_input('stream me')],
                        'stream': True,
                    }).encode('utf-8'),
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {session.token}',
                        'Accept': 'text/event-stream',
                    },
                    method='POST',
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.read().decode('utf-8')

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    raw = AsyncioLoopRunner.run(_go())
    # Find the error frame.
    frames = []
    for chunk in raw.split('\n\n'):
        lines = chunk.strip().split('\n')
        if len(lines) < 2 or not lines[0].startswith('event: '):
            continue
        event = lines[0][len('event: '):]
        data_line = next((ln for ln in lines if ln.startswith('data: ')), None)
        if data_line is None:
            continue
        frames.append((event, json.loads(data_line[len('data: '):])))

    assert len(frames) == 1, f'expected a single error frame, got {len(frames)}: {[ev for ev, _ in frames]}'
    event, payload = frames[0]
    assert event == 'error', f'event name must be "error" (not "response.failed"), got {event!r}'
    # OpenAI SDK ResponseErrorEvent shape: flat fields, not nested under 'error'.
    assert payload['type'] == 'error'
    assert payload['code'] == 'api_error'
    assert 'boom in stream' in payload['message']
    assert 'param' in payload  # may be None, but must be present
    assert isinstance(payload['sequence_number'], int) and payload['sequence_number'] >= 1
    # MUST NOT have data: [DONE] sentinel (Responses uses response.completed; on
    # failure no completion event is sent either — the error frame is terminal).
    assert '[DONE]' not in raw
