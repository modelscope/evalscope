"""Upstream-failure paths for the Responses bridge route.

When ``Model.generate_async`` raises, the bridge must return:
- JSON mode: HTTP 502 with ``{"error": {"type": "api_error", "message": ...}}``
- SSE mode: a single ``event: error`` frame using the OpenAI SDK
  ``ResponseErrorEvent`` shape (flat ``type/code/message/param/sequence_number``),
  NOT a fabricated ``response.failed`` shape (that one would require a fully
  constructed Response object — overkill for an upstream failure).

Reference: ``openai/types/responses/response_error_event.py`` in the openai
Python SDK v2.x.

The wire contract is only half of it: the bridge must also record the failed
attempt on ``AgentTrace``. The log is not the artifact anyone analyses -- the
trace is -- so a failure that is logged but not recorded makes the attempt
disappear, and a run served by a flaky endpoint reads as shorter and cheaper
than the same run on a stable one.
"""

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.agent.trace import EventType
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.models.mockllm import MockLLM
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner


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


def _build_model_failing_once(monkeypatch, exc: Exception) -> Model:
    """A Model whose ``generate_async`` raises on the first call, then succeeds.

    Mirrors what the agent client actually experiences: one transient upstream
    error followed by its own retry.
    """
    api = MockLLM(model_name='mock-responses', custom_outputs=[])
    model = Model(api=api, config=GenerateConfig())
    calls = {'n': 0}

    async def _flaky(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise exc
        return ModelOutput.from_content(model='mock-responses', content='recovered')

    monkeypatch.setattr(model, 'generate_async', _flaky)
    return model


def _user_input(text: str) -> dict:
    return {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}


def _post_responses(proxy, session, *, stream: bool) -> None:
    """Fire one Responses request through the bridge, discarding the wire result.

    The wire shape is asserted by the two contract tests in this module; the
    trace tests only need the round-trip to have happened.
    """
    body = {'model': 'mock-responses', 'input': [_user_input('hi')]}
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {session.token}'}
    if stream:
        body['stream'] = True
        headers['Accept'] = 'text/event-stream'
    req = urllib.request.Request(
        f'{proxy.base_url}/openai/v1/responses',
        data=json.dumps(body).encode('utf-8'),
        headers=headers,
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.HTTPError as he:
        he.read()


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


@pytest.mark.parametrize(('stream', 'mode'), [(False, 'json'), (True, 'stream')])
def test_responses_upstream_failure_is_recorded_on_the_trace(monkeypatch, stream, mode):
    """A generate that never returned must still leave an ``ERROR`` event.

    Only recording the turns that succeeded is what the native
    :class:`AgentLoop` already avoids -- it emits ``EventType.ERROR`` for its
    own failure modes -- so the two paths would otherwise disagree about what a
    trace means.
    """
    exc = RuntimeError('upstream went boom')

    async def _go():
        model = _build_raising_model(monkeypatch, exc)
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: _post_responses(proxy, session, stream=stream)
            )
            return session.recorder.snapshot()

    trace = AsyncioLoopRunner.run(_go())

    errors = [e for e in trace.events if e.type == EventType.ERROR]
    assert len(errors) == 1, f'expected one ERROR event, got {[e.type for e in trace.events]}'
    assert errors[0].payload['source'] == 'upstream'
    assert errors[0].payload['mode'] == mode
    assert errors[0].payload['error'] == 'RuntimeError'
    assert 'upstream went boom' in errors[0].payload['message']
    assert errors[0].latency_ms is not None
    assert not [e for e in trace.events if e.type == EventType.MODEL_GENERATE]


def test_responses_retry_after_failure_lands_on_the_same_step(monkeypatch):
    """The failed attempt shares its step with the retry that replaced it.

    Recording a failure must not consume a step: the turn produced no assistant
    message, so advancing would shift every later step by one and stop ``step``
    lining up with the native loop's numbering.
    """

    async def _go():
        model = _build_model_failing_once(monkeypatch, RuntimeError('transient'))
        proxy = await ModelProxyServer.get_or_start()
        async with proxy.trial_session(model=model, framework='mock') as session:
            loop = asyncio.get_running_loop()
            for _ in range(2):  # first attempt fails, the client retries
                await loop.run_in_executor(None, lambda: _post_responses(proxy, session, stream=False))
            return session.recorder.snapshot()

    trace = AsyncioLoopRunner.run(_go())

    errors = [e for e in trace.events if e.type == EventType.ERROR]
    generates = [e for e in trace.events if e.type == EventType.MODEL_GENERATE]
    assert len(errors) == 1, f'expected one ERROR event, got {[e.type for e in trace.events]}'
    assert len(generates) == 1, f'expected one MODEL_GENERATE event, got {[e.type for e in trace.events]}'
    assert errors[0].step == 0
    assert generates[0].step == 0
