"""``previous_response_id`` is intentionally not supported in P0.

codex ``exec`` occasionally includes it; the bridge logs a warning and
still processes the full ``input[]`` rather than erroring. This test
covers both the warning path and the "request still succeeds" property.
"""

import asyncio
import json
import logging
import pytest
import urllib.request

from evalscope.agent.external.bridge import ModelProxyServer
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.models.mockllm import MockLLM
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _build_model(text: str) -> Model:
    msg = ChatMessageAssistant(content=text)
    out = ModelOutput(model='mock-responses', choices=[ChatCompletionChoice(message=msg, stop_reason='stop')])
    api = MockLLM(model_name='mock-responses', custom_outputs=[out])
    return Model(api=api, config=GenerateConfig())


def _post_json(url: str, body: dict, token: str) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))


def test_previous_response_id_warns_then_processes_full_input(caplog, monkeypatch):
    """Send ``previous_response_id`` + full input[] — bridge logs WARN but returns 200."""
    # evalscope's logger sets propagate=False, so caplog (which attaches a
    # handler to the root logger) sees nothing by default. Temporarily allow
    # propagation for this test.
    ev_logger = logging.getLogger('evalscope')
    monkeypatch.setattr(ev_logger, 'propagate', True)
    caplog.set_level(logging.WARNING, logger='evalscope')

    async def _go():
        proxy = await ModelProxyServer.get_or_start()
        model = _build_model('ok, processed')
        async with proxy.trial_session(model=model, framework='mock') as session:
            url = f'{proxy.base_url}/openai/v1/responses'

            def _request():
                return _post_json(
                    url,
                    {
                        'model': 'mock-responses',
                        'previous_response_id': 'resp_pretend_we_have_state',
                        'input': [{
                            'type': 'message',
                            'role': 'user',
                            'content': [{'type': 'input_text', 'text': 'continue?'}],
                        }],
                    },
                    session.token,
                )

            return await asyncio.get_running_loop().run_in_executor(None, _request)

    body = AsyncioLoopRunner.run(_go())
    # Response shape sane — bridge processed the full input[] (not 400).
    assert body['object'] == 'response'
    assert body['status'] == 'completed'
    assert body['output'][0]['content'][0]['text'] == 'ok, processed'

    # WARN log mentions previous_response_id + the offending value.
    warn_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and 'previous_response_id' in r.getMessage()
    ]
    assert warn_records, f'expected a previous_response_id WARN, got: {[r.getMessage() for r in caplog.records]}'
    assert 'resp_pretend_we_have_state' in warn_records[0].getMessage()
