from typing import Any

import pytest

from evalscope.benchmarks.project_eval.agent_clients import (
    AgentProtocolError,
    HttpTaskClient,
    OpenAICompatClient,
    RestStepsClient,
    detect_capability,
)
from evalscope.project_eval.schema import EvalTask, HttpRequestSpec

TASK = EvalTask.model_validate({
    'id': 't1',
    'question': 'q',
    'system': 's',
    'rubric': [{'id': 'c1', 'criterion': 'ok'}],
    'exec': {
        'type': 'agent_api',
        'protocol': 'rest_steps',
        'agent_url_env': 'EVAL_AGENT_URL',
        'trace': {'mode': 'auto'},
        'session': {'id_from': 'sample_id'},
    },
})


def test_rest_client_maps_steps() -> None:
    response = {
        'reply': 'done',
        'trace_id': 'tr-1',
        'steps': [{
            'type': 'tool',
            'name': 'get_balance',
            'arguments': {},
            'result': {'balance_cent': 200000},
        }],
        'state': {'balance_cent': 200000},
        'usage': {'total_tokens': 12},
    }
    turn = RestStepsClient('http://agent', transport=lambda *a, **k: response).chat(TASK, 't1')
    assert turn.reply == 'done'
    assert turn.steps[0].name == 'get_balance'
    assert detect_capability(turn) == 'L2'


def test_rest_client_sends_system_for_single_turn() -> None:
    calls: list[dict[str, Any]] = []

    def transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append(payload)
        return {'reply': 'done'}

    turn = RestStepsClient('http://agent', transport=transport).chat(TASK, 't1')
    assert turn.reply == 'done'
    assert calls[0]['system'] == 's'


def test_openai_compat_has_no_steps() -> None:
    response = {
        'choices': [{'message': {'content': 'done'}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 2},
    }
    turn = OpenAICompatClient('http://agent', 'm', transport=lambda *a, **k: response).chat(TASK, 't1')
    assert turn.reply == 'done'
    assert turn.trace_available is False
    assert detect_capability(turn) == 'L0'


def test_trace_id_without_steps_is_l1() -> None:
    response = {'reply': 'done', 'trace_id': 'tr-1', 'steps': [], 'state': {}, 'usage': {}}
    turn = RestStepsClient('http://agent', transport=lambda *a, **k: response).chat(TASK, 't1')
    assert detect_capability(turn) == 'L1'


def test_malformed_response_names_the_missing_field() -> None:
    with pytest.raises(AgentProtocolError, match='reply'):
        RestStepsClient('http://agent', transport=lambda *a, **k: {'steps': []}).chat(TASK, 't1')


def test_multi_turn_sends_each_user_message_and_aggregates_usage() -> None:
    multi = EvalTask.model_validate({
        **TASK.model_dump(),
        'messages': [{'role': 'user', 'content': '第一轮'}, {'role': 'user', 'content': '确认'}],
    })
    calls: list[dict[str, Any]] = []

    def transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append(payload)
        return {
            'reply': f"reply-{len(calls)}",
            'steps': [{'type': 'tool', 'name': 'get_balance', 'arguments': {}}],
            'usage': {'total_tokens': 3},
        }

    turn = RestStepsClient('http://agent', transport=transport).chat(multi, 't1')
    assert [call['message'] for call in calls] == ['第一轮', '确认']
    assert all('system' not in call for call in calls)
    assert turn.reply == 'reply-2'
    assert len(turn.steps) == 2
    assert turn.usage['total_tokens'] == 6


def test_http_task_client_returns_state() -> None:
    turn = HttpTaskClient(
        'http://agent',
        transport=lambda method, url, body, timeout: {
            'reply': 'ok',
            'status': 'published',
            'url': url,
            'method': method,
        },
    ).call(HttpRequestSpec(method='POST', path='/release', json_body={'id': 'doc-1'}))
    assert turn.reply == 'ok'
    assert turn.state['status'] == 'published'
    assert turn.trace_available is False


def test_openai_compat_missing_choices_is_protocol_error() -> None:
    with pytest.raises(AgentProtocolError, match='choices'):
        OpenAICompatClient('http://agent', 'm', transport=lambda *a, **k: {}).chat(TASK, 't1')
