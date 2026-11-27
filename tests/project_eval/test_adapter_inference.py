import json
from pathlib import Path
from typing import Any

import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.evaluator import InferenceResult
from evalscope.benchmarks.project_eval.agent_clients import AgentProtocolError, AgentStep, AgentTurn, HttpTaskClient
from evalscope.benchmarks.project_eval.project_eval_adapter import ProjectEvalAdapter
from evalscope.config import TaskConfig
from evalscope.project_eval.schema import EvalTask

META = BenchmarkMeta(name='project_eval', dataset_id='project_eval')

ROW: dict[str, Any] = {
    'id': 't1',
    'version': 1,
    'question': '查余额',
    'system': '客服',
    'messages': [],
    'rubric': [{
        'id': 'c1',
        'criterion': '调用了 get_balance',
        'must_pass': True,
        'evidence_requirement': 'process',
    }],
    'checks': [],
    'exec': {
        'type': 'agent_api',
        'protocol': 'rest_steps',
        'agent_url_env': 'EVAL_AGENT_URL',
        'trace': {'mode': 'auto'},
        'session': {'id_from': 'sample_id'},
    },
    'source_refs': [],
    'origin': 'fact',
    'materials': [],
    'metadata': {},
}


def adapter_with(turn: AgentTurn | None) -> ProjectEvalAdapter:
    class StubClient:
        def chat(self, task: EvalTask, session_id: str) -> AgentTurn:
            assert turn is not None
            return turn

    adapter = ProjectEvalAdapter(benchmark_meta=META, task_config=None)
    adapter._client_for = lambda sample: StubClient()
    return adapter


def test_record_to_sample_keeps_metadata() -> None:
    adapter = adapter_with(None)
    sample = adapter.record_to_sample(ROW)
    assert sample.metadata['id'] == 't1'
    assert sample.target == ''


def test_reference_answer_becomes_sample_target() -> None:
    adapter = adapter_with(None)
    row = {**ROW, 'reference_answer': '当前余额 2000 元'}
    sample = adapter.record_to_sample(row)
    assert sample.target == '当前余额 2000 元'


def test_l2_turn_produces_trace_events() -> None:
    turn = AgentTurn(
        reply='余额 2000 元',
        trace_id='tr-1',
        steps=[AgentStep(
            type='tool',
            name='get_balance',
            arguments={},
            result={'balance_cent': 200000},
            elapsed_ms=12.0,
            raw={},
        )],
        state={'balance_cent': 200000},
        usage={'total_tokens': 20},
        trace_available=True,
    )
    adapter = adapter_with(turn)
    sample = adapter.record_to_sample(ROW)
    result = adapter._on_inference(None, sample)
    assert isinstance(result, InferenceResult)
    assert result.trace.trial_id == 'tr-1'
    assert sample.metadata['agent_state'] == {'balance_cent': 200000}
    assert [event.type.value for event in result.trace.events] == [
        'run_start',
        'tool_call',
        'tool_result',
        'submit',
        'run_end',
    ]


def test_l0_turn_marks_trace_unavailable() -> None:
    turn = AgentTurn(
        reply='余额 2000 元',
        trace_id=None,
        steps=[],
        state={},
        usage={},
        trace_available=False,
    )
    adapter = adapter_with(turn)
    result = adapter._on_inference(None, adapter.record_to_sample(ROW))
    assert result.trace.events[0].payload['trace_available'] is False


def test_messages_override_question() -> None:
    adapter = adapter_with(None)
    row = {**ROW, 'messages': [{'role': 'user', 'content': '第一轮'},
                               {'role': 'user', 'content': '确认'}]}
    sample = adapter.record_to_sample(row)
    assert [message.role for message in sample.input] == ['user', 'user']
    assert [message.content for message in sample.input] == ['第一轮', '确认']


def test_http_exec_uses_task_client() -> None:
    row = {
        **ROW,
        'exec': {
            'type': 'http',
            'base_url_env': 'EVAL_AGENT_URL',
            'request': {'method': 'POST', 'path': '/release', 'json': {'id': 'doc-1'}},
            'trace': {'mode': 'none'},
            'session': {'id_from': 'sample_id'},
        },
    }
    adapter = ProjectEvalAdapter(benchmark_meta=META, task_config=None)
    adapter._client_for = lambda sample: HttpTaskClient(
        'http://agent',
        transport=lambda method, url, body, timeout: {'reply': 'published', 'status': 'published'},
    )
    sample = adapter.record_to_sample(row)
    result = adapter._on_inference(None, sample)
    assert result.output.completion == 'published'
    assert sample.metadata['agent_state'] == {'reply': 'published', 'status': 'published'}
    assert result.trace.events[0].payload['trace_available'] is False


def test_reset_url_is_called_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        'evalscope.benchmarks.project_eval.project_eval_adapter.requests.post',
        lambda url, timeout: calls.append((url, timeout)) or Response(),
    )
    turn = AgentTurn(reply='ok', steps=[], trace_available=False)
    adapter = adapter_with(turn)
    row = {
        **ROW,
        'exec': {**ROW['exec'], 'reset_url': 'http://127.0.0.1:9201/v1/eval/reset'},
    }
    adapter._on_inference(None, adapter.record_to_sample(row))
    assert calls == [('http://127.0.0.1:9201/v1/eval/reset', 10)]


def test_non_loopback_reset_url_is_rejected() -> None:
    turn = AgentTurn(reply='ok', steps=[], trace_available=False)
    adapter = adapter_with(turn)
    row = {
        **ROW,
        'exec': {**ROW['exec'], 'reset_url': 'http://example.com/reset'},
    }
    with pytest.raises(AgentProtocolError, match='loopback'):
        adapter._on_inference(None, adapter.record_to_sample(row))


def test_missing_project_root_with_checks_is_rejected() -> None:
    turn = AgentTurn(reply='ok', steps=[], trace_available=False)
    adapter = adapter_with(turn)
    row = {
        **ROW,
        'checks': [{'id': 'f-1', 'type': 'file_exists', 'path': 'state.json'}],
    }
    with pytest.raises(AgentProtocolError, match='missing project_root'):
        adapter._on_inference(None, adapter.record_to_sample(row))


def test_inline_trace_mode_requires_steps() -> None:
    turn = AgentTurn(reply='ok', steps=[], trace_available=False)
    adapter = adapter_with(turn)
    row = {
        **ROW,
        'exec': {**ROW['exec'], 'trace': {'mode': 'inline'}},
    }
    with pytest.raises(AgentProtocolError, match='trace.mode=inline'):
        adapter._on_inference(None, adapter.record_to_sample(row))


def test_trace_mode_none_clears_trace_id_and_capability() -> None:
    turn = AgentTurn(reply='ok', trace_id='tr-1', steps=[], trace_available=False)
    adapter = adapter_with(turn)
    row = {
        **ROW,
        'exec': {**ROW['exec'], 'trace': {'mode': 'none'}},
    }
    sample = adapter.record_to_sample(row)
    result = adapter._on_inference(None, sample)
    assert sample.metadata['agent_capability'] == 'L0'
    assert result.trace.trial_id is None


def test_duplicate_task_ids_are_rejected_at_load(tmp_path: Path) -> None:
    eval_set = tmp_path / 'eval_set.v1.jsonl'
    eval_set.write_text('\n'.join(json.dumps(ROW) for _ in range(2)), encoding='utf-8')
    config = TaskConfig(
        model='m',
        eval_type='mock_llm',
        datasets=['project_eval'],
        dataset_args={'project_eval': {'local_path': str(eval_set)}},
        work_dir=str(tmp_path / 'outputs'),
        no_timestamp=True,
    )
    adapter = ProjectEvalAdapter(
        benchmark_meta=BenchmarkMeta(name='project_eval', dataset_id=str(eval_set)),
        task_config=config,
    )
    with pytest.raises(ValueError, match='duplicate task id'):
        adapter.load_from_disk()
