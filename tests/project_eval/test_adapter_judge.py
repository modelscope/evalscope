import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.evaluator import TaskState
from evalscope.api.evaluator.cache import ModelResult
from evalscope.api.judge import CaseVerdict, JudgeContext, JudgeDefinition
from evalscope.api.model import ModelOutput
from evalscope.benchmarks.project_eval.agent_clients import AgentProtocolError, AgentTurn
from evalscope.benchmarks.project_eval.project_eval_adapter import ProjectEvalAdapter, RubricVerdict
from evalscope.constants import ScoreStatus
from evalscope.project_eval.checks import run_checks
from evalscope.project_eval.schema import EvalTask

META = BenchmarkMeta(name='project_eval', dataset_id='project_eval')

ROW: dict[str, Any] = {
    'id': 't1',
    'question': '查余额',
    'system': '客服',
    'rubric': [{
        'id': 'c1',
        'criterion': '余额正确',
        'must_pass': True,
        'evidence_requirement': 'outcome',
    }],
    'checks': [{
        'id': 'j-1',
        'type': 'json_path',
        'path': 'state.json',
        'json_path': 'balance_cent',
        'expect': 200000,
    }],
    'exec': {
        'type': 'agent_api',
        'protocol': 'rest_steps',
        'agent_url_env': 'EVAL_AGENT_URL',
        'trace': {'mode': 'auto'},
        'session': {'id_from': 'sample_id'},
    },
    'metadata': {},
}


def _attach_checks(adapter: ProjectEvalAdapter, sample: Any, root: Path) -> None:
    task = adapter._task_from_metadata(sample.metadata)
    sample.metadata['runtime_checks'] = [result.model_dump() for result in run_checks(task, root)]


def context(tmp_path: Path, balance: int, trace_available: bool = False) -> tuple[ProjectEvalAdapter, JudgeContext]:
    (tmp_path / 'state.json').write_text(json.dumps({'balance_cent': balance}))
    adapter = ProjectEvalAdapter(benchmark_meta=META, task_config=None)
    sample = adapter.record_to_sample(ROW)
    sample.metadata['project_root'] = str(tmp_path)
    state = TaskState(
        model='m',
        sample=sample,
        output=ModelOutput.from_content(model='m', content='余额 2000 元'),
        completed=True,
    )
    if trace_available:
        from evalscope.api.agent import AgentTrace, EventType

        trace = AgentTrace(events=[])
        trace.add_event(step=1, type=EventType.TOOL_CALL, payload={'tool': 'get_balance'})
        state.agent_trace = trace
    _attach_checks(adapter, sample, tmp_path)
    return adapter, JudgeContext(
        task_state=state,
        original_prediction='余额 2000 元',
        filtered_prediction='余额 2000 元',
        reference='',
    )


def test_failed_must_pass_check_short_circuits(tmp_path: Path) -> None:
    adapter, ctx = context(tmp_path, 100)
    definition = adapter.judge_definition(ctx)
    assert definition.immediate_score is not None
    assert definition.immediate_score.value['task_pass'] == 0.0
    assert definition.immediate_score.value['evidence_coverage'] == 1.0
    assert 'must_pass' in definition.skip_reason


def test_passed_checks_open_rubric_workflow(tmp_path: Path) -> None:
    adapter, ctx = context(tmp_path, 200000)
    definition = adapter.judge_definition(ctx)
    assert definition.immediate_score is None
    assert len(definition.cases) == 1


def _definition_with(
    tmp_path: Path,
    rubric: list[dict[str, Any]],
    *,
    threshold: float = 0.6,
    trace_available: bool = False,
    agent_state: dict[str, Any] | None = None,
) -> tuple[JudgeDefinition, JudgeContext]:
    row = {
        **ROW,
        'rubric': rubric,
        'pass_threshold': threshold,
        'checks': [{
            'id': 'j-1', 'type': 'json_path', 'path': 'state.json',
            'json_path': 'balance_cent', 'expect': 200000,
        }],
    }
    (tmp_path / 'state.json').write_text(json.dumps({'balance_cent': 200000}))
    adapter = ProjectEvalAdapter(benchmark_meta=META, task_config=None)
    sample = adapter.record_to_sample(row)
    sample.metadata['project_root'] = str(tmp_path)
    state = TaskState(model='m', sample=sample,
                      output=ModelOutput.from_content(model='m', content='余额 2000 元'),
                      completed=True)
    if trace_available:
        from evalscope.api.agent import AgentTrace, EventType
        trace = AgentTrace(events=[])
        trace.add_event(step=1, type=EventType.TOOL_CALL, payload={'tool': 'get_balance'})
        state.agent_trace = trace
    if agent_state is not None:
        state.metadata['agent_state'] = agent_state
    _attach_checks(adapter, sample, tmp_path)
    ctx = JudgeContext(task_state=state, original_prediction='余额 2000 元',
                       filtered_prediction='余额 2000 元', reference='')
    return adapter.judge_definition(ctx), ctx


def test_weighted_score_and_threshold_fail(tmp_path: Path) -> None:
    definition, ctx = _definition_with(tmp_path, [
        {'id': 'c1', 'criterion': '余额正确', 'weight': 1, 'must_pass': True,
         'evidence_requirement': 'outcome'},
        {'id': 'c2', 'criterion': '回答简洁', 'weight': 1, 'must_pass': False,
         'evidence_requirement': 'answer'},
    ], threshold=0.9)
    reduced = definition.reduce([
        CaseVerdict(case_id='c1', value=RubricVerdict(criteria_met=True)),
        CaseVerdict(case_id='c2', value=RubricVerdict(criteria_met=False)),
    ], ctx)
    assert reduced.value['rubric_score'] == 0.5
    assert reduced.value['task_pass'] == 0.0


def test_must_pass_answer_criterion_fails_task(tmp_path: Path) -> None:
    definition, ctx = _definition_with(tmp_path, [
        {'id': 'c1', 'criterion': '余额正确', 'weight': 1, 'must_pass': True,
         'evidence_requirement': 'outcome'},
        {'id': 'c2', 'criterion': '必须说明货币单位', 'weight': 1, 'must_pass': True,
         'evidence_requirement': 'answer'},
    ])
    reduced = definition.reduce([
        CaseVerdict(case_id='c1', value=RubricVerdict(criteria_met=True)),
        CaseVerdict(case_id='c2', value=RubricVerdict(criteria_met=False)),
    ], ctx)
    assert reduced.value['rubric_score'] == 0.5
    assert reduced.value['task_pass'] == 0.0


def test_process_criterion_is_not_evaluable_without_trace(tmp_path: Path) -> None:
    definition, ctx = _definition_with(tmp_path, [
        {'id': 'c1', 'criterion': '余额正确', 'weight': 1, 'must_pass': True,
         'evidence_requirement': 'outcome'},
        {'id': 'c2', 'criterion': '调用了工具', 'weight': 1, 'must_pass': False,
         'evidence_requirement': 'process'},
    ])
    assert [case.case_id for case in definition.cases] == ['c1']
    reduced = definition.reduce([
        CaseVerdict(case_id='c1', value=RubricVerdict(criteria_met=True)),
    ], ctx)
    assert reduced.value['evidence_coverage'] == 0.5
    assert reduced.metadata['not_evaluable'] == ['c2']
    assert any(
        verdict['criterion_id'] == 'c2' and verdict.get('not_evaluable') is True
        for verdict in reduced.metadata['rubric_verdicts']
    )


def test_must_pass_process_without_trace_marks_needs_review(tmp_path: Path) -> None:
    definition, ctx = _definition_with(tmp_path, [
        {'id': 'c1', 'criterion': '余额正确', 'weight': 1, 'must_pass': True,
         'evidence_requirement': 'outcome'},
        {'id': 'c2', 'criterion': '调用了工具', 'weight': 1, 'must_pass': True,
         'evidence_requirement': 'process'},
    ])
    reduced = definition.reduce([
        CaseVerdict(case_id='c1', value=RubricVerdict(criteria_met=True)),
    ], ctx)
    assert reduced.metadata['needs_review'] is True
    assert reduced.value['evidence_coverage'] == 0.5


def test_all_process_without_trace_is_excluded(tmp_path: Path) -> None:
    definition, _ = _definition_with(tmp_path, [
        {'id': 'c1', 'criterion': '调用了工具', 'weight': 1, 'must_pass': True,
         'evidence_requirement': 'process'},
    ])
    assert definition.immediate_score is not None
    assert definition.immediate_score.status is ScoreStatus.EXCLUDED
    assert definition.immediate_score.value == {}
    assert definition.immediate_score.metadata['needs_review'] is True


def test_judge_request_requires_json_verdict(tmp_path: Path) -> None:
    definition, ctx = _definition_with(tmp_path, [
        {'id': 'c1', 'criterion': '余额正确', 'weight': 1, 'must_pass': True,
         'evidence_requirement': 'outcome'},
    ])
    request = definition.request(definition.cases[0], 'original', [], ctx)
    assert 'Reply with a single JSON object' in request.messages[0].content


def test_judge_request_includes_question_and_state(tmp_path: Path) -> None:
    definition, ctx = _definition_with(
        tmp_path,
        [
            {'id': 'c1', 'criterion': '余额正确', 'weight': 1, 'must_pass': True,
             'evidence_requirement': 'outcome'},
        ],
        agent_state={'balance_cent': 200000},
    )
    request = definition.request(definition.cases[0], 'original', [], ctx)
    prompt = request.messages[0].content
    assert '# Question' in prompt
    assert '查余额' in prompt
    assert '# Agent state' in prompt
    assert '200000' in prompt


def test_judge_request_includes_reference_and_expected_facts(tmp_path: Path) -> None:
    row = {
        **ROW,
        'reference_answer': '当前余额 2000 元',
        'expected_facts': ['2000 元'],
        'forbidden_facts': ['9999 元'],
    }
    (tmp_path / 'state.json').write_text(json.dumps({'balance_cent': 200000}))
    adapter = ProjectEvalAdapter(benchmark_meta=META, task_config=None)
    sample = adapter.record_to_sample(row)
    sample.metadata['project_root'] = str(tmp_path)
    state = TaskState(model='m', sample=sample,
                      output=ModelOutput.from_content(model='m', content='余额 2000 元'),
                      completed=True)
    _attach_checks(adapter, sample, tmp_path)
    ctx = JudgeContext(task_state=state, original_prediction='余额 2000 元',
                       filtered_prediction='余额 2000 元', reference='')
    definition = adapter.judge_definition(ctx)
    request = definition.request(definition.cases[0], 'original', [], ctx)
    prompt = request.messages[0].content
    assert '# Reference answer' in prompt
    assert '当前余额 2000 元' in prompt
    assert '2000 元' in prompt
    assert '9999 元' in prompt


def test_cached_review_uses_saved_end_state_checks(tmp_path: Path) -> None:
    db = tmp_path / 'agent.db'
    connection = sqlite3.connect(db)
    connection.execute('create table agent_audit_log(confirmed integer)')
    connection.execute('insert into agent_audit_log values (1)')
    connection.commit()
    connection.close()

    row = {
        **ROW,
        'rubric': [{
            'id': 'c1',
            'criterion': '退款已确认',
            'weight': 1,
            'must_pass': True,
            'evidence_requirement': 'outcome',
        }],
        'checks': [{
            'id': 'state-1',
            'type': 'sqlite',
            'dsn': 'agent.db',
            'query': 'select confirmed from agent_audit_log limit 1',
            'expect': 1,
        }],
    }
    adapter = ProjectEvalAdapter(benchmark_meta=META, task_config=None)
    sample = adapter.record_to_sample(row)
    sample.metadata['project_root'] = str(tmp_path)

    class FakeClient:
        def chat(self, task: EvalTask, session_id: str) -> AgentTurn:
            return AgentTurn(reply='已退款')

    adapter._client_for = lambda _sample: FakeClient()
    result = adapter._on_inference(None, sample)
    state = TaskState(model='m', sample=sample, output=result.output, completed=True)
    state.messages = result.messages or []
    ctx = JudgeContext(
        task_state=state,
        original_prediction=result.output.completion,
        filtered_prediction=result.output.completion,
        reference='',
    )

    first = adapter.judge_definition(ctx).reduce(
        [CaseVerdict(case_id='c1', value=RubricVerdict(criteria_met=True))],
        ctx,
    )
    assert first.value['task_pass'] == 1.0
    saved_checks: list[dict[str, Any]] = list(sample.metadata['runtime_checks'])
    cached = ModelResult.from_task_state(state)
    assert cached.metadata['runtime_checks'] == saved_checks
    cached.index = 0
    restored = cached.to_task_state([sample])
    cached_ctx = JudgeContext(
        task_state=restored,
        original_prediction=result.output.completion,
        filtered_prediction=result.output.completion,
        reference='',
    )

    connection = sqlite3.connect(db)
    connection.execute('delete from agent_audit_log')
    connection.commit()
    connection.close()

    second = adapter.judge_definition(cached_ctx).reduce(
        [CaseVerdict(case_id='c1', value=RubricVerdict(criteria_met=True))],
        cached_ctx,
    )
    assert second.value['task_pass'] == 1.0
    assert second.metadata['checks'] == saved_checks

    sample.metadata.pop('runtime_checks')
    with pytest.raises(AgentProtocolError, match='missing saved end-state checks'):
        adapter.judge_definition(ctx)


def test_corrupt_saved_checks_fail_loudly(tmp_path: Path) -> None:
    adapter, ctx = context(tmp_path, 200000)
    ctx.task_state.metadata['runtime_checks'] = [1]
    with pytest.raises(AgentProtocolError, match='missing saved end-state checks'):
        adapter.judge_definition(ctx)
