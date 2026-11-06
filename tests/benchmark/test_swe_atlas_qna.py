"""Unit tests for the SWE-Atlas QnA adapter.

These cover the pieces that run without Docker or a judge model: dataset
conversion, rubric parsing (both polarities), final-answer extraction, and the
reduce step that folds rubric verdicts into the resolve-rate / rubric-pass-rate
metrics. The sandbox run and real judging are exercised end-to-end elsewhere.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
from evalscope.agent.external.runners import AgentRunResult
from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
from evalscope.agent.tools.bash import BASH_TOOL_INFO
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import CaseVerdict, JudgeContext
from evalscope.api.messages import ChatMessageAssistant, ContentReasoning, ContentText
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.benchmarks.swe_atlas_qna.prompts import JUDGE_SYSTEM_PROMPT
from evalscope.benchmarks.swe_atlas_qna.swe_atlas_qna_adapter import (
    RUBRIC_CONTRACT,
    RubricResponse,
    RubricVerdict,
    SweAtlasEnclaveEnvironment,
    SweAtlasQnAAdapter,
    SweAtlasSandboxMixin,
    SweAtlasSandboxState,
)
from evalscope.benchmarks.swe_atlas_qna.utils import (
    dirty_check_command,
    extract_final_answer,
    parse_rubric,
    reset_repo_command,
    rubric_item_passed,
)
from evalscope.config import TaskConfig
from evalscope.report.generator import ReportGenerator


def make_adapter(tmp_path, **extra_params: Any) -> SweAtlasQnAAdapter:
    cfg = TaskConfig(
        datasets=['swe_atlas_qna'],
        dataset_args={'swe_atlas_qna': {'extra_params': extra_params}},
        work_dir=str(tmp_path / 'outputs'),
    )
    adapter = get_benchmark('swe_atlas_qna', cfg)
    assert isinstance(adapter, SweAtlasQnAAdapter)
    return adapter


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_command_timeout_must_be_finite_and_positive(tmp_path, value: float) -> None:
    with pytest.raises(ValueError, match='command_timeout must be a finite positive number'):
        make_adapter(tmp_path, command_timeout=value)


@pytest.mark.parametrize('value', [-0.01, 1.01, float('nan'), float('inf')])
def test_pass_threshold_must_be_finite_and_in_unit_interval(tmp_path, value: float) -> None:
    with pytest.raises(ValueError, match=r'pass_threshold must be a finite number in \[0, 1\]'):
        make_adapter(tmp_path, pass_threshold=value)


# ---------------------------------------------------------------------------
# Pure helpers (no sandbox / judge)
# ---------------------------------------------------------------------------


def test_parse_rubric_marks_negative_verifier() -> None:
    items = parse_rubric([
        {'id': 'r1', 'title': 'Mentions the retry loop', 'annotations': {'type': 'positive hli verifier'}},
        {'id': 'r2', 'title': 'Does not invent an API', 'annotations': {'type': 'negative hli verifier'}},
    ])
    assert [item['id'] for item in items] == ['r1', 'r2']
    assert items[0]['negated'] is False
    assert items[1]['negated'] is True


def test_parse_rubric_accepts_json_string_and_defaults_ids() -> None:
    items = parse_rubric('[{"title": "no type key"}]')
    assert len(items) == 1
    assert items[0]['id'] == 'rubric_0'
    assert items[0]['negated'] is False


def test_parse_rubric_rejects_unknown_verifier_type() -> None:
    with pytest.raises(ValueError, match='unknown rubric verifier type'):
        parse_rubric([{'title': 'weighted result', 'annotations': {'type': 'weighted hli verifier'}}])


def test_rubric_item_passed_honours_polarity() -> None:
    # Positive verifier: the claim being met is a pass.
    assert rubric_item_passed(claim_met=True, negated=False) is True
    assert rubric_item_passed(claim_met=False, negated=False) is False
    # Negative verifier: the claim being met is a failure.
    assert rubric_item_passed(claim_met=True, negated=True) is False
    assert rubric_item_passed(claim_met=False, negated=True) is True


def test_extract_final_answer_reads_between_sentinels() -> None:
    text = 'thinking...\n<<FINAL_ANSWER>>\nThe cache is in foo.py.\n<<FINAL_ANSWER>>'
    assert extract_final_answer(text) == 'The cache is in foo.py.'


def test_extract_final_answer_without_tag_returns_text() -> None:
    assert extract_final_answer('  just prose, no sentinel  ') == 'just prose, no sentinel'
    assert extract_final_answer('') == ''


def _native_result(terminal_message: ChatMessageAssistant) -> SimpleNamespace:
    return SimpleNamespace(
        messages=[ChatMessageAssistant(content='I will inspect the repository.'), terminal_message],
        final_output=SimpleNamespace(message=terminal_message),
    )


@pytest.mark.parametrize(
    'tool_name,arguments',
    [('submit', {'answer': ''}), ('bash', {'command': 'pwd'})],
    ids=['empty_submit', 'max_steps_tool_call'],
)
def test_empty_native_answer_does_not_recover_earlier_narration(tmp_path, tool_name, arguments) -> None:
    adapter = make_adapter(tmp_path)
    call = ToolCall(id='terminal-call', function=ToolFunction(name=tool_name, arguments=arguments))
    result = _native_result(
        ChatMessageAssistant(
            content=[ContentReasoning(reasoning='Still exploring.'), ContentText(text='')],
            tool_calls=[call],
        )
    )

    answer = adapter._extract_final_answer(result, FunctionCallingStrategy())

    assert answer == ''


def test_native_answer_uses_terminal_prose(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    result = _native_result(
        ChatMessageAssistant(
            content=[
                ContentReasoning(reasoning='The evidence is complete.'),
                ContentText(text='<<FINAL_ANSWER>>\nThe cache is in foo.py.\n<<FINAL_ANSWER>>'),
            ]
        )
    )

    answer = adapter._extract_final_answer(result, FunctionCallingStrategy())

    assert answer == 'The cache is in foo.py.'


@pytest.mark.parametrize(
    'runner_stdout',
    [
        '{"type":"text","text":"opencode protocol output"}',
        '{"response":"gemini protocol output"}',
    ],
    ids=['opencode_json', 'gemini_json'],
)
def test_external_answer_uses_bridge_text_instead_of_json_stdout(tmp_path, runner_stdout: str) -> None:
    adapter = make_adapter(tmp_path)
    run_result = AgentRunResult(
        output=runner_stdout,
        final_answer='<<FINAL_ANSWER>>\nThe cache is in foo.py.\n<<FINAL_ANSWER>>',
    )

    answer = asyncio.run(adapter._external_extract_prediction(None, run_result, Sample(input='q')))

    assert answer == 'The cache is in foo.py.'


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------


def _record() -> Dict[str, Any]:
    return {
        'task_id': 'kitty-001',
        'prompt': 'Where is the key repeat handled?',
        'reference_answer': 'In keys.c',
        'repository_url': 'https://github.com/kovidgoyal/kitty',
        'repository_base_commit': 'abc123',
        'language': 'c',
        'category': 'codebase_qa',
        'docker_image': 'ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_kovidgoyal_kitty_1.0',
        'rubric': '[{"id": "r1", "title": "names keys.c", "annotations": {"type": "positive hli verifier"}}]',
    }


def test_record_to_sample_carries_task_metadata(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    sample = adapter.record_to_sample(_record())
    assert sample.input == 'Where is the key repeat handled?'
    assert sample.metadata['task_id'] == 'kitty-001'
    assert sample.metadata['docker_image'] == 'ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_kovidgoyal_kitty_1.0'
    assert sample.metadata['repository_base_commit'] == 'abc123'
    assert sample.metadata['rubric'][0]['id'] == 'r1'
    assert sample.metadata['rubric'][0]['negated'] is False


def test_build_environment_requires_base_commit_when_reset_is_enabled(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    sample = Sample(input='q', metadata={'task_id': 'missing-commit', 'docker_image': 'unused'})
    with pytest.raises(RuntimeError, match='repository_base_commit missing'):
        adapter.build_environment(sample)


def test_build_environment_uses_static_class_and_shared_typed_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(EnclaveAgentEnvironment, '__init__', lambda self, **kwargs: None)
    adapter = make_adapter(tmp_path)
    sample = adapter.record_to_sample(_record())

    env = adapter.build_environment(sample)

    assert isinstance(env, SweAtlasEnclaveEnvironment)
    state = sample.metadata['_sandbox_state']
    assert isinstance(state, SweAtlasSandboxState)
    assert env.state is state
    assert sample.model_dump()['metadata']['_sandbox_state']['setup_returncode'] is None


def test_bash_timeout_uses_benchmark_default_and_native_override(tmp_path) -> None:
    adapter = make_adapter(tmp_path, command_timeout=123)
    sample = Sample(input='q', tools=[BASH_TOOL_INFO])
    timeouts = []

    async def exec_command(cmd, **kwargs):
        timeouts.append(kwargs['timeout'])
        return SimpleNamespace(stdout='', stderr='', returncode=0, timed_out=False)

    env = SimpleNamespace(exec=exec_command)
    for agent_config, expected in [(None, 123), (NativeAgentConfig(command_timeout=45), 45)]:
        handlers, tools = adapter._resolve_tools(sample, agent_config)
        assert tools[0].parameters.properties['timeout'].default == expected
        call = ToolCall(id='1', function=ToolFunction(name='bash', arguments={'command': 'pwd'}))
        asyncio.run(handlers['bash'](call, env))

    assert timeouts == [123, 45]


# ---------------------------------------------------------------------------
# Scoring (reduce)
# ---------------------------------------------------------------------------


def _context(rubric: List[Dict[str, Any]], dirty: Optional[Dict[str, Any]] = None) -> JudgeContext:
    sandbox_state = dirty if dirty is not None else {'setup_returncode': 0, 'check_returncode': 0, 'tree_status': ''}
    sample = Sample(input='q', metadata={'prompt': 'q', 'rubric': rubric, '_sandbox_state': sandbox_state})
    task_state = TaskState(model='mock', sample=sample, output=None)
    return JudgeContext(task_state=task_state, original_prediction='a', filtered_prediction='a', reference='')


def _verdict(case_id: str, status: str, negated: bool) -> CaseVerdict:
    return CaseVerdict(
        case_id=case_id,
        value=RubricResponse(
            ratings=[
                RubricVerdict(
                    rubric_statement=case_id,
                    status=status,
                    score='1' if status == 'YES' else '0',
                    justification='',
                )
            ]
        ),
        metadata={'title': case_id, 'negated': negated},
    )


def test_reduce_resolves_when_every_rubric_item_passes(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}, {'id': 'r2', 'title': 't2', 'negated': True}]
    definition = adapter.judge_definition(_context(rubric))
    # r1 positive + YES -> pass; r2 negative + NO -> pass. All pass => resolved.
    verdicts = [_verdict('r1', 'YES', False), _verdict('r2', 'NO', True)]
    reduced = definition.reduce_verdicts(verdicts, _context(rubric))
    assert reduced.value['resolved'] == 1.0
    assert reduced.value['rubric_pass_rate'] == 1.0


def test_reduce_partial_pass_is_not_resolved(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}, {'id': 'r2', 'title': 't2', 'negated': False}]
    definition = adapter.judge_definition(_context(rubric))
    verdicts = [_verdict('r1', 'YES', False), _verdict('r2', 'NO', False)]
    reduced = definition.reduce_verdicts(verdicts, _context(rubric))
    assert reduced.value['resolved'] == 0.0
    assert reduced.value['rubric_pass_rate'] == 0.5
    assert reduced.metadata['failed_rubric_item_ids'] == ['r2']


def test_report_resolves_primary_metric_semantics(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    sample_score = SampleScore(
        sample_id='task-1',
        score=Score(value={'resolved': 0.0, 'rubric_pass_rate': 3 / 7}, main_score_name='resolved'),
    )
    scores = adapter.aggregate_scores([sample_score])

    report = ReportGenerator.generate_report({'default': scores}, 'mock', adapter)

    assert report.primary_metric_identity.name == 'resolve_rate'
    assert report.score == 0.0
    assert report.num == 1
    assert {metric.identity.name for metric in report.metrics} == {'resolve_rate', 'rubric_pass_rate'}


def test_reduce_fails_task_when_source_tree_is_dirty(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}]
    dirty = {'setup_returncode': 0, 'check_returncode': 0, 'tree_status': ' M keys.c'}
    definition = adapter.judge_definition(_context(rubric, dirty))
    verdicts = [_verdict('r1', 'YES', False)]
    reduced = definition.reduce_verdicts(verdicts, _context(rubric, dirty))
    # Rubric passed, but the dirty tree forces resolved=0 while keeping the rate.
    assert reduced.value['rubric_pass_rate'] == 1.0
    assert reduced.value['resolved'] == 0.0
    assert reduced.metadata['modified_source'] is True


@pytest.mark.parametrize(
    'sandbox_state',
    [
        {'setup_returncode': 1, 'setup_error': 'reset failed'},
        {'check_returncode': 1, 'check_error': 'status failed'},
        {'check_returncode': 0, 'tree_status': ''},
        {},
    ],
)
def test_reduce_excludes_unverified_environment(tmp_path, sandbox_state: Dict[str, Any]) -> None:
    adapter = make_adapter(tmp_path)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}]
    context = _context(rubric, sandbox_state)
    definition = adapter.judge_definition(context)
    reduced = definition.reduce_verdicts([_verdict('r1', 'YES', False)], context)
    assert reduced.value == {}
    assert reduced.metadata['environment_error']


def test_reduce_excludes_missing_reset_when_dirty_check_is_disabled(tmp_path) -> None:
    adapter = make_adapter(tmp_path, fail_on_source_modification=False)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}]
    context = _context(rubric, {})
    definition = adapter.judge_definition(context)
    reduced = definition.reduce_verdicts([_verdict('r1', 'YES', False)], context)
    assert reduced.value == {}
    assert reduced.metadata['environment_error'] == 'repository reset did not complete'


def test_reduce_allows_unverified_environment_when_both_checks_are_disabled(tmp_path) -> None:
    adapter = make_adapter(tmp_path, reset_to_base_commit=False, fail_on_source_modification=False)
    rubric = [{'id': 'r1', 'title': 't1', 'negated': False}]
    context = _context(rubric, {})
    definition = adapter.judge_definition(context)
    reduced = definition.reduce_verdicts([_verdict('r1', 'YES', False)], context)
    assert reduced.value == {'resolved': 1.0, 'rubric_pass_rate': 1.0}


def test_judge_prompt_does_not_request_fields_forbidden_by_the_contract() -> None:
    assert 'criterion_id' not in JUDGE_SYSTEM_PROMPT


def test_official_ratings_envelope_parses_through_output_contract() -> None:
    reply = (
        '{"ratings":[{"rubric_statement":"names keys.c","status":"YES","score":"1",'
        '"justification":"The answer names keys.c."}]}'
    )
    result = RUBRIC_CONTRACT.parse(reply)
    assert result.ok
    assert result.value.ratings[0].status == 'YES'


@pytest.mark.parametrize(
    'reply',
    [
        '{"status":"YES","justification":"missing envelope"}',
        '{"ratings":[]}',
        '{"ratings":[{"rubric_statement":"a","status":"YES","score":"1","justification":"a"}],"extra":true}',
        '{"ratings":[{"rubric_statement":"a","status":"YES","score":"0","justification":"a"}]}',
        '{"ratings":[{"rubric_statement":"a","status":"NO","score":"1","justification":"a"}]}',
        '{"ratings":[{"criterion_id":"1.2","rubric_statement":"a","status":"YES","score":"1","justification":"a"}]}',
        '{"ratings":['
        '{"rubric_statement":"a","status":"YES","score":"1","justification":"a"},'
        '{"rubric_statement":"b","status":"NO","score":"0","justification":"b"}'
        ']}',
    ],
)
def test_ratings_contract_rejects_wrong_shape(reply: str) -> None:
    assert not RUBRIC_CONTRACT.parse(reply).ok


def test_sandbox_commands_stop_on_failure_and_preserve_stderr() -> None:
    reset = reset_repo_command('/app', 'abc123')
    check = dirty_check_command('/app')
    assert ';' not in reset
    assert ';' not in check
    assert '2>/dev/null' not in reset
    assert '2>/dev/null' not in check
    assert check == 'cd /app && git status --porcelain'


class _FakeEnvironment:
    def __init__(self, results: List[Any]) -> None:
        self.results = list(results)
        self.commands: List[Any] = []
        self.closed = False

    async def exec(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
        self.commands.append(cmd)
        return self.results.pop(0)

    async def close(self) -> None:
        self.closed = True


def _sandbox_environment(results: List[Any], state: SweAtlasSandboxState):
    environment_cls = type('TestSweAtlasEnvironment', (SweAtlasSandboxMixin, _FakeEnvironment), {})
    return environment_cls(
        setup_command='reset', check_command='check', state=state, results=results
    )


def test_setup_failure_stops_agent_and_skips_dirty_check() -> None:
    state = SweAtlasSandboxState()
    env = _sandbox_environment(
        [SimpleNamespace(returncode=1, stdout='', stderr='unknown revision')], state
    )
    with pytest.raises(RuntimeError, match='unknown revision'):
        asyncio.run(env.exec(['bash']))
    asyncio.run(env.close())
    assert env.commands == [['bash', '-c', 'reset']]
    assert state.setup_returncode == 1
    assert state.setup_error == 'unknown revision'
    assert env.closed


def test_bash_tool_terminates_agent_when_setup_fails(tmp_path) -> None:
    adapter = make_adapter(tmp_path)
    state = SweAtlasSandboxState()
    env = _sandbox_environment(
        [SimpleNamespace(returncode=1, stdout='', stderr='unknown revision')], state
    )
    handler = adapter.build_tools(Sample(input='q'))['bash']
    call = ToolCall(id='1', function=ToolFunction(name='bash', arguments={'command': 'pwd'}))

    result = asyncio.run(handler(call, env))
    asyncio.run(env.close())

    assert result.terminate is True
    assert result.final_answer == ''
    assert result.text == 'unknown revision'
    assert env.commands == [['bash', '-c', 'reset']]


def test_dirty_check_nonzero_returncode_is_an_error() -> None:
    state = SweAtlasSandboxState()
    env = _sandbox_environment(
        [
            SimpleNamespace(returncode=0, stdout='', stderr=''),
            SimpleNamespace(returncode=0, stdout='agent output', stderr=''),
            SimpleNamespace(returncode=128, stdout='', stderr='not a git repository'),
        ],
        state,
    )
    asyncio.run(env.exec(['bash']))
    asyncio.run(env.close())
    assert state.check_returncode == 128
    assert state.check_error == 'not a git repository'
    assert state.tree_status is None
