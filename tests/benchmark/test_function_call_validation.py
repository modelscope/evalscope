"""Tool payload errors must survive parsing, caching and schema-score aggregation."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessage

from evalscope.api.benchmark import FunctionCallAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.metric import SampleScore
from evalscope.api.model import ChatCompletionChoice, ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.models.utils.openai import chat_tool_calls_from_openai

BENCHMARKS = ['general_fc', 'k2_verifier', 'minimax_verifier']
CASES = [
    pytest.param(['{}'], 1, id='valid-optional-empty'),
    pytest.param(['{"query": "weather"}'], 1, id='valid-argument'),
    pytest.param(['{"query": 123}'], 0, id='wrong-type'),
    pytest.param(['{"query":'], 0, id='truncated-json'),
    pytest.param(['{"query": "weather",}'], 0, id='malformed-json'),
    pytest.param(['{}', '{"query":'], 0, id='mixed-parallel-calls'),
    pytest.param([], 0, id='empty-call-list'),
    pytest.param(None, 0, id='missing-call-list'),
]


@pytest.fixture(params=BENCHMARKS)
def adapter(request: pytest.FixtureRequest) -> FunctionCallAdapter:
    name = request.param
    return get_benchmark(name, TaskConfig(model='mock', datasets=[name]))


def make_state(
    adapter: FunctionCallAdapter,
    arguments: list[str] | None,
    *,
    required: bool = False,
    stop_reason: str = 'tool_calls',
    expected_call: bool = True,
    cached_output: bool = False,
    sample_id: int = 0,
    tools: list[dict[str, Any]] | None = None,
) -> TaskState:
    tools = tools if tools is not None else [{
        'type': 'function', 'function': {
            'name': 'search', 'description': 'Search with an optional query.',
            'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                           'required': ['query'] if required else []},
        },
    }]
    sample = adapter.record_to_sample({
        'messages': [{'role': 'user', 'content': 'Search for something.'}],
        'tools': tools, 'should_call_tool': expected_call, 'expected_tool_call': expected_call,
    })
    sample.id = sample_id
    raw_message = ChatCompletionMessage.model_validate({
        'role': 'assistant', 'content': None,
        'tool_calls': None if arguments is None else [
            {'id': f'call-{i}', 'type': 'function', 'function': {
                'name': tools[0]['function']['name'], 'arguments': value,
            }} for i, value in enumerate(arguments)
        ],
    })
    calls = chat_tool_calls_from_openai(raw_message, sample.tools)
    message = ChatMessageAssistant(content='', tool_calls=calls)
    output = ModelOutput(model='mock', choices=[ChatCompletionChoice(message=message, stop_reason=stop_reason)])
    if cached_output:
        output = ModelOutput.model_validate_json(output.model_dump_json())
    return TaskState(model='mock', sample=sample, output=output, completed=True)


def schema_score(adapter: FunctionCallAdapter, scores: list[SampleScore]) -> float:
    return next(item.score for item in adapter.aggregate_scores(scores) if item.metric_name == 'schema_accuracy')


@pytest.mark.parametrize(('arguments', 'expected'), CASES)
@pytest.mark.parametrize('cached_output', [False, True], ids=['parsed', 'cached-output'])
def test_tool_payload_validity(
    adapter: FunctionCallAdapter, arguments: list[str] | None, expected: int, cached_output: bool
) -> None:
    state = make_state(adapter, arguments, cached_output=cached_output)
    score = adapter.calculate_metrics(state)
    assert schema_score(adapter, [score]) == expected
    if adapter.name != 'minimax_verifier':
        assert score.score.value['successful_tool_call'] == expected
        assert score.score.value['finish_reason_tool_call'] == 1
        if not expected:
            assert score.score.metadata['error_reason']
    else:
        assert score.score.value['tool_calls_schema_valid'] == expected
        assert score.score.value['tool_calls_match'] == 1
    if adapter.name == 'general_fc':
        assert score.score.value['passed'] == bool(expected)
    calls = state.output.message.tool_calls or []
    if arguments and any(value.endswith((':', ',}')) for value in arguments):
        assert any(call.parse_error for call in calls)


@pytest.mark.parametrize(('arguments', 'expected'), [(['{}'], 0), (['{"query": "weather"}'], 1)])
def test_required_arguments_remain_enforced(
    adapter: FunctionCallAdapter, arguments: list[str], expected: int
) -> None:
    score = adapter.calculate_metrics(make_state(adapter, arguments, required=True))
    assert schema_score(adapter, [score]) == expected


def test_normal_no_call_is_not_a_validation_error(adapter: FunctionCallAdapter) -> None:
    score = adapter.calculate_metrics(make_state(adapter, None, stop_reason='stop', expected_call=False))
    assert not (score.score.metadata or {}).get('error_reason')
    assert schema_score(adapter, [score]) == 0
    if adapter.name == 'general_fc':
        assert score.score.value['passed'] is True
    elif adapter.name == 'minimax_verifier':
        assert score.score.value['tool_calls_match'] == 1
    else:
        assert score.score.value['finish_reason_tool_call'] == 0


def test_inference_error_cannot_become_a_valid_call(adapter: FunctionCallAdapter) -> None:
    state = make_state(adapter, ['{}'])
    state.output.error = 'provider failed'
    assert schema_score(adapter, [adapter.calculate_metrics(state)]) == 0


@pytest.mark.parametrize('mode', ['serial', 'concurrent', 'cached-reviews'])
def test_mixed_run_scores_and_reports_only_valid_calls(adapter: FunctionCallAdapter, mode: str) -> None:
    arguments = [['{}'], ['{"query": "weather"}'], ['{"query": 123}'], ['{"query":'], [], ['{}', '{"query":']]
    states = [make_state(adapter, value, sample_id=i) for i, value in enumerate(arguments)]
    if mode == 'concurrent':
        with ThreadPoolExecutor(max_workers=4) as pool:
            scores = list(pool.map(adapter.calculate_metrics, states))
    else:
        scores = [adapter.calculate_metrics(state) for state in states]
    if mode == 'cached-reviews':
        scores = [SampleScore.model_validate_json(score.model_dump_json()) for score in reversed(scores)]
    assert schema_score(adapter, scores) == pytest.approx(2 / 6)
    aggregates = adapter.aggregate_scores(scores)
    report = adapter.generate_report({'default': aggregates}, model_name='mock', output_dir='')
    serialized = report.model_dump_json()
    from evalscope.report.report import Report

    restored = Report.model_validate_json(serialized)
    metric = next(metric for metric in restored.metrics if metric.identity.name == 'schema_accuracy')
    assert metric.score == pytest.approx(round(2 / 6, 4))
    # All six responses declare a tool call; payload validity must not change this decision metric.
    assert report.score == 1.0


def test_evaluation_version_changes_with_scoring(adapter: FunctionCallAdapter) -> None:
    assert adapter.benchmark_meta.evaluation_version == 'v1.1'


@pytest.mark.parametrize(('arguments', 'expected'), [
    (['{"command": ["ls", "-l"]}'], 1),
    (['{"command": ["ls -l"]}'], 0),
])
def test_minimax_command_array_check_is_preserved(arguments: list[str], expected: int) -> None:
    adapter = get_benchmark('minimax_verifier', TaskConfig(model='mock', datasets=['minimax_verifier']))
    tools = [{'type': 'function', 'function': {
        'name': 'execute', 'description': 'Execute an array command.', 'parameters': {
            'type': 'object', 'properties': {'command': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['command'],
        },
    }}]
    assert schema_score(adapter, [adapter.calculate_metrics(make_state(adapter, arguments, tools=tools))]) == expected
