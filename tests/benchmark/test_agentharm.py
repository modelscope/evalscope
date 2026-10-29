"""Offline AgentHarm data, tool and adapter regression tests."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.evaluator.cache import ModelResult
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction
from evalscope.benchmarks.agentharm.agentharm_adapter import AgentHarmAdapter
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus
from evalscope.third_party.agentharm.content import tool_function, tool_info
from evalscope.third_party.agentharm.resources import MANIFEST, resource_path
from evalscope.third_party.agentharm.types import AgentHarmSampleMeta
from tests.api.judge.test_migrated_adapters import ScriptedJudge

FIXTURES = Path(__file__).parent / 'fixtures/agentharm'


def make_adapter(name: str = 'agentharm', **overrides: Any) -> AgentHarmAdapter:
    args = {'local_path': str(FIXTURES), **overrides.pop('dataset_args', {})}
    config = TaskConfig(
        model='agentharm-scripted',
        eval_type='mock_llm',
        datasets=[name],
        judge=overrides.pop('judge', {'models': [{'model_id': 'scripted', 'eval_type': 'mock_llm'}]}),
        dataset_args={name: args},
        **overrides,
    )
    return get_benchmark(name, config)


def task_state(adapter: Any, behavior_id: str, messages: list) -> TaskState:
    adapter._benchmark_meta.extra_params['behavior_ids']['value'] = [behavior_id]
    data = adapter.load_dataset()
    sample = next(sample for dataset in data.values() for sample in dataset)
    output = ModelOutput.from_content(
        'scripted', messages[-1].text if messages and messages[-1].role == 'assistant' else ''
    )
    return TaskState(model='scripted', sample=sample, messages=sample.input + messages, output=output, completed=True)


def tool_turn(
    name: str, arguments: dict, *, call_id: str = 'call', error: bool = False, observation: str = 'ok'
) -> list:
    call = ToolCall(id=call_id, function=ToolFunction(name=name, arguments=arguments))
    return [
        ChatMessageAssistant(content='', tool_calls=[call]),
        ChatMessageTool(
            content=observation,
            function=name,
            tool_call_id=call_id,
            error=ToolCallError(type='parsing', message='rejected') if error else None,
        ),
    ]


def rule_score(
    adapter: Any, behavior_id: str, messages: list, reply: str = '{"verdict":"pass","refusal":false}'
) -> Any:
    adapter.llm_judge = ScriptedJudge([reply])
    return adapter.calculate_metrics(task_state(adapter, behavior_id, messages)).score


@pytest.mark.parametrize('name', ['agentharm', 'agentharm_benign'])
@pytest.mark.parametrize('split', ['test_public', 'val'])
def test_native_data_loading_keeps_identity_and_standard_limits(name: str, split: str) -> None:
    adapter = make_adapter(name, limit=1, repeats=2, dataset_args={'eval_split': split})
    datasets = adapter.load_dataset()
    assert all(len(dataset) in (0, 2) for dataset in datasets.values())
    for category, dataset in datasets.items():
        for sample in dataset:
            meta = AgentHarmSampleMeta.model_validate(sample.metadata['agentharm'])
            assert meta.category == category and meta.split == split
            assert set(meta.tool_order) == set(meta.target_functions) == {t.name for t in sample.tools}
            assert sample.input[1].text.startswith('\n\n')
            assert len(meta.dataset_checksum) == 64
        if dataset:
            assert dataset[0] is not dataset[1]
            assert dataset[0].metadata == dataset[1].metadata


def test_ids_override_variant_filters_before_category_limit() -> None:
    adapter = make_adapter(
        limit=1,
        dataset_args={
            'subset_list': ['Disinformation'],
            'extra_params': {'behavior_ids': ['1-2'], 'detailed_behaviors': False, 'hint_included': False},
        },
    )
    assert adapter.load_dataset()['Disinformation'][0].metadata['agentharm']['behavior_id'] == '1-2'
    bad = make_adapter(dataset_args={'extra_params': {'behavior_ids': ['missing']}})
    with pytest.raises(ValueError, match='Unknown AgentHarm behavior IDs'):
        bad.load_dataset()


@pytest.mark.parametrize('ac', [{'tools': ['bash']}, {'strategy': 'react'}, {'kwargs': {'system_prompt': 'changed'}}])
def test_incompatible_agent_config_fails_before_requests(ac: dict) -> None:
    with pytest.raises(ValueError, match='native function_calling'):
        make_adapter(agent_config=ac)


@pytest.mark.parametrize(
    'judge',
    [
        {'models': []},
        {'models': [{'model_id': 'j'}], 'repeats': 2},
        {'models': [{'model_id': 'j'}], 'position_swap': 'on'},
    ],
)
def test_incompatible_judge_policy_fails_before_requests(judge: dict) -> None:
    with pytest.raises(ValueError, match='one judge.models'):
        make_adapter(judge=judge)


@pytest.mark.parametrize('mode', ['harmful', 'benign'])
def test_all_public_tool_schemas_and_bundled_reads(mode: str, monkeypatch: Any, tmp_path: Path) -> None:
    assert len(MANIFEST[mode]['tools']) == 80
    for name in MANIFEST[mode]['tools']:
        info = tool_info(mode, name)
        assert info.name == name and info.description
        assert set(info.parameters.required) <= set(info.parameters.properties)
    monkeypatch.chdir(tmp_path)
    read = tool_function(mode, 'read_file')
    result = asyncio.run(read(file='paper_abstract.txt'))
    expected = resource_path(mode, f'benchmark/{mode}_tools/content/paper_abstract.txt').read_text()
    assert expected.splitlines()[0] in result
    assert 'not a valid file' in asyncio.run(read(file='/etc/passwd'))
    convert = tool_function(mode, 'convert_pdf_to_text')
    assert 'PDF content from' in asyncio.run(convert(file_path='attention.pdf'))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('new_id', ['1-2', '2-1'])
def test_cache_identity_survives_metadata_replacement(new_id: str) -> None:
    old = make_adapter()
    old_state = task_state(old, '1-1', [ChatMessageAssistant(content='Done')])
    cache = ModelResult.from_task_state(old_state)
    current = make_adapter(dataset_args={'subset_list': ['Disinformation'], 'extra_params': {'behavior_ids': [new_id]}})
    sample_set = current.load_dataset()['Disinformation']
    restored = cache.to_task_state(sample_set)
    current.llm_judge = ScriptedJudge(['{"refusal": false}'])
    score = current.calculate_metrics(restored).score
    assert score.status is ScoreStatus.EXCLUDED and score.value == {}
    assert 'identity' in score.explanation and not current.llm_judge.calls


@pytest.mark.parametrize('reason', ['content_filter', 'model_length'])
def test_provider_termination_is_excluded(reason: str) -> None:
    adapter = make_adapter()
    state = task_state(adapter, '1-1', [ChatMessageAssistant(content='blocked')])
    state.output = ModelOutput.from_content('scripted', 'blocked', stop_reason=reason)
    adapter.llm_judge = ScriptedJudge(['{"refusal": false}'])
    score = adapter.calculate_metrics(state).score
    assert score.status is ScoreStatus.EXCLUDED and score.value == {}
    assert not adapter.llm_judge.calls
