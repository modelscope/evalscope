import asyncio
import json
import pytest
from collections import deque
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.benchmarks.researchrubrics.researchrubrics_adapter import ResearchRubricsAdapter
from evalscope.benchmarks.researchrubrics.utils import TemporaryLocalAgentEnvironment, chunk_document
from evalscope.config import TaskConfig
from evalscope.constants import JudgeStrategy


class FakeJudge:

    def __init__(self, responses: List[str]) -> None:
        self.responses = deque(responses)
        self.model_id = 'fake-judge'
        self.calls: List[Dict[str, str]] = []

    def judge(self, prompt: str = '', system_prompt: str = '', **kwargs: Any) -> str:
        self.calls.append({'prompt': prompt, 'system_prompt': system_prompt})
        if not self.responses:
            raise AssertionError('No fake judge response remaining.')
        return self.responses.popleft()


class ChunkAwareJudge:

    model_id = 'fake-chunk-judge'

    def __init__(self) -> None:
        self.calls: List[str] = []

    def judge(self, prompt: str = '', system_prompt: str = '', **kwargs: Any) -> str:
        self.calls.append(prompt)
        if 'large document in chunks' in prompt:
            return json.dumps({
                'relevant_evidence': ['evidence'],
                'satisfaction': True,
                'confidence_for_chunk': 0.9,
                'notes': 'found',
            })
        return binary_response('Satisfied', 1.0)


def binary_response(verdict: str, score: float) -> str:
    return json.dumps({
        'verdict': verdict,
        'score': score,
        'confidence': 0.9,
        'reasoning': 'reason',
        'evidence_quotes': ['quote'],
        'missing_elements': [],
    })


def make_adapter(**extra_params: Any) -> ResearchRubricsAdapter:
    dataset_args = {'researchrubrics': {'extra_params': extra_params}} if extra_params else {}
    config = TaskConfig(
        model='mock-model',
        datasets=['researchrubrics'],
        dataset_args=dataset_args,
        judge_strategy=JudgeStrategy.LLM,
        judge_model_args={
            'model_id': 'fake-judge',
            'api_url': 'http://localhost:1/v1',
            'api_key': 'fake-key',
        },
        eval_batch_size=1,
    )
    adapter = get_benchmark('researchrubrics', config)
    assert isinstance(adapter, ResearchRubricsAdapter)
    return adapter


def make_state(adapter: ResearchRubricsAdapter, rubrics: List[Dict[str, Any]], report: str = 'Report') -> TaskState:
    sample = adapter.record_to_sample({
        'prompt': 'Research the topic.',
        'sample_id': 'sample-1',
        'domain': 'Other',
        'conceptual_breadth': 'Simple',
        'logical_nesting': 'Shallow',
        'exploration': 'Low',
        'rubrics': rubrics,
    })
    sample.id = 0
    sample.group_id = 0
    output = ModelOutput(
        model='mock-model',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=report))],
    )
    return TaskState(model='mock-model', sample=sample, output=output, completed=True)


def test_researchrubrics_registration_and_sample_conversion() -> None:
    adapter = make_adapter()

    sample = adapter.record_to_sample({
        'prompt': 'Research the topic.',
        'sample_id': 'source-id',
        'domain': 'AI & ML',
        'conceptual_breadth': 'Moderate',
        'logical_nesting': 'Intermediate',
        'exploration': 'High',
        'rubrics': [{'criterion': 'Cites sources', 'weight': 5.0, 'axis': 'References & Citation Quality'}],
    })

    assert isinstance(adapter, AgentLoopAdapter)
    assert adapter.dataset_id == 'evalscope/researchrubrics'
    assert adapter.strategy_name == 'function_calling'
    assert adapter.max_steps == 50
    assert adapter.use_batch_scoring is True
    assert sample.input == 'Research the topic.'
    assert 'Cites sources' not in sample.input
    assert json.loads(sample.target)[0]['criterion'] == 'Cites sources'
    assert sample.metadata['sample_id'] == 'source-id'
    assert [tool.name for tool in sample.tools] == ['bash']


def test_react_strategy_is_configurable() -> None:
    adapter = make_adapter(strategy='react')
    strategy = adapter.build_strategy(Sample(input='prompt'))

    assert adapter.strategy_name == 'react'
    assert strategy.name == 'react'


def test_temporary_local_environment_cleans_working_directory() -> None:
    environment = TemporaryLocalAgentEnvironment(sample_id='sample')
    working_dir = environment.working_dir
    (working_dir / 'artifact.txt').write_text('data', encoding='utf-8')

    asyncio.run(environment.close())

    assert not working_dir.exists()


def test_default_agent_loop_runs_bash_without_agent_config() -> None:
    adapter = make_adapter(max_steps=2)
    bash = ToolCall(id='bash-1', function=ToolFunction(name='bash', arguments={'command': 'pwd'}))
    submit = ToolCall(id='submit-1', function=ToolFunction(name='submit', arguments={'answer': '# Report'}))
    bash_output = ModelOutput(
        model='mock-model',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content='', tool_calls=[bash]))],
    )
    submit_output = ModelOutput(
        model='mock-model',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content='', tool_calls=[submit]))],
    )
    model = AsyncMock()
    model.name = 'mock-model'
    model.generate_async.side_effect = [bash_output, submit_output]
    sample = Sample(id=0, input='Research this.', target='[]', tools=[adapter.record_to_sample({
        'prompt': 'p',
        'sample_id': 'source-id',
        'rubrics': [],
    }).tools[0]])

    result = adapter._on_inference(model, sample)

    assert result.output.completion == '# Report'
    assert result.trace.strategy == 'function_calling'
    assert result.trace.environment == 'local'
    assert result.trace.max_steps == 2
    tool_message = next(message for message in result.messages if message.role == 'tool')
    working_dir = Path(tool_message.text.strip())
    assert 'evalscope-researchrubrics-' in working_dir.name
    assert not working_dir.exists()


def test_max_steps_requests_a_tool_free_final_report() -> None:
    adapter = make_adapter(max_steps=2)
    bash = ToolCall(id='bash-1', function=ToolFunction(name='bash', arguments={'command': 'echo evidence'}))
    tool_output = ModelOutput(
        model='mock-model',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content='', tool_calls=[bash]))],
    )
    final_output = ModelOutput(
        model='mock-model',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content='# Final report'))],
    )
    model = AsyncMock()
    model.name = 'mock-model'
    model.generate_async.side_effect = [tool_output, tool_output]
    model.generate = MagicMock(return_value=final_output)
    sample = adapter.record_to_sample({
        'prompt': 'Research this.',
        'sample_id': 'source-id',
        'rubrics': [],
    })
    sample.id = 0

    result = adapter._on_inference(model, sample)

    assert result.output.completion == '# Final report'
    assert model.generate.call_args.kwargs['tools'] is None
    assert 'tool-use budget is exhausted' in model.generate.call_args.kwargs['input'][-1].text
    assert result.trace.events[-1].type.value == 'submit'
    assert result.trace.events[-1].payload['phase'] == 'max_steps_finalization'


def test_binary_scoring_preserves_negative_weight_penalty() -> None:
    adapter = make_adapter()
    adapter._llm_judge = FakeJudge([
        binary_response('Satisfied', 1.0),
        binary_response('Not Satisfied', 0.0),
        binary_response('Satisfied', 1.0),
    ])
    state = make_state(adapter, [
        {'criterion': 'Includes the required answer', 'weight': 5.0, 'axis': 'Explicit Criteria'},
        {'criterion': 'Provides extra context', 'weight': 3.0, 'axis': 'Implicit Criteria'},
        {'criterion': 'Contains a factual error', 'weight': -2.0, 'axis': 'Explicit Criteria'},
    ])

    score = adapter._score_task_state(state)

    assert score.value['compliance_score'] == pytest.approx(3 / 8)
    assert score.value['axis/Explicit Criteria'] == pytest.approx(3 / 5)
    assert score.value['axis/Implicit Criteria'] == 0.0
    assert score.main_score_name == 'compliance_score'
    assert score.metadata['grading_mode'] == 'binary'
    assert len(score.metadata['rubrics']) == 3
    assert 'Do not invert the binary mapping' in adapter._llm_judge.calls[-1]['prompt']


def test_judge_retries_parse_errors(monkeypatch: Any) -> None:
    adapter = make_adapter(judge_retries=3)
    adapter._llm_judge = FakeJudge(['not json', '{}', binary_response('Satisfied', 1.0)])
    monkeypatch.setattr('evalscope.benchmarks.researchrubrics.researchrubrics_adapter.time.sleep', lambda _: None)

    result = adapter._request_json(
        prompt='prompt',
        system_prompt='system',
        validator=lambda data: data if data.get('verdict') == 'Satisfied' else (_ for _ in ()).throw(ValueError()),
        context='test',
    )

    assert result['verdict'] == 'Satisfied'
    assert len(adapter._llm_judge.calls) == 3


def test_judge_failure_raises_after_retries(monkeypatch: Any) -> None:
    adapter = make_adapter(judge_retries=2)
    adapter._llm_judge = FakeJudge(['bad', 'still bad'])
    monkeypatch.setattr('evalscope.benchmarks.researchrubrics.researchrubrics_adapter.time.sleep', lambda _: None)

    with pytest.raises(RuntimeError, match='after 2 attempts'):
        adapter._request_json('prompt', 'system', lambda data: data, 'test')


def test_long_report_uses_chunk_and_synthesis() -> None:
    adapter = make_adapter(judge_context_limit=1, judge_chunk_size=1)
    judge = ChunkAwareJudge()
    adapter._llm_judge = judge
    state = make_state(
        adapter,
        [{'criterion': 'Uses evidence', 'weight': 5.0, 'axis': 'Synthesis of Information'}],
        report='abcdefghij',
    )

    score = adapter._score_task_state(state)

    assert score.value['compliance_score'] == 1.0
    assert score.metadata['used_chunking'] is True
    assert len(judge.calls) == len(chunk_document('abcdefghij', max_tokens=1)) + 1
    assert 'Evidence points' in judge.calls[-1]


def test_calculate_metrics_returns_placeholder_for_two_phase_review() -> None:
    adapter = make_adapter()
    state = make_state(adapter, [{'criterion': 'Criterion', 'weight': 1.0, 'axis': 'Explicit Criteria'}])

    sample_score = adapter.calculate_metrics(state)

    assert sample_score.score.value == {}
    assert sample_score.score.prediction == 'Report'


@pytest.mark.parametrize('judge_strategy', [JudgeStrategy.RULE, JudgeStrategy.LLM_RECALL])
def test_rejects_unsupported_judge_strategies(judge_strategy: str) -> None:
    adapter = make_adapter()
    adapter._task_config.judge_strategy = judge_strategy

    with pytest.raises(ValueError, match='requires judge_strategy'):
        adapter._validate_judge_config()


def test_aggregate_scores_outputs_diagnostic_dimensions() -> None:
    adapter = make_adapter()
    sample_scores = [
        SampleScore(
            sample_id=0,
            score=Score(value={'compliance_score': 0.5, 'axis/Explicit Criteria': 0.6}),
            sample_metadata={
                'domain': 'AI & ML',
                'conceptual_breadth': 'Moderate',
                'logical_nesting': 'Intermediate',
                'exploration': 'High',
            },
        ),
        SampleScore(
            sample_id=1,
            score=Score(value={'compliance_score': -0.1}),
            sample_metadata={
                'domain': 'Other',
                'conceptual_breadth': 'Simple',
                'logical_nesting': 'Shallow',
                'exploration': 'Low',
            },
        ),
    ]

    scores = adapter.aggregate_scores(sample_scores)
    by_name = {score.metric_name: score for score in scores}

    assert scores[0].metric_name == 'compliance_score'
    assert by_name['compliance_score'].score == pytest.approx(0.2)
    assert by_name['axis/Explicit Criteria'].score == 0.6
    assert by_name['axis/Explicit Criteria'].num == 1
    assert by_name['domain/AI & ML'].score == 0.5
    assert by_name['domain/Other'].score == -0.1
    assert by_name['conceptual_breadth/Moderate'].num == 1
