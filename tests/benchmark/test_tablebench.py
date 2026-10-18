import json
from pathlib import Path
from typing import Any, Dict

import pytest

import evalscope  # noqa: F401  # imported for benchmark and metric registration side effects
from evalscope.api.benchmark import DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.tablebench.utils import exact_match, metric_for, parse_final_answer, rouge_l
from evalscope.config import TaskConfig


def _record(qtype: str, qsubtype: str, answer: str, instruction_type: str = 'TCoT') -> Dict[str, Any]:
    return {
        'id': 'x1',
        'qtype': qtype,
        'qsubtype': qsubtype,
        'table': "{'columns': ['a'], 'data': [[1]]}",
        'question': 'q?',
        'answer': answer,
        'instruction_type': instruction_type,
        'instruction': f'You are a table analyst... [{instruction_type}]\nQuestion: q?',
    }


def _adapter(**extra: Any) -> DefaultDataAdapter:
    args = {'tablebench': {'extra_params': extra}} if extra else {}
    return get_benchmark('tablebench', TaskConfig(datasets=['tablebench'], dataset_args=args))


def _score(adapter: DefaultDataAdapter, sample: Sample, completion: str) -> float:
    state = TaskState(
        model='mock',
        sample=sample,
        messages=[],
        output=ModelOutput.from_content(model='mock', content=completion),
        completed=True,
    )
    return adapter.calculate_metrics(state).score.value['mix_metric']


def test_registered_meta() -> None:
    adapter = _adapter()

    assert adapter.subset_list == ['FactChecking', 'NumericalReasoning', 'DataAnalysis']
    assert adapter.eval_split == 'test'
    assert adapter.metric_list == ['mix_metric']
    assert adapter.instruction_type == 'TCoT'


def test_invalid_instruction_type() -> None:
    with pytest.raises(ValueError, match='not in choices'):
        _adapter(instruction_type='PoT')


def test_load_reads_the_selected_style_file(tmp_path: Path) -> None:
    for style in ('TCoT', 'DP'):
        rows = [
            _record('FactChecking', 'MatchBased', 'Yes', style),
            _record('NumericalReasoning', 'Counting', '3', style),
            _record('Visualization', 'ChartGeneration', 'y_references = [[1]]', style),
        ]
        (tmp_path / f'TableBench_{style}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    adapter = get_benchmark(
        'tablebench',
        TaskConfig(
            datasets=['tablebench'],
            dataset_args={'tablebench': {'dataset_id': str(tmp_path), 'extra_params': {'instruction_type': 'DP'}}},
        ),
    )
    data = adapter.load_dataset()

    assert {k: len(v) for k, v in data.items()} == {'FactChecking': 1, 'NumericalReasoning': 1, 'DataAnalysis': 0}
    sample = data['FactChecking'][0]
    assert sample.metadata['instruction_type'] == 'DP'
    assert sample.input[-1].content.endswith('[DP]\nQuestion: q?')


def test_parse_uses_first_final_answer_line() -> None:
    assert parse_final_answer('Thinking...\nFinal Answer: 12, 13\nmore\nFinal Answer: 99') == '12, 13'
    assert parse_final_answer('no marker') == ''


@pytest.mark.parametrize(
    'qtype, qsubtype, metric',
    [
        ('FactChecking', 'MatchBased', 'em'),
        ('NumericalReasoning', 'Aggregation', 'em'),
        ('DataAnalysis', 'ImpactAnalysis', 'em'),
        ('DataAnalysis', 'CorrelationAnalysis', 'em_tolerance_10'),
        ('DataAnalysis', 'TrendForecasting', 'em_tolerance_10'),
        ('DataAnalysis', 'StatisticalAnalysis', 'em_tolerance_10'),
        ('DataAnalysis', 'CausalAnalysis', 'rouge_l'),
        ('DataAnalysis', 'DescriptiveAnalysis', 'rouge_l'),
    ],
)
def test_metric_assignment(qtype: str, qsubtype: str, metric: str) -> None:
    assert metric_for(qtype, qsubtype) == metric


def test_exact_match_follows_official_normalization() -> None:
    assert exact_match('10.6', '10.6') == 1.0
    assert exact_match('The Marian.', 'marian') == 1.0
    # normalization removes the decimal point, so these differ
    assert exact_match('10.60', '10.6') == 0.0
    # and the minus sign, so these match (official behaviour)
    assert exact_match('-5', '5') == 1.0
    # commas are punctuation too, so a multi-answer reference is one string
    assert exact_match('x, y', 'x, y') == 1.0
    assert exact_match('y, x', 'x, y') == 0.0


def test_exact_match_with_tolerance() -> None:
    assert exact_match('0.84', '0.85', tolerance=0.1) == 1.0
    assert exact_match('0.70', '0.85', tolerance=0.1) == 0.0
    assert exact_match('0', '0', tolerance=0.1) == 1.0
    assert exact_match('n/a', '0.85', tolerance=0.1) == 0.0


def test_rouge_l() -> None:
    assert rouge_l('the sales increased steadily', 'Sales increased steadily.') == pytest.approx(1.0)
    assert 0.0 < rouge_l('sales went up', 'sales increased steadily') < 1.0
    assert rouge_l('', 'sales increased') == 0.0


def test_scoring_end_to_end() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(_record('NumericalReasoning', 'Aggregation', '10.6'))

    assert _score(adapter, sample, 'The mean is 106/10.\nFinal Answer: 10.6') == 1.0
    assert _score(adapter, sample, 'Final Answer: 11') == 0.0
    assert _score(adapter, sample, 'The answer is 10.6') == 0.0  # no Final Answer line


def test_aggregation_is_mean_of_per_question_scores() -> None:
    adapter = _adapter()
    em_sample = adapter.record_to_sample(_record('FactChecking', 'MatchBased', 'yes'))
    rouge_sample = adapter.record_to_sample(_record('DataAnalysis', 'CausalAnalysis', 'price drives demand'))
    scores = [
        SampleScore(score=_state_score(adapter, em_sample, 'Final Answer: yes'), sample_id=0),
        SampleScore(score=_state_score(adapter, em_sample, 'Final Answer: no'), sample_id=1),
        SampleScore(score=_state_score(adapter, rouge_sample, 'Final Answer: price drives demand'), sample_id=2),
        SampleScore(score=_state_score(adapter, rouge_sample, 'Final Answer: unrelated'), sample_id=3),
    ]
    aggregates = {agg.metric_name: agg for agg in adapter.aggregate_scores(scores)}

    assert aggregates['mix_metric'].score == pytest.approx(0.5)
    assert aggregates['mix_metric'].num == 4


def _state_score(adapter: DefaultDataAdapter, sample: Sample, completion: str):
    state = TaskState(
        model='mock',
        sample=sample,
        messages=[],
        output=ModelOutput.from_content(model='mock', content=completion),
        completed=True,
    )
    return adapter.calculate_metrics(state).score
