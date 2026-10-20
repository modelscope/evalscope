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
from evalscope.benchmarks.tat_qa.utils import parse_response, score_question
from evalscope.config import TaskConfig


def _qa(answer: Any, answer_type: str, scale: str = '', answer_from: str = 'table', uid: str = 'q1') -> Dict[str, Any]:
    return {
        'uid': uid,
        'order': 1,
        'question': 'What is the value?',
        'answer': answer,
        'answer_type': answer_type,
        'answer_from': answer_from,
        'scale': scale,
    }


def _record(qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'table': [['', '2019', '2018'], ['Revenue', '1,200', '1,000']],
        'paragraphs': [{'uid': 'p2', 'order': 2, 'text': 'Second.'}, {'uid': 'p1', 'order': 1, 'text': 'First.'}],
        **qa,
    }


def _adapter() -> DefaultDataAdapter:
    return get_benchmark('tat_qa', TaskConfig(datasets=['tat_qa']))


def _state_score(adapter: DefaultDataAdapter, sample: Sample, completion: str):
    state = TaskState(
        model='mock',
        sample=sample,
        messages=[],
        output=ModelOutput.from_content(model='mock', content=completion),
        completed=True,
    )
    return adapter.calculate_metrics(state).score


def test_registered_meta() -> None:
    adapter = _adapter()

    assert adapter.subset_list == ['table', 'text', 'table-text']
    assert adapter.eval_split == 'test'
    assert adapter.metric_list == ['exact_match', 'f1']


def test_load_groups_questions_by_answer_source(tmp_path: Path) -> None:
    docs = [{
        'table': {'uid': 't1', 'table': [['a', 'b'], ['1', '2']]},
        'paragraphs': [{'uid': 'p1', 'order': 1, 'text': 'Some text.'}],
        'questions': [
            _qa(['x'], 'span', answer_from='text', uid='q1'),
            _qa(3, 'arithmetic', answer_from='table', uid='q2'),
            _qa(['a', 'b'], 'multi-span', answer_from='table', uid='q3'),
        ],
    }]
    (tmp_path / 'tatqa_dataset_test_gold.json').write_text(json.dumps(docs), encoding='utf-8')
    adapter = get_benchmark(
        'tat_qa', TaskConfig(datasets=['tat_qa'], dataset_args={'tat_qa': {'dataset_id': str(tmp_path)}})
    )
    data = adapter.load_dataset()

    assert {k: len(v) for k, v in data.items()} == {'table': 2, 'text': 1, 'table-text': 0}
    sample = data['text'][0]
    assert sample.metadata['uid'] == 'q1'
    assert 'Table:\na | b\n1 | 2\n\nParagraphs:\nSome text.' in sample.input[-1].content


def test_prompt_orders_paragraphs() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(_record(_qa(['First.'], 'span')))

    assert 'Revenue | 1,200 | 1,000' in sample.input
    assert 'Paragraphs:\nFirst.\nSecond.' in sample.input
    assert sample.input.endswith('Scale: <none, thousand, million, billion or percent>')


def test_parse_response() -> None:
    assert parse_response('work\nAnswer: 1,200\nScale: thousand') == (['1,200'], 'thousand')
    assert parse_response('Answer: a | b\nScale: none') == (['a', 'b'], '')
    assert parse_response('**Answer:** 5\n**Scale:** Million.') == (['5'], 'million')
    assert parse_response('Answer: first\nAnswer: 7\nScale: percent') == (['7'], 'percent')
    assert parse_response('Final Answer: 7') == (['7'], '')
    assert parse_response('Answer: 7\nScale: units') == (['7'], '')
    assert parse_response('no answer line') == ([], '')


def test_scale_is_applied_before_comparing_numbers() -> None:
    gold = _qa(1.2, 'arithmetic', scale='million')

    assert score_question(gold, ['1.2'], 'million') == (1.0, 1.0)
    assert score_question(gold, ['1,200,000'], '') == (1.0, 1.0)
    assert score_question(gold, ['1.2'], 'thousand') == (0.0, 0.0)
    assert score_question(gold, ['1.21'], 'million') == (0.0, 0.0)


def test_percent_answers() -> None:
    gold = _qa(23.42, 'arithmetic', scale='percent')

    assert score_question(gold, ['23.42'], 'percent') == (1.0, 1.0)
    assert score_question(gold, ['23.42%'], '') == (1.0, 1.0)
    assert score_question(gold, ['0.2342'], '') == (1.0, 1.0)


def test_spans() -> None:
    gold = _qa(['the modified retrospective method'], 'span')

    assert score_question(gold, ['The modified retrospective method.'], '') == (1.0, 1.0)
    em, f1 = score_question(gold, ['modified method'], '')
    assert em == 0.0 and 0.0 < f1 < 1.0
    assert score_question(gold, [], '') == (0.0, 0.0)


def test_multi_span_order_does_not_matter() -> None:
    gold = _qa(['2019', '2018'], 'multi-span')

    assert score_question(gold, ['2018', '2019'], '') == (1.0, 1.0)
    # as in the official script, the spans are joined into one string before the token F1
    assert score_question(gold, ['2019'], '') == (0.0, 0.67)


def test_scoring_end_to_end_and_aggregation() -> None:
    adapter = _adapter()
    number = adapter.record_to_sample(_record(_qa(200, 'arithmetic', scale='thousand')))
    span = adapter.record_to_sample(_record(_qa(['Revenue'], 'span')))
    scores = [
        SampleScore(score=_state_score(adapter, number, '1,200 - 1,000 = 200\nAnswer: 200\nScale: thousand'),
                    sample_id=0),
        SampleScore(score=_state_score(adapter, number, 'Answer: 200\nScale: million'), sample_id=1),
        SampleScore(score=_state_score(adapter, span, 'Answer: revenue\nScale: none'), sample_id=2),
        SampleScore(score=_state_score(adapter, span, 'It is revenue.'), sample_id=3),
    ]
    assert [s.score.value for s in scores] == [
        {'exact_match': 1.0, 'f1': 1.0},
        {'exact_match': 0.0, 'f1': 0.0},
        {'exact_match': 1.0, 'f1': 1.0},
        {'exact_match': 0.0, 'f1': 0.0},
    ]
    aggregates = {agg.metric_name: agg for agg in adapter.aggregate_scores(scores)}
    assert aggregates['exact_match'].score == pytest.approx(0.5)
    assert aggregates['f1'].num == 4
