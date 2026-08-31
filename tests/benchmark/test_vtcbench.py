from types import SimpleNamespace

import pytest

from evalscope.api.messages import ContentImage, ContentText
from evalscope.api.metric import SampleScore
from evalscope.benchmarks.vtcbench.vtcbench_adapter import (
    PROMPT_PREFIXES,
    PROMPT_SUFFIXES,
    VTCBenchAdapter,
    _calculate_metrics,
    _normalize_response,
    _remove_html_tags,
)


def _make_adapter(subset: str, eval_mode: str) -> VTCBenchAdapter:
    adapter = object.__new__(VTCBenchAdapter)
    adapter.current_subset_name = subset
    adapter.eval_mode = eval_mode
    adapter._missing_media_warned = set()
    adapter._benchmark_meta = SimpleNamespace(aggregation='mean')
    return adapter


def test_official_metric_normalization_and_dispatch() -> None:
    assert _normalize_response('<think>4075987</think> Wrong ') == 'wrong'

    metrics = _calculate_metrics('<think>4075987</think> 5943250', ['4075987', '5943250'])
    assert metrics['contains_any'] == 1.0
    assert metrics['contains_all'] == 0.5

    memory_metrics = _calculate_metrics('Adoption agencies', ['adoption agencies', 'agencies'])
    assert memory_metrics['rouge_l'] == 1.0

    adapter = _make_adapter('Memory', 'text')
    score = adapter.match_score(
        original_prediction='Adoption agencies',
        filtered_prediction='ignored',
        reference='ignored',
        task_state=SimpleNamespace(metadata={
            'subset': 'Memory',
            'answers': ['adoption agencies', 'agencies'],
        }),
    )
    assert score.value['score'] == 1.0
    assert score.main_score_name == 'score'

    aggregates = adapter.aggregate_scores([
        SampleScore(score=score, sample_id=0),
        SampleScore(score=score, sample_id=1),
    ])
    unified = next(item for item in aggregates if item.metric_name == 'normalized_score')
    assert unified.score == 1.0
    assert unified.num == 2


def test_empty_answers_are_rejected() -> None:
    with pytest.raises(ValueError, match='at least one reference answer'):
        _calculate_metrics('answer', [])


def test_text_mode_matches_official_context_order_and_html_cleanup() -> None:
    adapter = _make_adapter('Memory', 'text')
    record = {
        'problem': 'What did Evan buy?',
        'answers': ['a Prius'],
        '_context': "<div><span data-speaker='Evan'>I bought a Prius.</span></div>\n<image 1>",
    }

    sample = adapter.record_to_sample(record)
    content = sample.input[0].content

    assert all(isinstance(item, ContentText) for item in content)
    assert content[0].text == _remove_html_tags(record['_context'])
    assert content[-1].text == PROMPT_SUFFIXES['Memory'].format(question=record['problem'])
    assert 'data-speaker' not in content[0].text
    assert 'Evan:' in content[0].text


def test_vtc_mode_places_images_inside_official_template() -> None:
    adapter = _make_adapter('Reasoning', 'vtc')
    record = {
        'problem': 'Who has been to Paris?',
        'answers': ['Yuki'],
        'images': ['data:image/jpeg;base64,AA==', 'data:image/jpeg;base64,AQ=='],
    }

    sample = adapter.record_to_sample(record)
    content = sample.input[0].content

    assert isinstance(content[0], ContentText)
    assert content[0].text == PROMPT_PREFIXES['Reasoning']
    assert isinstance(content[1], ContentImage)
    assert isinstance(content[2], ContentImage)
    assert isinstance(content[3], ContentText)
    assert content[3].text == PROMPT_SUFFIXES['Reasoning'].format(question=record['problem'])


def test_vtc_mode_requires_context_images() -> None:
    adapter = _make_adapter('Retrieval', 'vtc')
    with pytest.raises(ValueError, match='at least one context image'):
        adapter.record_to_sample({
            'problem': 'What is the number?',
            'answers': ['42'],
            'images': [],
        })
