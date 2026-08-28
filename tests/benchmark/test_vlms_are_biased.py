"""Unit tests for the VLMs Are Biased adapter and official scoring rules."""
import io
from types import SimpleNamespace
from typing import Any, Dict

from PIL import Image

from evalscope.api.metric import MetricDirection, SampleScore
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.vlms_are_biased.utils import normalize_answer, score_answer
from evalscope.config import TaskConfig
from evalscope.metrics.semantics.catalog import METRIC_DEFINITIONS


def _adapter() -> Any:
    task_config = TaskConfig(model='mock', datasets=['vlms_are_biased'])
    return get_benchmark('vlms_are_biased', task_config)


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buffer, format='PNG')
    return buffer.getvalue()


def _record(**overrides: Any) -> Dict[str, Any]:
    record = {
        'image': {
            'bytes': _image_bytes(),
            'path': 'sample.png'
        },
        'ID': 'sample_Q1_px384',
        'topic': 'Logos',
        'sub_topic': 'Shoe Logos',
        'prompt': 'How many stripes are visible? Answer with a number in curly brackets, e.g., {9}.',
        'ground_truth': '4',
        'expected_bias': '3',
        'with_title': False,
        'type_of_question': 'Q1',
        'pixel': 384,
    }
    record.update(overrides)
    return record


def test_normalize_answer_matches_official_brace_rule() -> None:
    assert normalize_answer('  {Yes}\n') == 'yes'
    assert normalize_answer('{{4}}') == '4'
    assert normalize_answer('The answer is {4}.') == 'the answer is {4}.'


def test_score_answer_exact_and_bias_matches() -> None:
    assert score_answer('{4}', '4', '3') == {'acc': 1.0, 'bias_ratio': 0.0}
    assert score_answer('{3}', '4', '3') == {'acc': 0.0, 'bias_ratio': 1.0}
    assert score_answer('{2}', '4', '3') == {'acc': 0.0, 'bias_ratio': 0.0}
    assert score_answer('{YES}', 'Yes', 'No') == {'acc': 1.0, 'bias_ratio': 0.0}


def test_score_answer_uses_official_digit_fallback() -> None:
    assert score_answer('There are {4} visible stripes.', '4', '3') == {'acc': 1.0, 'bias_ratio': 0.0}
    assert score_answer('I see 3 stripes.', '4', '3') == {'acc': 0.0, 'bias_ratio': 1.0}
    assert score_answer('I counted 3, then 4.', '4', '3') == {'acc': 0.0, 'bias_ratio': 0.0}


def test_original_control_omits_bias_ratio() -> None:
    assert score_answer('{Audi}', 'Audi', None) == {'acc': 1.0}


def test_bias_ratio_semantics_are_lower_is_better() -> None:
    semantics = METRIC_DEFINITIONS['bias_ratio'].resolve('bias_ratio')

    assert semantics.metric_name == 'Bias Ratio'
    assert semantics.direction is MetricDirection.LOWER_IS_BETTER


def test_record_to_sample_preserves_official_prompt_and_metadata() -> None:
    sample = _adapter().record_to_sample(_record())

    assert sample.target == '4'
    assert sample.input[0].content[0].image.startswith('data:image/png;base64,')
    assert sample.input[0].content[1].text == _record()['prompt']
    assert sample.metadata == {
        'id': 'sample_Q1_px384',
        'topic': 'Logos',
        'sub_topic': 'Shoe Logos',
        'type_of_question': 'Q1',
        'expected_bias': '3',
        'with_title': False,
        'pixel': 384,
    }


def test_match_score_uses_expected_bias_metadata() -> None:
    score = _adapter().match_score(
        original_prediction='{3}',
        filtered_prediction='3',
        reference='4',
        task_state=SimpleNamespace(metadata={'expected_bias': '3'}),
    )

    assert score.value == {'acc': 0.0, 'bias_ratio': 1.0}
    assert score.main_score_name == 'acc'


def test_aggregation_reports_official_per_topic_accuracy() -> None:
    sample_scores = [
        SampleScore(sample_id='1', score=Score(value={'acc': 1.0, 'bias_ratio': 0.0}), sample_metadata={'topic': 'Logos'}),
        SampleScore(sample_id='2', score=Score(value={'acc': 0.0, 'bias_ratio': 1.0}), sample_metadata={'topic': 'Logos'}),
        SampleScore(sample_id='3', score=Score(value={'acc': 1.0, 'bias_ratio': 0.0}), sample_metadata={'topic': 'Flags'}),
    ]

    aggregate_scores = _adapter().aggregate_scores(sample_scores)
    by_identity = {(score.metric_name, score.dimensions.get('topic')): score.score for score in aggregate_scores}

    assert by_identity[('accuracy', None)] == 2 / 3
    assert by_identity[('bias_ratio', None)] == 1 / 3
    assert by_identity[('accuracy', 'Logos')] == 0.5
    assert by_identity[('accuracy', 'Flags')] == 1.0
