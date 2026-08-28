import json
from types import SimpleNamespace

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.ref_adv_s.ref_adv_s_adapter import FOLLOWUP_PROMPT, RefAdvSAdapter
from evalscope.benchmarks.ref_adv_s.utils import distractor_bin, iou_xyxy, parse_bboxes, to_normalized_xyxy
from evalscope.config import TaskConfig


def make_adapter(box_format: str = 'norm_1000_xyxy') -> RefAdvSAdapter:
    config = TaskConfig(
        model='mock',
        datasets=['ref_adv_s'],
        dataset_args={'ref_adv_s': {'extra_params': {'pred_box_format': box_format}}},
    )
    return get_benchmark('ref_adv_s', config)


@pytest.mark.parametrize(
    ('box_format', 'box'),
    [
        ('abs_xyxy', [64, 48, 320, 240]),
        ('norm_1000_xyxy', [100, 100, 500, 500]),
        ('norm_1_xyxy', [0.1, 0.1, 0.5, 0.5]),
    ],
)
def test_parse_bboxes_supports_official_coordinate_formats(box_format: str, box: list[float]) -> None:
    response = f'```json\n{{"bboxes": [{json.dumps(box)}]}}\n```'
    boxes, error = parse_bboxes(response, image_size=(640, 480), box_format=box_format)

    assert error == ''
    assert boxes[0] == pytest.approx([0.1, 0.1, 0.5, 0.5])


def test_parse_bboxes_uses_last_fenced_json_and_deduplicates() -> None:
    response = (
        '```json\n{"bbox": [0, 0, 100, 100]}\n```\n'
        'Final answer:\n```json\n{"objects": [{"bbox_2d": [200, 300, 800, 900]}, '
        '{"bbox_2d": [200, 300, 800, 900]}]}\n```'
    )
    boxes, error = parse_bboxes(response, image_size=(1000, 1000), box_format='norm_1000_xyxy')

    assert error == ''
    assert boxes == [[0.2, 0.3, 0.8, 0.9]]


def test_parse_bboxes_requires_complete_json_at_end_without_fence() -> None:
    boxes, error = parse_bboxes('Reasoning {"bboxes": [[1, 2, 3, 4]]} trailing text', (10, 10), 'abs_xyxy')
    assert boxes == []
    assert error == 'no_bbox_found'

    boxes, error = parse_bboxes('Reasoning\n{"bboxes": [[1, 2, 3, 4]]}', (10, 10), 'abs_xyxy')
    assert error == ''
    assert boxes == [[0.1, 0.2, 0.3, 0.4]]


def test_to_normalized_xyxy_sorts_and_clips_coordinates() -> None:
    assert to_normalized_xyxy([1200, 900, -100, -50], (1, 1), 'norm_1000_xyxy') == [0.0, 0.0, 1.0, 0.9]
    with pytest.raises(ValueError, match='Unsupported pred_box_format'):
        to_normalized_xyxy([0, 0, 1, 1], (1, 1), 'xywh')


def test_iou_and_distractor_bins_match_official_definitions() -> None:
    assert iou_xyxy([0, 0, 1, 1], [0.5, 0.5, 1, 1]) == pytest.approx(0.25)
    assert distractor_bin(2) == '2-3'
    assert distractor_bin(4) == '4-6'
    assert distractor_bin(7) == '>=7'
    assert distractor_bin(1) is None


def test_followup_is_only_used_after_parse_failure() -> None:
    adapter = make_adapter()
    sample = Sample(
        input=[ChatMessageUser(content='prompt')],
        metadata={'sent_size': [640, 480], 'retry_followup_used': False},
    )

    valid_history = [sample.input[0], ChatMessageAssistant(content='{"bboxes": [[1, 2, 3, 4]]}')]
    assert adapter.build_turn_prompt(sample, valid_history, 1) is None
    assert sample.metadata['retry_followup_used'] is False

    invalid_history = [sample.input[0], ChatMessageAssistant(content='not json')]
    assert adapter.build_turn_prompt(sample, invalid_history, 1) == FOLLOWUP_PROMPT
    assert sample.metadata['retry_followup_used'] is True


def test_match_score_reports_official_thresholds_and_bin() -> None:
    adapter = object.__new__(RefAdvSAdapter)
    state = SimpleNamespace(
        metadata={
            'target_box_normalized': [0.0, 0.0, 1.0, 1.0],
            'distractor_count': 5,
            'parse_error': '',
            'retry_followup_used': False,
        }
    )

    score = adapter.match_score('raw', '[0.0, 0.0, 1.0, 0.8]', 'unused', state)

    assert isinstance(score, Score)
    assert score.value == {'ACC@0.5': 1.0, 'ACC@0.75': 1.0, 'ACC@0.9': 0.0, '4-6/ACC@0.5': 1.0}
    assert score.main_score_name == 'ACC@0.5'
    assert score.metadata['iou'] == pytest.approx(0.8)


def test_aggregation_emits_structured_official_metrics() -> None:
    adapter = make_adapter()
    scores = []
    for sample_id, value, bin_name in ((0, 1.0, '2-3'), (1, 0.0, '4-6')):
        score = Score(value={'ACC@0.5': value, 'ACC@0.75': 0.0, 'ACC@0.9': 0.0})
        score.value[f'{bin_name}/ACC@0.5'] = value
        scores.append(SampleScore(score=score, sample_id=sample_id))

    aggregates = adapter.aggregate_scores(scores)
    identities = {(item.metric_name, item.aggregation, frozenset(item.dimensions.items())) for item in aggregates}

    assert ('accuracy', 'mean', frozenset({('scope', 'overall'), ('threshold', 0.5)})) in identities
    assert ('accuracy', 'mean', frozenset({('scope', '2-3'), ('threshold', 0.5)})) in identities
    assert next(
        item for item in aggregates if item.dimensions == {'scope': 'overall', 'threshold': 0.5}
    ).score == pytest.approx(0.5)


def test_parse_failure_scores_zero() -> None:
    adapter = make_adapter()
    state = SimpleNamespace(
        metadata={
            'sent_size': [640, 480],
            'target_box_normalized': [0.1, 0.1, 0.2, 0.2],
            'distractor_count': 2,
            'retry_followup_used': True,
        }
    )

    extracted = adapter.extract_answer('no valid bbox', state)
    score = adapter.match_score('no valid bbox', extracted, 'unused', state)

    assert extracted == ''
    assert score.value['ACC@0.5'] == 0.0
    assert score.metadata == {'iou': 0.0, 'parse_error': 'no_bbox_found', 'retry_followup_used': True}
