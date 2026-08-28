import base64

import pytest

from evalscope.api.messages import ContentImage, ContentText
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.visfactor.utils import (
    aggregate_item_accuracy,
    extract_json_answer,
    normalize_prediction,
    render_question,
    score_prediction,
)
from evalscope.config import TaskConfig

_PNG_BYTES = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
)


def test_render_question_resolves_additional_image_placeholder() -> None:
    question = 'First:<IMAGE_0><br><ADDITIONAL_0> target:<ADDITIONAL_2>'
    assert render_question(question, 'three;unused;<IMAGE_1>') == 'First:<IMAGE_0>\nthree target:<IMAGE_1>'


def test_extract_json_answer_uses_last_object() -> None:
    prediction = 'Try {"answer": "FALSE"}. Final: {"answer": "TRUE"}.'
    assert extract_json_answer(prediction) == 'TRUE'


@pytest.mark.parametrize(
    ('category_id', 'prediction', 'reference', 'additional', 'normalized', 'score'),
    [
        ('CF1', '{"answer": "yes"}', 'T', '', 'T', 1.0),
        ('S2', '{"answer": "0"}', 'F', 'same cube', 'F', 1.0),
        ('CS1', '{"answer": "Boat"}', 'ship,boat,sailboat', '', 'Boat', 1.0),
        ('CS1', '{"answer": " Boat "}', 'ship, boat, sailboat', '', ' Boat ', 1.0),
        ('CF3', '{"answer": "(3, 5)"}', '(3, 5)', '5 times 5', '(3, 5)', 1.0),
        ('I3', 'The groups are 1 and 2, so the result is 2.', '2', 'three', '2', 1.0),
        ('VZ3', '{"answer": "H"}', 'H', '1', 'H', 1.0),
        ('VZ3', '{"answer": "false"}', 'F', 'edge pair', 'F', 1.0),
        ('unknown', '{"answer": "yes"}', 'T', '', '', 0.0),
    ],
)
def test_official_category_scoring(
    category_id: str,
    prediction: str,
    reference: str,
    additional: str,
    normalized: str,
    score: float,
) -> None:
    assert score_prediction(category_id, prediction, reference, additional) == (normalized, score)


def test_normalize_prediction_matches_official_fallback_rules() -> None:
    assert normalize_prediction('CF3', 'reasoning 1, 2; final row 4 column 5') == '(4, 5)'
    assert normalize_prediction('MA1', 'The lookup contains 21 entries. Final answer: 49') == '49'
    assert normalize_prediction('VZ3', 'I considered A and B; the matching edge is H', '1') == 'H'


def test_aggregate_item_accuracy_requires_every_row() -> None:
    category_scores = aggregate_item_accuracy(
        [
            ('CF1', 0, 1.0),
            ('CF1', 0, 1.0),
            ('CF1', 1, 1.0),
            ('CF1', 1, 0.0),
            ('CS1', 0, 1.0),
        ]
    )
    assert category_scores == {'CF1': (0.5, 2), 'CS1': (1.0, 1)}


def test_adapter_interleaves_images_and_selects_official_primary_metric() -> None:
    adapter = get_benchmark('visfactor', TaskConfig(datasets=['visfactor']))
    record = {
        'index': 7,
        'category_id': 'I3',
        'category_name': 'Figure Classification',
        'eval_index': 2,
        'image': [{'bytes': _PNG_BYTES, 'path': None}] * 3,
        'question': 'Group 1:<IMAGE_0>Group 2:<IMAGE_1><ADDITIONAL_0>',
        'answer': '2',
        'additional': '<IMAGE_2>',
    }

    sample = adapter.record_to_sample(record)
    content = sample.input[0].content
    assert [type(item) for item in content] == [
        ContentText,
        ContentImage,
        ContentText,
        ContentImage,
        ContentImage,
    ]

    scores = [
        SampleScore(
            score=Score(value={'accuracy': value}, main_score_name='accuracy'),
            sample_metadata={
                'category_id': category_id,
                'category_name': category_id,
                'eval_index': eval_index,
            },
        )
        for category_id, eval_index, value in [
            ('CF1', 0, 1.0),
            ('CF1', 0, 1.0),
            ('CF1', 1, 0.0),
            ('CS1', 0, 1.0),
        ]
    ]
    aggregates = adapter.aggregate_scores(scores)
    assert aggregates[0].score == pytest.approx(0.75)
    assert aggregates[0].num == 3

    report = adapter.generate_report({'default': aggregates}, model_name='mock', output_dir='')
    assert report.score == pytest.approx(0.75)
