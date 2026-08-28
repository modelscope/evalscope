import json
from pathlib import Path

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, MetricIdentity, SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.surds.utils import (
    IMAGE_SIZE,
    SUBSET_LIST,
    build_official_vqa_records,
    compute_centerness,
    extract_tagged_answer,
    normalize_answer,
    parse_point,
)
from evalscope.config import TaskConfig
from evalscope.report import ReportGenerator


def _source_record(file_name: str, descriptions: list[str]) -> dict:
    count = len(descriptions)
    return {
        'file_name': file_name,
        'descs': descriptions,
        'xy2Ds': [[400 + 300 * index, 450] for index in range(count)],
        'xyz3Ds': [[0.0, 0.0, 5.0 + 5 * index] for index in range(count)],
        'bboxes2D': [[350 + 300 * index, 400, 450 + 300 * index, 500] for index in range(count)],
        'categories': ['car'] * count,
        'yaws': [0.0] * count,
        'yaw_descs': ['North'] * count,
        'depths': [5.0 + 5 * index for index in range(count)],
        'distances': [5 + 5 * index for index in range(count)],
    }


def test_official_vqa_generation_is_deterministic() -> None:
    records = [
        _source_record('image/CAM_FRONT/single-a.webp', ['white car']),
        _source_record('image/CAM_FRONT/single-b.webp', ['adult wearing blue clothes']),
        _source_record('image/CAM_FRONT/multi.webp', ['white car', 'adult wearing red clothes']),
    ]

    first = build_official_vqa_records(records)
    second = build_official_vqa_records(records)

    assert first == second
    assert {subset: len(first[subset]) for subset in SUBSET_LIST} == dict.fromkeys(SUBSET_LIST, 1)
    assert len(first['yaw'][0]['prompts']) == 2
    assert len(first['distance'][0]['prompts']) == 2
    assert len(first['left_right'][0]['prompts']) == 2
    assert len(first['front_behind'][0]['prompts']) == 2
    assert len(first['xy2d'][0]['prompts']) == 1
    assert len(first['depth'][0]['prompts']) == 1
    assert first['distance'][0]['answers'] == ['The white car', 'The adult wearing red clothes']
    assert first['left_right'][0]['answers'] == ['The white car', 'The adult wearing red clothes']
    assert first['front_behind'][0]['answers'] == ['No', 'Yes']


def test_answer_extraction_and_normalization_match_official_rules() -> None:
    assert extract_tagged_answer('<think>...</think><answer> The white car. </answer>') == 'The white car.'
    assert extract_tagged_answer('<answer>first</answer><answer>second</answer>') == 'first'
    assert extract_tagged_answer('The white car') == ''
    assert normalize_answer(' The White Car. ') == 'white car'
    assert normalize_answer('Between 2 meters, and 8 meters!') == 'between 2 meters and 8 meters'


def test_point_parsing_and_centerness_match_official_rules() -> None:
    assert parse_point('[0.5, 0.5]') == (800.0, 450.0)
    assert parse_point('[10, 20, 30, 40]') == (20.0, 30.0)
    assert parse_point('[1600, 450]') is None
    assert parse_point('[1, 1]') == (1.0, 1.0)
    assert parse_point('no coordinates') is None

    bbox = [100, 100, 300, 300]
    assert compute_centerness((200, 200), bbox) == pytest.approx(1.0)
    assert compute_centerness((100, 200), bbox) == pytest.approx(0.0)
    assert compute_centerness((99, 200), bbox) == pytest.approx(0.0)


def test_adapter_applies_official_validity_and_pixel_scoring() -> None:
    adapter = get_benchmark('surds', TaskConfig(model='mock', datasets=['surds']))
    categorical_state = TaskState(
        model='mock',
        sample=Sample(
            input='question',
            target='The white car',
            metadata={'task': 'distance', 'options': ['The white car', 'The red car']},
        ),
    )
    exact = adapter.match_score('', 'The white car', 'The white car', categorical_state)
    missing_article = adapter.match_score('', 'white car', 'The white car', categorical_state)
    assert exact.value == {'normalized_score': 1.0}
    assert missing_article.value == {'normalized_score': 0.0}

    pixel_state = TaskState(
        model='mock',
        sample=Sample(
            input='question',
            target='[50, 50]',
            metadata={'task': 'xy2d', 'bbox': [0, 0, 100, 100], 'image_size': [1600, 900]},
        ),
    )
    pixel = adapter.match_score('', '[50, 50]', '[50, 50]', pixel_state)
    assert pixel.value == {'normalized_score': 1.0}


def test_pair_aggregation_requires_both_complementary_answers() -> None:
    adapter = get_benchmark('surds', TaskConfig(model='mock', datasets=['surds']))
    scores = [
        SampleScore(
            score=Score(value={'normalized_score': 1.0}),
            sample_id=0,
            generation_index=0,
            sample_metadata={'paired': True, 'pair_id': 'a'},
        ),
        SampleScore(
            score=Score(value={'normalized_score': 1.0}),
            sample_id=1,
            generation_index=0,
            sample_metadata={'paired': True, 'pair_id': 'a'},
        ),
        SampleScore(
            score=Score(value={'normalized_score': 1.0}),
            sample_id=2,
            generation_index=1,
            sample_metadata={'paired': True, 'pair_id': 'a'},
        ),
        SampleScore(
            score=Score(value={'normalized_score': 0.0}),
            sample_id=3,
            generation_index=1,
            sample_metadata={'paired': True, 'pair_id': 'a'},
        ),
    ]

    aggregate = adapter.aggregate_scores(scores)[0]

    assert aggregate.metric_name == 'normalized_score'
    assert aggregate.score == pytest.approx(0.5)
    assert aggregate.num == 2

    incomplete = adapter.aggregate_scores([
        SampleScore(
            score=Score(value={'normalized_score': 1.0}),
            sample_id=4,
            sample_metadata={'paired': True, 'pair_id': 'incomplete'},
        )
    ])[0]
    assert incomplete.score == 0.0
    assert incomplete.num == 1


def test_report_table_uses_official_equal_task_average() -> None:
    adapter = get_benchmark('surds', TaskConfig(model='mock', datasets=['surds']))
    subset_scores = {
        subset: [AggScore(metric_name='normalized_score', aggregation='mean', score=index / 5, num=5)]
        for index, subset in enumerate(SUBSET_LIST)
    }

    report = ReportGenerator.generate_report(subset_scores, 'mock', adapter)
    table = report.to_dataframe(add_overall_metric=True)
    overall = table[table['Subset'] == 'OVERALL'].iloc[0]

    assert report.primary_metric_identity == MetricIdentity(name='normalized_score', aggregation='mean')
    assert table[table['Subset'] != 'OVERALL']['Num'].tolist() == [5] * 6
    assert overall['Num'] == 30
    assert overall['Score'] == pytest.approx(0.5)
    assert report.metrics[0].macro_score == pytest.approx(0.5)


def test_local_dataset_loads_all_six_subsets(tmp_path: Path) -> None:
    split_root = tmp_path / 'validation'
    image_root = split_root / 'image' / 'CAM_FRONT'
    image_root.mkdir(parents=True)
    records = [
        _source_record('image/CAM_FRONT/single-a.webp', ['white car']),
        _source_record('image/CAM_FRONT/single-b.webp', ['adult wearing blue clothes']),
        _source_record('image/CAM_FRONT/multi.webp', ['white car', 'adult wearing red clothes']),
    ]
    for record in records:
        (split_root / record['file_name']).write_bytes(b'image')
    with open(split_root / 'metadata.jsonl', 'w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record) + '\n')

    config = TaskConfig(
        model='mock',
        datasets=['surds'],
        dataset_args={'surds': {
            'local_path': str(tmp_path)
        }},
    )
    datasets = get_benchmark('surds', config).load_dataset()

    assert {subset: len(datasets[subset]) for subset in SUBSET_LIST} == {
        'yaw': 2,
        'xy2d': 1,
        'depth': 1,
        'distance': 2,
        'left_right': 2,
        'front_behind': 2,
    }
    assert all(sample.metadata['image_size'] == list(IMAGE_SIZE) for dataset in datasets.values() for sample in dataset)
