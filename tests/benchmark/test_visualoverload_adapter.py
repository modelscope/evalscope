import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.visualoverload.visualoverload_adapter import VisualOverloadAdapter
from evalscope.config import TaskConfig
from evalscope.run import run_task


@pytest.fixture
def adapter() -> VisualOverloadAdapter:
    return VisualOverloadAdapter(
        benchmark_meta=BENCHMARK_REGISTRY['visualoverload'],
        task_config=TaskConfig(datasets=['visualoverload']),
    )


@pytest.fixture
def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new('RGB', (3840, 2160)).save(stream, format='JPEG')
    return stream.getvalue()


@pytest.mark.parametrize('question_type', ['choice', 'counting', 'ocr'])
def test_preserves_image_bytes_and_official_prompt(
    adapter: VisualOverloadAdapter, image_bytes: bytes, question_type: str
) -> None:
    sample = adapter.record_to_sample({
        'image': {'bytes': image_bytes, 'path': None},
        'default_prompt': 'The official prompt, including output constraints.',
        'question_id': '001abc',
        'question_type': question_type,
        'category': 'counting',
        'difficulty': 'hard',
    })
    content = sample.input[0].content
    assert base64.b64decode(content[0].image.split(',', 1)[1]) == image_bytes
    assert content[1].text == 'The official prompt, including output constraints.'
    assert sample.target == ''
    assert sample.metadata['question_id'] == '001abc'


def test_image_path_preserves_bytes(adapter: VisualOverloadAdapter, image_bytes: bytes, tmp_path: Path) -> None:
    image_path = tmp_path / 'image.jpg'
    image_path.write_bytes(image_bytes)
    sample = adapter.record_to_sample({
        'image': {'bytes': None, 'path': str(image_path)},
        'default_prompt': 'Count the animals.',
        'question_id': 'count', 'question_type': 'counting', 'category': 'counting', 'difficulty': 'hard',
    })
    assert base64.b64decode(sample.input[0].content[0].image.split(',', 1)[1]) == image_bytes


@pytest.mark.parametrize('response,ready', [('A', 1.0), ('17 animals', 1.0), ('   ', 0.0)])
def test_readiness_is_not_accuracy(adapter: VisualOverloadAdapter, response: str, ready: float) -> None:
    state = TaskState(model='mock', sample=Sample(input='prompt'))
    score = adapter.match_score(response, 'filtered output', '', state)
    assert score.value == {'response_present': ready}
    assert score.metadata['response_present'] == bool(ready)
    assert score.prediction == response
    assert score.metadata['official_score'] is None


def _write_reviews(report_dir: Path, records: list[dict]) -> None:
    review_dir = report_dir.parent.parent / 'reviews' / report_dir.name
    review_dir.mkdir(parents=True, exist_ok=True)
    with (review_dir / 'visualoverload_default.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps({'sample_score': {
                'sample_metadata': {'question_id': record['question_id']},
                'score': {'prediction': record['response']},
            }}) + '\n')


def test_export_from_persisted_reviews(adapter: VisualOverloadAdapter, tmp_path: Path) -> None:
    report_dir = tmp_path / 'reports' / 'model'
    expected = [
        {'question_id': '001abc', 'response': 'B'},
        {'question_id': 'count', 'response': 'There are 17 animals.'},
        {'question_id': 'ocr', 'response': 'café'},
        {'question_id': 'empty', 'response': ''},
    ]
    _write_reviews(report_dir, expected)
    adapter._on_generate_report_end(None, str(report_dir))
    assert json.loads((report_dir / 'visualoverload_submission.json').read_text()) == expected


@pytest.mark.parametrize('ids', [['same', 'same'], ['valid', None]])
def test_export_rejects_invalid_ids(adapter: VisualOverloadAdapter, tmp_path: Path, ids: list) -> None:
    report_dir = tmp_path / 'reports' / 'model'
    _write_reviews(report_dir, [{'question_id': question_id, 'response': 'A'} for question_id in ids])
    with pytest.raises(ValueError, match='question_id'):
        adapter._on_generate_report_end(None, str(report_dir))


def test_mock_pipeline_exports_submission(
    tmp_path: Path, image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise dataset loading, generation, reviews, reports, and cached export offline."""
    from datasets import Dataset
    from datasets import Image as DatasetImage
    from datasets import config as dataset_config

    monkeypatch.setattr(dataset_config, 'HF_DATASETS_CACHE', str(tmp_path / 'cache'))

    records = [{
        'question_id': f'question-{index}',
        'image': {'bytes': image_bytes, 'path': None},
        'default_prompt': f'Official prompt {index}',
        'question_type': 'counting', 'category': 'counting', 'difficulty': 'hard',
    } for index in range(2)]
    dataset = Dataset.from_list(records).cast_column('image', DatasetImage(decode=False))
    dataset_path = tmp_path / 'dataset'
    dataset_path.mkdir()
    dataset.to_parquet(dataset_path / 'test.parquet')
    config = TaskConfig(
        model='mock-model', eval_type='mock_llm', datasets=['visualoverload'],
        dataset_args={'visualoverload': {'local_path': str(dataset_path)}},
        work_dir=str(tmp_path / 'output'), dataset_dir=str(tmp_path / 'cache'), limit=2,
    )
    run_task(config)
    submissions = list((tmp_path / 'output').rglob('visualoverload_submission.json'))
    assert len(submissions) == 1
    report = json.loads((submissions[0].parent / 'visualoverload.json').read_text())
    assert report['primary_metric_identity'] is None
    assert report['metrics'][0]['semantics']['kind'] == 'diagnostic'
    first = json.loads(submissions[0].read_text())
    assert [item['question_id'] for item in first] == ['question-0', 'question-1']
    assert all(set(item) == {'question_id', 'response'} for item in first)
    submissions[0].unlink()
    config.use_cache = str(submissions[0].parents[2])
    run_task(config)
    assert json.loads(submissions[0].read_text()) == first
