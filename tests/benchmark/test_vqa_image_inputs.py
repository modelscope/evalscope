"""Image normalization and offline native pipelines for the single-image VQA adapters."""

import base64
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from evalscope.api.benchmark import VisionLanguageAdapter
from evalscope.api.model.model import ModelCache
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.models.utils.openai import openai_chat_completion_part
from evalscope.run import run_task

BENCHMARKS = ['docvqa', 'infovqa', 'pope']


def _adapter(name: str, **dataset_args: Any) -> VisionLanguageAdapter:
    config = TaskConfig(model='test', eval_type='mock_llm', datasets=[name], dataset_args={name: dataset_args})
    return get_benchmark(name, config)


def _record(image: Any, question: str = 'Is a cat visible?') -> dict:
    return {'question': question, 'image': image, 'answers': ['yes'], 'answer': 'yes', 'questionId': 42}


def _image_bytes(image_format: str, noisy: bool = False) -> bytes:
    image = (Image.frombytes('RGB', (768, 768), random.Random(42).randbytes(768 * 768 * 3))
             if noisy else Image.new('RGB', (4, 4), color='red'))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture(params=BENCHMARKS)
def adapter(request: pytest.FixtureRequest) -> VisionLanguageAdapter:
    return _adapter(request.param)


@pytest.mark.parametrize('image_format', ['PNG', 'JPEG'])
@pytest.mark.parametrize('limit', [None, 65536])
def test_bytes_preserve_format_payload_and_sample(
    adapter: VisionLanguageAdapter, image_format: str, limit: int | None
) -> None:
    adapter = _adapter(adapter.name, max_image_bytes=limit)
    data = _image_bytes(image_format)
    sample = adapter.record_to_sample(_record({'bytes': data, 'path': '/missing/stale.png'}))
    content = sample.input[0].content
    assert [part.type for part in content] == ['text', 'image']
    uri = openai_chat_completion_part(content[1])['image_url']['url']
    assert uri.startswith(f'data:image/{image_format.lower()};base64,')
    assert base64.b64decode(uri.split(',', 1)[1]) == data
    assert 'Is a cat visible?' in content[0].text
    assert sample.target == ('YES' if adapter.name == 'pope' else '["yes"]')
    assert sample.metadata == adapter.record_to_sample(_record(None)).metadata


@pytest.mark.parametrize('path_dict', [True, False])
def test_local_image_reaches_request(
    adapter: VisionLanguageAdapter, tmp_path: Path, path_dict: bool
) -> None:
    data = _image_bytes('JPEG')
    path = tmp_path / 'image.jpg'
    path.write_bytes(data)
    image = {'bytes': None, 'path': str(path)} if path_dict else str(path)
    sample = adapter.record_to_sample(_record(image))
    uri = openai_chat_completion_part(sample.input[0].content[1])['image_url']['url']
    assert uri.startswith('data:image/jpeg;base64,')
    assert base64.b64decode(uri.split(',', 1)[1]) == data


def test_configured_byte_image_limit_is_applied(adapter: VisionLanguageAdapter) -> None:
    adapter = _adapter(adapter.name, max_image_bytes=65536)
    data = _image_bytes('PNG', noisy=True)
    assert len(data) > 65536
    sample = adapter.record_to_sample(_record({'bytes': data, 'path': None}))
    uri = openai_chat_completion_part(sample.input[0].content[1])['image_url']['url']
    encoded = base64.b64decode(uri.split(',', 1)[1])
    assert len(encoded) <= 65536
    assert uri.startswith('data:image/jpeg;base64,')
    with Image.open(BytesIO(encoded)) as image:
        image.load()
        assert image.format == 'JPEG'


@pytest.mark.parametrize('image', [None, {}, ''])
def test_missing_image_keeps_text_only_behavior(adapter: VisionLanguageAdapter, image: Any) -> None:
    sample = adapter.record_to_sample(_record(image))
    assert [part.type for part in sample.input[0].content] == ['text']


@pytest.mark.parametrize('image_format', ['JPEG', 'PNG'])
def test_chartqa_image_format_and_subset(image_format: str) -> None:
    data = _image_bytes(image_format)
    image = {'bytes': data, 'path': 'stale.png'}
    adapter = _adapter('chartqa')
    sample = adapter.record_to_sample({
        'question': 'What is the maximum?', 'answer': '42', 'type': 'augmented_test', 'image': image,
    })
    text, picture = sample.input[0].content
    assert 'What is the maximum?' in text.text
    assert (sample.target, sample.subset_key) == ('42', 'augmented_test')
    uri = openai_chat_completion_part(picture)['image_url']['url']
    header, encoded = uri.split(',', 1)
    raw = base64.b64decode(encoded, validate=True)
    assert raw == data
    with Image.open(BytesIO(raw)) as decoded:
        assert decoded.format == image_format
        assert header == f'data:{Image.MIME[decoded.format]};base64'


@pytest.mark.parametrize('name', BENCHMARKS)
@pytest.mark.parametrize('source', ['parquet_bytes', 'jsonl_path'])
def test_native_pipeline_serializes_images_scores_and_resumes(
    name: str, source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep loaders, model SDK, scoring and reports real; mock only HTTP transport."""
    from datasets import Dataset
    from datasets import config as dataset_config

    from evalscope.models import openai_compatible

    monkeypatch.setattr(dataset_config, 'HF_DATASETS_CACHE', str(tmp_path / 'hf-cache'))
    monkeypatch.setattr(ModelCache, '_models', {})
    data = _image_bytes('JPEG', noisy=True)
    image_path = tmp_path / 'image.jpg'
    image_path.write_bytes(data)
    image = {'bytes': data, 'path': None} if source == 'parquet_bytes' else {'bytes': None, 'path': str(image_path)}
    records = [_record(image, question=f'Question {index}: Is a cat visible?') for index in range(2)]
    subset = 'popular' if name == 'pope' else ('DocVQA' if name == 'docvqa' else 'InfographicVQA')
    dataset_dir = tmp_path / 'data'
    dataset_dir.mkdir()
    stem = 'Full_popular' if name == 'pope' else subset
    dataset_path = dataset_dir / (f'{stem}.parquet' if source == 'parquet_bytes' else f'{stem}.jsonl')
    config_name = 'Full' if name == 'pope' else subset
    split = 'popular' if name == 'pope' else 'validation'
    (dataset_dir / 'README.md').write_text(
        f'---\nconfigs:\n- config_name: {config_name}\n  data_files:\n'
        f'  - split: {split}\n    path: {dataset_path.name}\n---\n', encoding='utf-8')
    if source == 'parquet_bytes':
        Dataset.from_list(records).to_parquet(str(dataset_path))
    else:
        dataset_path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')

    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        assert request.url.path == '/v1/chat/completions'
        content = next(m['content'] for m in payload['messages'] if m['role'] == 'user')
        assert [part['type'] for part in content] == ['text', 'image_url']
        uri = content[1]['image_url']['url']
        assert uri.startswith('data:image/jpeg;base64,')
        raw = base64.b64decode(uri.split(',', 1)[1])
        if source == 'parquet_bytes':
            assert len(raw) <= 65536
        else:
            assert raw == data
        with Image.open(BytesIO(raw)) as decoded:
            decoded.load()
        # One correct and one incorrect answer, independent of concurrent request order.
        answer = 'YES' if 'Question 0:' in content[0]['text'] else 'NO'
        return httpx.Response(200, json={
            'id': 'chat-test', 'model': 'image-test', 'object': 'chat.completion', 'created': 0,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': f'ANSWER: {answer}'},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15},
        })

    sdk_client = openai_compatible.OpenAI
    monkeypatch.setattr(openai_compatible, 'OpenAI', lambda **kwargs: sdk_client(
        http_client=httpx.Client(transport=httpx.MockTransport(handle)), **kwargs))
    output = tmp_path / 'output'
    config = TaskConfig(
        model='image-test', eval_type='openai_api', api_url='https://example.test/v1', api_key='offline-test',
        datasets=[name], dataset_args={name: {'local_path': str(dataset_dir), 'subset_list': [subset],
                                            'max_image_bytes': 65536}},
        generation_config={'max_tokens': 8, 'retries': 1}, eval_batch_size=2, work_dir=str(output),
        dataset_dir=str(tmp_path / 'dataset-cache'), no_timestamp=True,
    )
    run_task(config)
    assert len(seen) == 2
    predictions = [json.loads(line) for line in next((output / 'predictions').rglob('*.jsonl')).read_text().splitlines()]
    reviews = [json.loads(line) for line in next((output / 'reviews').rglob('*.jsonl')).read_text().splitlines()]
    assert len(predictions) == len(reviews) == 2
    metric = 'acc' if name == 'pope' else 'anls'
    assert sorted(row['sample_score']['score']['value'][metric] for row in reviews) == [0.0, 1.0]
    report_path = next((output / 'reports').rglob(f'{name}.json'))
    report = json.loads(report_path.read_text())
    assert report['execution_summary']['succeeded'] == 2
    assert report['execution_summary']['errored'] == 0
    report_metric = 'accuracy' if name == 'pope' else 'anls'
    aggregate = next(item for item in report['metrics'] if item['identity']['name'] == report_metric)
    assert aggregate['score'] == 0.5
    assert aggregate['num'] == 2
    assert (output / 'reports' / 'report.html').is_file()

    config.use_cache = str(output)
    config.rerun_review = True
    run_task(config)
    assert len(seen) == 2
    assert json.loads(report_path.read_text())['metrics'] == report['metrics']
