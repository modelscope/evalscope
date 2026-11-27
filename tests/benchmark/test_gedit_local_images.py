"""Offline image manifests must select the intended files before judge scoring."""

import base64
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from evalscope import TaskConfig, run_task
from evalscope.api.benchmark import BenchmarkMeta, ImageEditAdapter
from evalscope.api.dataset import Sample
from evalscope.api.model.model import ModelCache
from evalscope.api.registry import get_benchmark
from evalscope.constants import ModelTask
from evalscope.models.mockllm import MockLLM


def _manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return path


def _adapter(manifest: str | None) -> ImageEditAdapter:
    return ImageEditAdapter(benchmark_meta=BenchmarkMeta(
        name='image_edit_probe', dataset_id='local', extra_params={'local_file': manifest}))


@pytest.mark.parametrize('relative_manifest', [False, True])
@pytest.mark.parametrize('relative_image', [False, True])
def test_manifest_path_is_retained_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_manifest: bool, relative_image: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / 'outputs' / 'images' / 'edited.png'
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b'expected image')
    shadow = tmp_path / 'images' / 'edited.png'
    shadow.parent.mkdir()
    shadow.write_bytes(b'wrong image')
    row = {'id': 'sample', 'image_path': 'images/edited.png' if relative_image else str(expected)}
    manifest = _manifest(tmp_path / 'outputs' / 'predictions.jsonl', [row])
    adapter = _adapter(str(manifest.relative_to(tmp_path) if relative_manifest else manifest))

    selected = Path(adapter.get_image_path_from_id('sample'))
    assert selected.is_absolute()
    assert selected == expected
    assert selected.read_bytes() == b'expected image'
    assert json.loads(manifest.read_text()) == row
    monkeypatch.chdir(tmp_path.parent)
    assert Path(adapter.get_image_path_from_id('sample')).read_bytes() == b'expected image'


@pytest.mark.parametrize('sample_id', [0, 'sample'])
def test_manifest_selects_matching_ids_only(tmp_path: Path, sample_id: Any) -> None:
    image = tmp_path / 'edited.png'
    image.write_bytes(b'image')
    adapter = _adapter(str(_manifest(tmp_path / 'predictions.jsonl', [{'id': sample_id, 'image_path': str(image)}])))
    assert adapter.sample_filter(Sample(input='Edit', metadata={'id': sample_id}))
    assert not adapter.sample_filter(Sample(input='Edit', metadata={'id': 'missing'}))
    assert not adapter.sample_filter(Sample(input='Edit'))


def test_empty_manifest_selects_no_samples(tmp_path: Path) -> None:
    adapter = _adapter(str(_manifest(tmp_path / 'empty.jsonl', [])))
    assert not adapter.sample_filter(Sample(input='Edit', metadata={'id': 'sample'}))


def test_no_manifest_preserves_online_sample_selection() -> None:
    assert _adapter(None).sample_filter(Sample(input='Edit'))


@pytest.mark.parametrize('missing_key', ['id', 'image_path'])
def test_manifest_requires_lookup_and_image_keys(tmp_path: Path, missing_key: str) -> None:
    row = {'id': 'sample', 'image_path': 'edited.png'}
    row.pop(missing_key)
    with pytest.raises(ValueError, match='not found in record'):
        _adapter(str(_manifest(tmp_path / 'predictions.jsonl', [row])))


@pytest.mark.parametrize('directory', [False, True])
def test_manifest_requires_an_existing_image_file(tmp_path: Path, directory: bool) -> None:
    image = tmp_path / 'edited.png'
    if directory:
        image.mkdir()
    with pytest.raises(FileNotFoundError, match='Image file'):
        _adapter(str(_manifest(tmp_path / 'predictions.jsonl', [{'id': 'sample', 'image_path': str(image)}])))


@pytest.mark.parametrize('custom_keys', [False, True])
def test_gedit_accepts_offline_manifest_options(tmp_path: Path, custom_keys: bool) -> None:
    id_key, image_key = ('sample_key', 'result') if custom_keys else ('id', 'image_path')
    image = tmp_path / 'edited.png'
    image.write_bytes(b'image')
    manifest = _manifest(tmp_path / 'predictions.jsonl', [{id_key: 'sample', image_key: str(image)}])
    config = TaskConfig(datasets=['gedit'], dataset_args={'gedit': {'extra_params': {
        'local_file': str(manifest), 'id_key': id_key, 'image_key': image_key}}})

    assert get_benchmark('gedit', config).get_image_path_from_id('sample') == str(image)


def _png(color: str) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (8, 8), color).save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.mark.parametrize('custom_keys', [False, True])
@pytest.mark.parametrize('relative_images', [False, True])
def test_native_offline_images_reach_judge_reports_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_keys: bool, relative_images: bool,
) -> None:
    """Use the real loader, judge SDK, contracts and reports; replace only HTTP transport."""
    from datasets import Dataset

    from evalscope.models import openai_compatible

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ModelCache, '_models', {})
    original, red, green = _png('black'), _png('red'), _png('green')
    records = [{'key': key, 'instruction': f'Edit {key}', 'instruction_language': language,
                'task_type': subset, 'input_image': {'bytes': original}, 'input_image_raw': None}
               for key, subset, language in [('red', 'color_alter', 'en'), ('green', 'material_alter', 'en'),
                                              ('unlisted', 'color_alter', 'en'), ('wrong-language', 'color_alter', 'cn')]]
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    Dataset.from_list(records).to_parquet(str(dataset / 'default.parquet'))
    image_dir = tmp_path / 'offline' / 'images'
    image_dir.mkdir(parents=True)
    (tmp_path / 'images').mkdir()
    id_key, image_key = ('sample_key', 'result') if custom_keys else ('id', 'image_path')
    rows = []
    for key, content in [('red', red), ('green', green), ('wrong-language', red)]:
        (image_dir / f'{key}.png').write_bytes(content)
        (tmp_path / 'images' / f'{key}.png').write_bytes(_png('blue'))
        rows.append({id_key: key, image_key: f'images/{key}.png' if relative_images else str(image_dir / f'{key}.png')})
    manifest = _manifest(tmp_path / 'offline' / 'predictions.jsonl', rows)
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == '/v1/chat/completions'
        assert payload['model'] == 'offline-judge'
        content = next(message['content'] for message in payload['messages'] if message['role'] == 'user')
        images = [base64.b64decode(part['image_url']['url'].split(',', 1)[1]) for part in content
                  if part['type'] == 'image_url']
        assert images[-1] in (red, green)
        kind = 'SC' if len(images) == 2 else 'PQ'
        if kind == 'SC':
            assert images[0] == original
        calls.append((images[-1], kind))
        verdict = [9, 4] if kind == 'SC' else [9, 10]
        return httpx.Response(200, json={
            'id': 'judge-fixture', 'model': 'offline-judge', 'object': 'chat.completion', 'created': 0,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps({'score': verdict})},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17},
        })

    client = openai_compatible.OpenAI
    monkeypatch.setattr(openai_compatible, 'OpenAI', lambda **kwargs: client(
        http_client=httpx.Client(transport=httpx.MockTransport(handle)), **kwargs))

    def unexpected_generate(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Offline image evaluation must not invoke generation.')

    monkeypatch.setattr(MockLLM, 'generate', unexpected_generate)
    output = tmp_path / 'output'
    config = TaskConfig(model=None, model_id='offline-image-test', model_task=ModelTask.IMAGE_GENERATION,
                        datasets=['gedit'], dataset_args={'gedit': {
                            'dataset_id': str(dataset), 'subset_list': ['color_alter', 'material_alter'],
                            'extra_params': {'local_file': str(manifest), 'id_key': id_key, 'image_key': image_key},
                        }}, judge={'strategy': 'llm', 'models': [{
                            'model_id': 'offline-judge', 'api_url': 'https://example.test/v1', 'api_key': 'offline-test',
                            'generation_config': {'retries': 1, 'max_tokens': 32},
                        }]}, eval_batch_size=2, work_dir=str(output), no_timestamp=True)
    run_task(config)
    expected_calls = Counter({(red, 'SC'): 1, (red, 'PQ'): 1, (green, 'SC'): 1, (green, 'PQ'): 1})
    assert Counter(calls) == expected_calls
    review_paths = list((output / 'reviews').rglob('*.jsonl'))
    reviews = [json.loads(line) for path in review_paths for line in path.read_text().splitlines()]
    assert len(reviews) == 2
    assert all(row['sample_score']['score']['value'] == {
        'semantic_consistency': 4.0, 'perceptual_similarity': 9.0, 'normalized_score': 6.0} for row in reviews)
    assert {row['sample_score']['score']['prediction'] for row in reviews} == {
        str(image_dir / 'red.png'), str(image_dir / 'green.png')}
    report_path = next((output / 'reports').rglob('gedit.json'))
    report = json.loads(report_path.read_text())
    assert {m['identity']['name']: m['score'] for m in report['metrics']} == {
        'semantic_consistency': 4.0, 'perceptual_similarity': 9.0, 'normalized_score': 6.0}
    assert report['execution_summary']['succeeded'] == 2
    assert (output / 'reports' / 'report.html').is_file()
    predictions = {path: path.read_bytes() for path in (output / 'predictions').rglob('*.jsonl')}
    assert sum(len(data.splitlines()) for data in predictions.values()) == 2
    config.use_cache = str(output)
    run_task(config)
    assert Counter(calls) == expected_calls
    config.rerun_review = True
    run_task(config)
    assert Counter(calls) == expected_calls + expected_calls
    assert json.loads(report_path.read_text())['metrics'] == report['metrics']
    assert all(path.read_bytes() == data for path, data in predictions.items())
