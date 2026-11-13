"""Compiled collection JSONL files use the same loading/evaluation pipeline as snapshot directories."""

import json
import random
from pathlib import Path

import pytest

from evalscope.api.dataset import DatasetHub
from evalscope.api.registry import get_benchmark
from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.config import TaskConfig
from evalscope.constants import DataCollection
from evalscope.models.mockllm import MockLLM
from evalscope.run import run_task


@pytest.fixture
def collection_file(tmp_path: Path) -> Path:
    source = tmp_path / 'source'
    source.mkdir()
    rows = [
        {'id': index, 'question': f'Pick A: {index}', 'A': 'alpha', 'B': 'beta', 'answer': 'A'}
        for index in range(6)
    ]
    (source / 'example.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    schema = CollectionSchema(name='suite', datasets=[DatasetInfo(
        name='general_mcq', task_type='reasoning', tags=['en'],
        args={'local_path': str(source), 'subset_list': ['example']},
    )])
    records = WeightedSampler(schema).sample(6)
    output = tmp_path / '混合集合 files'
    output.mkdir()
    path = output / 'stratified_mixed_data.jsonl'
    path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
    return path


@pytest.mark.parametrize('relative', [False, True])
@pytest.mark.parametrize('hub', ['modelscope', 'huggingface'])
def test_collection_loads_explicit_jsonl_without_snapshot_download(
    collection_file: Path, monkeypatch: pytest.MonkeyPatch, relative: bool, hub: str,
) -> None:
    if relative:
        monkeypatch.chdir(collection_file.parent)
    path = collection_file.name if relative else str(collection_file)

    def unexpected_snapshot(*args: object, **kwargs: object) -> str:
        pytest.fail('An explicit local collection file must not be resolved as a snapshot')

    monkeypatch.setattr(DatasetHub, 'download_snapshot', unexpected_snapshot)
    config = TaskConfig(datasets=['data_collection'], dataset_hub=hub,
                        dataset_args={'data_collection': {'local_path': path}})
    adapter = get_benchmark('data_collection', config)
    dataset = adapter.load_dataset()['default']

    assert len(dataset) == 6
    assert dataset.location == str(collection_file.resolve())
    assert [sample.id for sample in dataset] == list(range(6))
    assert {sample.target for sample in dataset} == {'A'}
    assert all(sample.metadata[DataCollection.INFO]['dataset_name'] == 'general_mcq' for sample in dataset)
    assert set(adapter.dataset_adapters) == {'general_mcq'}


def test_jsonl_symlink_to_extensionless_cache_file(collection_file: Path) -> None:
    blob = collection_file.parent / 'cached-blob'
    blob.write_bytes(collection_file.read_bytes())
    link = collection_file.parent / 'linked_collection.jsonl'
    link.symlink_to(blob.name)
    config = TaskConfig(datasets=['data_collection'],
                        dataset_args={'data_collection': {'local_path': str(link)}})

    dataset = get_benchmark('data_collection', config).load_dataset()['default']

    assert len(dataset) == 6
    assert dataset.location == str(link.absolute())


@pytest.mark.parametrize('relative', [False, True])
def test_parent_component_after_directory_symlink_selects_the_original_file(
    collection_file: Path, monkeypatch: pytest.MonkeyPatch, relative: bool,
) -> None:
    root = collection_file.parent
    child = root / 'real' / 'child'
    child.mkdir(parents=True)
    (root / 'link').symlink_to(child, target_is_directory=True)
    (child.parent / 'collection.jsonl').write_bytes(collection_file.read_bytes())
    # Collapsing link/.. lexically would silently load this different, valid collection.
    (root / 'collection.jsonl').write_text(collection_file.read_text().splitlines()[0] + '\n', encoding='utf-8')
    path = root / 'link' / '..' / 'collection.jsonl'
    if relative:
        monkeypatch.chdir(root)
        path = Path('link/../collection.jsonl')
    config = TaskConfig(datasets=['data_collection'],
                        dataset_args={'data_collection': {'local_path': str(path)}})

    dataset = get_benchmark('data_collection', config).load_dataset()['default']

    assert len(dataset) == 6
    assert {sample.metadata['id'] for sample in dataset} == set(range(6))


@pytest.mark.parametrize('directory', [False, True])
@pytest.mark.parametrize('seed', [0, 17])
def test_file_and_directory_preserve_shuffle_limit_and_repeat_groups(
    collection_file: Path, directory: bool, seed: int,
) -> None:
    path = collection_file
    if directory:
        path = collection_file.parent / 'snapshot'
        path.mkdir()
        (path / 'test.jsonl').write_bytes(collection_file.read_bytes())
    records = [json.loads(line) for line in collection_file.read_text(encoding='utf-8').splitlines()]
    expected_ids = [record['prompt']['metadata']['id'] for record in records]
    random.Random(seed).shuffle(expected_ids)
    config = TaskConfig(
        datasets=['data_collection'], seed=seed, limit=3, repeats=2,
        dataset_args={'data_collection': {'local_path': str(path), 'shuffle': True}},
    )
    dataset = get_benchmark('data_collection', config).load_dataset()['default']
    assert [sample.metadata['id'] for sample in dataset] == [value for value in expected_ids[:3] for _ in range(2)]
    assert [sample.id for sample in dataset] == [0, 1, 2, 3, 4, 5]
    assert [sample.group_id for sample in dataset] == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize('relative', [False, True])
def test_documented_jsonl_path_completes_evaluation_and_cache_reuse(
    collection_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool,
) -> None:
    if relative:
        monkeypatch.chdir(collection_file.parent)
    path = collection_file.name if relative else str(collection_file)
    monkeypatch.setattr(MockLLM, 'default_output', 'A')
    work_dir = tmp_path / 'run'
    config = {
        'model': 'collection-file-mock', 'eval_type': 'mock_llm', 'datasets': ['data_collection'],
        'dataset_args': {'data_collection': {'local_path': path}},
        'no_timestamp': True, 'work_dir': str(work_dir), 'eval_batch_size': 2,
    }
    report = run_task(TaskConfig(**config))['data_collection']
    assert len(report.metrics) == 1
    assert (report.metrics[0].score, report.metrics[0].num) == (1.0, 6)
    predictions = work_dir / 'predictions' / 'collection-file-mock' / 'data_collection_default.jsonl'
    reviews = work_dir / 'reviews' / 'collection-file-mock' / 'data_collection_default.jsonl'
    assert len(predictions.read_text().splitlines()) == 6
    assert len(reviews.read_text().splitlines()) == 6
    before = (predictions.read_bytes(), reviews.read_bytes())
    details = list((work_dir / 'reports').rglob(DataCollection.REPORT_NAME))
    assert len(details) == 1
    subset = json.loads(details[0].read_text())['subset_level'][0]
    assert (subset['micro_avg.'], subset['count']) == (1.0, 6)
    assert (work_dir / 'reports' / 'report.html').is_file()

    def unexpected_inference(*args: object, **kwargs: object) -> None:
        pytest.fail('A repeated collection run should reuse its predictions')

    monkeypatch.setattr(MockLLM, 'generate', unexpected_inference)
    cached = run_task(TaskConfig(**config, use_cache=str(work_dir)))['data_collection']
    assert (cached.metrics[0].score, cached.metrics[0].num) == (1.0, 6)
    assert (predictions.read_bytes(), reviews.read_bytes()) == before
