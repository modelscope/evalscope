"""Collection review caches must track the scoring semantics of their member benchmarks."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.config import TaskConfig
from evalscope.evaluation_versioning import ResolvedBenchmarkSpec, build_benchmark_identity
from evalscope.models.mockllm import MockLLM
from evalscope.run import run_task
from evalscope.utils.io_utils import dict_to_yaml, yaml_to_dict


@pytest.fixture
def collection_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'example.jsonl').write_text(
        json.dumps({'question': 'Pick A', 'A': 'alpha', 'B': 'beta', 'answer': 'A'}) + '\n'
    )
    schema = CollectionSchema(name='suite', datasets=[DatasetInfo(
        name='general_mcq', task_type='reasoning', tags=['en'],
        args={'local_path': str(source), 'subset_list': ['example']},
    )])
    collection = tmp_path / 'collection'
    collection.mkdir()
    (collection / 'test.jsonl').write_text(json.dumps(WeightedSampler(schema).sample(1)[0]) + '\n')
    monkeypatch.setattr(MockLLM, 'default_output', 'A')
    return {
        'model': 'collection-mock', 'eval_type': 'mock_llm', 'datasets': ['data_collection'],
        'dataset_args': {'data_collection': {'local_path': str(collection)}},
        'no_timestamp': True, 'work_dir': str(tmp_path / 'run'),
    }


def _snapshot(config: dict) -> Path:
    return Path(config['work_dir']) / 'configs' / 'task_config.yaml'


def _identity(config: dict) -> dict:
    return yaml_to_dict(str(_snapshot(config)))['evaluation_identity']['benchmarks']['data_collection']


@pytest.mark.parametrize('change', ['version', 'scoring_config'])
def test_child_semantics_change_rejects_old_reviews_before_writes(
    collection_config: dict, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    run_task(TaskConfig(**collection_config))
    root = Path(collection_config['work_dir'])
    protected = [_snapshot(collection_config), *root.rglob('*.jsonl')]
    before = {path: path.read_bytes() for path in protected}
    changed = copy.deepcopy(collection_config)
    if change == 'version':
        monkeypatch.setattr(BENCHMARK_REGISTRY.get('general_mcq'), 'evaluation_version', 'v9.9')
    else:
        changed['dataset_args']['general_mcq'] = {'extra_params': {'multiple_correct': True}}

    with pytest.raises(ValueError, match='Cannot reuse cache for data_collection'):
        run_task(TaskConfig(**changed, use_cache=changed['work_dir']))

    assert {path: path.read_bytes() for path in protected} == before


def test_unchanged_child_semantics_reuses_cache(collection_config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    run_task(TaskConfig(**collection_config))
    before = _identity(collection_config)

    def unexpected_inference(*args: object, **kwargs: object) -> None:
        pytest.fail('An unchanged collection should reuse its predictions')

    monkeypatch.setattr(MockLLM, 'generate', unexpected_inference)
    changed = copy.deepcopy(collection_config)
    # Unused member configuration and transport concurrency must not invalidate this collection.
    changed['dataset_args']['general_qa'] = {'prompt_template': 'unused'}
    run_task(TaskConfig(**changed, use_cache=changed['work_dir'], eval_batch_size=8))
    assert _identity(changed) == before


def test_rerun_review_reuses_predictions_and_records_new_child_semantics(collection_config: dict) -> None:
    run_task(TaskConfig(**collection_config))
    root = Path(collection_config['work_dir'])
    prediction = root / 'predictions' / 'collection-mock' / 'data_collection_default.jsonl'
    review = root / 'reviews' / 'collection-mock' / 'data_collection_default.jsonl'
    previous_prediction = prediction.read_bytes()
    previous_review = review.read_bytes()
    previous = _identity(collection_config)
    changed = copy.deepcopy(collection_config)
    changed['dataset_args']['general_mcq'] = {'extra_params': {'multiple_correct': True}}
    run_task(TaskConfig(**changed, use_cache=changed['work_dir'], rerun_review=True))

    current = _identity(changed)
    assert current['fingerprint'] != previous['fingerprint']
    assert current['cache_source']['fingerprint'] == previous['fingerprint']
    assert current['review_pending'] is False
    assert prediction.read_bytes() == previous_prediction
    assert review.read_bytes() != previous_review
    assert 'multi_choice_acc' in review.read_text()
    snapshot = yaml_to_dict(str(_snapshot(changed)))
    dependency = snapshot['resolved_benchmarks']['data_collection']['dependencies']['general_mcq']
    assert dependency['evaluation_version'] == 'v1.1'
    assert dependency['benchmark']['extra_params']['multiple_correct'] is True


def test_old_collection_identity_requires_explicit_review_rerun(collection_config: dict) -> None:
    run_task(TaskConfig(**collection_config))
    config = TaskConfig(**collection_config)
    adapter = get_benchmark('data_collection', config)
    old_spec = ResolvedBenchmarkSpec.from_meta(adapter.benchmark_meta, config)
    old_identity = build_benchmark_identity(old_spec, adapter.benchmark_meta.evaluation_version, config)
    snapshot = yaml_to_dict(str(_snapshot(collection_config)))
    snapshot['resolved_benchmarks']['data_collection'] = old_spec.model_dump(mode='json', exclude={'dependencies'})
    snapshot['evaluation_identity']['benchmarks']['data_collection'] = old_identity.model_dump(mode='json')
    dict_to_yaml(snapshot, str(_snapshot(collection_config)))

    with pytest.raises(ValueError, match='rerun_review=True'):
        run_task(TaskConfig(**collection_config, use_cache=collection_config['work_dir']))
    run_task(TaskConfig(**collection_config, use_cache=collection_config['work_dir'], rerun_review=True))
    assert _identity(collection_config)['cache_source']['fingerprint'] == old_identity.fingerprint


def test_concurrent_collection_resolution_keeps_child_configs_isolated(collection_config: dict) -> None:
    def resolve(multiple_correct: bool) -> tuple[str, bool]:
        config = TaskConfig(**copy.deepcopy(collection_config))
        config.dataset_args['general_mcq'] = {'extra_params': {'multiple_correct': multiple_correct}}
        adapter = get_benchmark('data_collection', config)
        spec = ResolvedBenchmarkSpec.from_adapter(adapter, config)
        identity = build_benchmark_identity(spec, adapter.benchmark_meta.evaluation_version, config)
        return identity.fingerprint, spec.dependencies['general_mcq'].benchmark.extra_params['multiple_correct']

    single = resolve(False)
    multiple = resolve(True)
    assert single[0] != multiple[0]
    flags = [False, True] * 16
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(resolve, flags))
    assert results == [multiple if flag else single for flag in flags]


def test_identity_resolution_and_evaluation_share_loaded_collection(
    collection_config: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TaskConfig(**collection_config)
    adapter = get_benchmark('data_collection', config)
    load = adapter.load
    loads = []

    def counted_load() -> tuple:
        result = load()
        loads.append(result)
        return result

    monkeypatch.setattr(adapter, 'load', counted_load)
    spec = ResolvedBenchmarkSpec.from_adapter(adapter, config)
    members = adapter.dataset_adapters.copy()
    dataset = adapter.load_dataset()
    assert len(loads) == 1
    assert dataset is loads[0][0]
    assert adapter.dataset_adapters == members
    assert set(spec.dependencies) == {'general_mcq'}


def test_ordinary_adapter_identity_does_not_load_data(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TaskConfig(model='mock', eval_type='mock_llm', datasets=['general_mcq'])
    adapter = get_benchmark('general_mcq', config)

    def unexpected_load() -> None:
        pytest.fail('Resolving an ordinary benchmark must remain metadata-only')

    monkeypatch.setattr(adapter, 'load_dataset', unexpected_load)
    spec = ResolvedBenchmarkSpec.from_adapter(adapter, config)
    assert spec == ResolvedBenchmarkSpec.from_meta(adapter.benchmark_meta, config)
    assert not spec.dependencies
