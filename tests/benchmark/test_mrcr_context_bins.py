"""MRCR context buckets must not absorb out-of-range samples or cached bin assignments."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.evaluator.cache import ModelResult
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.openai_mrcr.openai_mrcr_adapter import OpenAIMRCRAdapter
from evalscope.benchmarks.openai_mrcr.utils import bin_index_for
from evalscope.config import TaskConfig
from evalscope.evaluation_versioning import ResolvedBenchmarkSpec, build_benchmark_identity
from evalscope.models.mockllm import MockLLM
from evalscope.report import Report
from evalscope.run import run_task
from evalscope.utils.io_utils import dict_to_yaml, yaml_to_dict


class Tokenizer:
    """Provide controlled lengths without downloading tokenizer assets."""

    def encode(self, text: str) -> range:
        return range(int(text.removeprefix('tokens:')) if text.startswith('tokens:') else 1)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> OpenAIMRCRAdapter:
    monkeypatch.setattr('evalscope.benchmarks.openai_mrcr.openai_mrcr_adapter.check_import', lambda **kwargs: None)
    result = get_benchmark('openai_mrcr', TaskConfig(model='mock', eval_type='mock_llm', datasets=['openai_mrcr']))
    result.tik_enc = Tokenizer()
    return result


@pytest.mark.parametrize(('tokens', 'expected'), [
    (0, None), (4095, None), (4096, 0), (8191, 0), (8192, 0), (8193, 1),
    (16383, 1), (16384, 1), (16385, 2), (32768, 2), (32769, 3), (65536, 3),
    (65537, 4), (131072, 4), (131073, 5), (262144, 5), (262145, 6),
    (524288, 6), (524289, 7), (1048576, 7), (1048577, None), (2097152, None),
])
def test_bucket_boundaries_match_official_card(tokens: int, expected: int | None) -> None:
    assert bin_index_for(tokens) == expected


def make_state(adapter: OpenAIMRCRAdapter, tokens: int, prediction: str, index: int = 0) -> TaskState:
    sample = adapter.record_to_sample({
        'prompt': json.dumps([{'role': 'user', 'content': f'tokens:{tokens - 1}'}]),
        'answer': 'X', 'n_needles': 2,
    })
    sample.id = index
    return TaskState(model='mock', sample=sample, output=ModelOutput.from_content('mock', prediction), completed=True)


def summarize(adapter: OpenAIMRCRAdapter, scores: list[SampleScore]) -> list[tuple]:
    return [(a.dimensions, a.score, a.num) for a in adapter.aggregate_scores(scores)]


def test_native_scoring_and_report_keep_outliers_only_in_overall(adapter: OpenAIMRCRAdapter, tmp_path: Path) -> None:
    scores = [adapter.calculate_metrics(make_state(adapter, n, p, i))
              for i, (n, p) in enumerate([(5001, 'X'), (2097152, 'Y'), (100, 'Y')])]
    assert summarize(adapter, scores) == [
        ({'scope': 'overall'}, 1 / 3, 3),
        ({'max_tokens': 8192, 'min_tokens': 4096}, 1.0, 1),
    ]
    report = adapter.generate_report({'default': adapter.aggregate_scores(scores)}, 'mock', str(tmp_path))
    assert report.score == pytest.approx(1 / 3, abs=0.0001)


def test_all_outliers_still_produce_overall(adapter: OpenAIMRCRAdapter) -> None:
    scores = [adapter.calculate_metrics(make_state(adapter, n, p, i))
              for i, (n, p) in enumerate([(100, 'X'), (2097152, 'Y')])]
    assert summarize(adapter, scores) == [({'scope': 'overall'}, 0.5, 2)]


@pytest.mark.parametrize('metadata', [None, {}, {'bin_index': None}, {'bin_index': -1}, {'bin_index': 8}])
def test_missing_or_unknown_bin_does_not_drop_a_scored_sample(adapter: OpenAIMRCRAdapter, metadata: dict | None) -> None:
    scores = [SampleScore(sample_id=0, score=Score(value={'mrcr_score': 0.25}, metadata=metadata)),
              SampleScore(sample_id=1, score=Score(value={}, metadata={'bin_index': 0}))]
    assert summarize(adapter, scores) == [({'scope': 'overall'}, 0.25, 1)]
    assert adapter.aggregate_scores([]) == []


@pytest.mark.parametrize(('tokens', 'bounds'), [(100, None), (2097152, None), (5001, (4096, 8192)), (16384, (8192, 16384))])
def test_rereview_repairs_legacy_prediction_bin_and_survives_concurrency(
    adapter: OpenAIMRCRAdapter, tokens: int, bounds: tuple[int, int] | None
) -> None:
    state = make_state(adapter, tokens, 'X')
    fresh = MemoryDataset([Sample(id=0, input=state.input, target='X', metadata=dict(state.metadata))])
    legacy = ModelResult.from_task_state(state)
    legacy.metadata['bin_index'] = 0  # v1.0's erroneous fallback, persisted in predictions.
    cached = ModelResult.model_validate_json(legacy.model_dump_json())
    restored = cached.to_task_state(fresh)
    assert restored.metadata['bin_index'] == 0
    barrier = Barrier(8)

    def review(_: int) -> list[tuple]:
        barrier.wait(timeout=10)
        score = adapter.calculate_metrics(restored)
        roundtrip = SampleScore.model_validate_json(score.model_dump_json())
        return summarize(adapter, [roundtrip])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(review, range(16)))
    expected = [({'scope': 'overall'}, 1.0, 1)]
    if bounds is not None:
        expected.append(({'min_tokens': bounds[0], 'max_tokens': bounds[1]}, 1.0, 1))
    assert all(result == expected for result in results)
    assert restored.metadata['bin_index'] == 0
    assert adapter._benchmark_meta.evaluation_version == 'v1.1'


def test_context_filters_remain_based_on_input_length(adapter: OpenAIMRCRAdapter) -> None:
    adapter.max_context_size = 5000
    assert make_state(adapter, 5001, 'X').metadata['bin_index'] == 0
    record = {'prompt': json.dumps([{'role': 'user', 'content': 'tokens:5001'}]), 'answer': 'X'}
    assert adapter.record_to_sample(record) == []
    adapter.max_context_size = None
    adapter.min_context_size = 5001
    assert adapter.record_to_sample(record) == []


def test_missing_token_count_does_not_reuse_an_unverifiable_bin(adapter: OpenAIMRCRAdapter) -> None:
    state = make_state(adapter, 5001, 'X')
    del state.metadata['raw_input_tok_cnt']
    assert summarize(adapter, [adapter.calculate_metrics(state)]) == [({'scope': 'overall'}, 1.0, 1)]


def test_native_run_and_legacy_cache_rereview(
    adapter: OpenAIMRCRAdapter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import datasets

    monkeypatch.setitem(sys.modules, 'tiktoken', SimpleNamespace(get_encoding=lambda _: Tokenizer(), Encoding=Tokenizer))
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', str(tmp_path / 'dataset-cache'))
    monkeypatch.setattr(MockLLM, 'default_output', 'X')
    data_path = tmp_path / 'data'
    data_path.mkdir()
    records = [
        {'prompt': json.dumps([{'role': 'user', 'content': f'tokens:{n - 1}'}]), 'answer': answer}
        for n, answer in [(5001, 'X'), (100, 'Y'), (2097152, 'Y')]
    ]
    (data_path / 'train.jsonl').write_text('\n'.join(json.dumps(record) for record in records))
    output_path = tmp_path / 'output'
    base = {
        'model': 'mock', 'eval_type': 'mock_llm', 'datasets': ['openai_mrcr'],
        'dataset_args': {'openai_mrcr': {'local_path': str(data_path)}},
        'eval_batch_size': 8, 'no_timestamp': True, 'work_dir': str(output_path),
    }
    run_task(TaskConfig(**base))
    report_path = output_path / 'reports' / 'mock' / 'openai_mrcr.json'
    first_report = json.loads(report_path.read_text())
    assert Report.from_dict(first_report).score == pytest.approx(1 / 3, abs=0.0001)
    assert [(metric['identity']['dimensions'], metric['score'], metric['num']) for metric in first_report['metrics']] == [
        ({'scope': 'overall'}, 0.3333, 3),
        ({'min_tokens': 4096, 'max_tokens': 8192}, 1.0, 1),
    ]

    prediction_path = output_path / 'predictions' / 'mock' / 'openai_mrcr_default.jsonl'
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    for prediction in predictions:
        prediction['metadata']['bin_index'] = 0
    prediction_path.write_text('\n'.join(json.dumps(prediction) for prediction in predictions) + '\n')
    legacy_predictions = prediction_path.read_text()
    snapshot_path = output_path / 'configs' / 'task_config.yaml'
    snapshot = yaml_to_dict(str(snapshot_path))
    spec = ResolvedBenchmarkSpec.model_validate(snapshot['resolved_benchmarks']['openai_mrcr'])
    legacy_identity = build_benchmark_identity(spec, 'v1.0', TaskConfig(**base))
    snapshot['evaluation_identity']['benchmarks']['openai_mrcr'] = legacy_identity.model_dump(mode='json')
    dict_to_yaml(snapshot, str(snapshot_path))

    with pytest.raises(ValueError, match='rerun_review=True'):
        run_task(TaskConfig(**base, use_cache=str(output_path)))

    def unexpected_inference(*args: object, **kwargs: object) -> None:
        pytest.fail('Re-review must reuse predictions without model inference')

    monkeypatch.setattr(MockLLM, 'generate', unexpected_inference)
    run_task(TaskConfig(**base, use_cache=str(output_path), rerun_review=True))
    report = json.loads(report_path.read_text())
    assert Report.from_dict(report).score == Report.from_dict(first_report).score
    assert report['metrics'] == first_report['metrics']
    assert prediction_path.read_text() == legacy_predictions
    identity = yaml_to_dict(str(snapshot_path))['evaluation_identity']['benchmarks']['openai_mrcr']
    assert identity['evaluation_version'] == 'v1.1'
    assert identity['cache_source']['evaluation_version'] == 'v1.0'
