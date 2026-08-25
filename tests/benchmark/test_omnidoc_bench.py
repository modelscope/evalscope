import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from evalscope.api.dataset import DictDataLoader
from evalscope.api.metric import MetricSelector, SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.omnidoc_bench.legacy.omnidoc_bench_adapter import OmniDocBenchAdapter
from evalscope.config import TaskConfig


def test_v16_loads_snapshot_once_and_reads_images_locally(tmp_path: Path) -> None:
    image_names = ['page.png', 'long_' + '页面' * 36 + '.png']
    records = [{'page_info': {'image_path': image_name}} for image_name in image_names]
    (tmp_path / 'OmniDocBench.json').write_text(json.dumps(records), encoding='utf-8')
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    for image_name in image_names:
        (image_dir / image_name).write_bytes(b'image')

    adapter = get_benchmark(
        'omni_doc_bench_v1_6',
        TaskConfig(datasets=['omni_doc_bench_v1_6']),
    )
    with mock.patch(
        'evalscope.benchmarks.omnidoc_bench.v1_6.omnidoc_bench_v1_6_adapter.resolve_snapshot_or_local_path',
        return_value=str(tmp_path),
    ) as resolve_snapshot:
        dataset = adapter.load_subset('default', DictDataLoader)

    resolve_snapshot.assert_called_once_with(adapter)
    assert [sample.metadata['image_name'] for sample in dataset] == image_names
    assert all(sample.input[0].content[0].image.startswith('data:image/png;base64,') for sample in dataset)


def test_legacy_omnidoc_aggregates_canonical_metrics() -> None:
    selector = MetricSelector(name='normalized_score', aggregation='macro_mean')
    adapter = OmniDocBenchAdapter.__new__(OmniDocBenchAdapter)
    adapter._benchmark_meta = SimpleNamespace(metric_list=[], primary_metric=selector)
    adapter.match_method = 'quick_match'
    sample_scores = [SampleScore(sample_id=1, score=Score(prediction='markdown', metadata={'reference': {}}))]
    raw_scores = {
        'text_block_Edit_dist_EN': 0.2,
        'table_TEDS_CH': 0.8,
        'overall_EN': 0.7,
        'overall_CH': 0.9,
    }

    with mock.patch(
        'evalscope.benchmarks.omnidoc_bench.legacy.end2end_eval.End2EndEvaluator.score',
        return_value=raw_scores,
    ):
        aggregated = adapter.aggregate_scores(sample_scores)

    identities = [score.identity for score in aggregated]
    assert any(
        identity.name == 'text_block_edit_dist' and identity.dimensions == {'language': 'en'} for identity in identities
    )
    assert any(identity.name == 'table_teds' and identity.dimensions == {'language': 'ch'} for identity in identities)
    primary_matches = [score for score in aggregated if selector.matches(score.identity)]
    assert len(primary_matches) == 1
    assert primary_matches[0].score == pytest.approx(0.8)
