import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from evalscope.api.dataset import DictDataLoader
from evalscope.api.metric import MetricSelector, SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.omnidoc_bench.legacy.omnidoc_bench_adapter import OmniDocBenchAdapter
from evalscope.benchmarks.omnidoc_bench.v1_6.omnidoc_bench_v1_6_adapter import OmniDocBenchV16Adapter
from evalscope.benchmarks.omnidoc_bench.v1_6.sandbox_scorer import RESULT_SENTINEL, parse_scoring_result
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus


def _sandbox_result(metrics: dict[str, float]) -> dict[str, str]:
    return {'status': 'success', 'output': RESULT_SENTINEL + json.dumps(metrics)}


def _v16_adapter_with_result(result: dict[str, str]) -> OmniDocBenchV16Adapter:
    adapter = OmniDocBenchV16Adapter.__new__(OmniDocBenchV16Adapter)
    adapter._task_config = SimpleNamespace(sandbox=SimpleNamespace(enabled=True))
    adapter._benchmark_meta = SimpleNamespace(review_timeout=1)
    adapter.execute_code_in_sandbox = mock.Mock(return_value=result)
    return adapter


@pytest.mark.parametrize(
    ('metrics', 'expected'),
    [({}, {}), ({'text_block_Edit_dist': 0.25}, {'text_block_Edit_dist': 0.25})],
)
def test_v16_parse_scoring_result_accepts_empty_and_normal_results(
    metrics: dict[str, float], expected: dict[str, float]
) -> None:
    assert parse_scoring_result(_sandbox_result(metrics)) == expected


@pytest.mark.parametrize(
    ('metrics', 'status', 'main_score_name'),
    [({}, ScoreStatus.EXCLUDED, None), ({'text_block_Edit_dist': 0.25}, ScoreStatus.SUCCESS, 'text_block_Edit_dist')],
)
def test_v16_match_score_handles_pages_without_supported_metrics(
    metrics: dict[str, float], status: ScoreStatus, main_score_name: str | None
) -> None:
    adapter = _v16_adapter_with_result(_sandbox_result(metrics))
    task_state = SimpleNamespace(metadata={'image_name': 'page.png'})

    score = adapter.match_score('prediction', 'prediction', '{}', task_state)

    assert score.value == metrics
    assert score.status is status
    assert score.main_score_name == main_score_name
    if not metrics:
        assert score.metadata == {'scoring_excluded_reason': 'no_page_metrics'}


def test_v16_aggregation_excludes_unscored_pages_from_normalized_score_count() -> None:
    adapter = OmniDocBenchV16Adapter.__new__(OmniDocBenchV16Adapter)
    sample_scores = [
        SampleScore(
            sample_id=1,
            score=Score(
                value={
                    'text_block_Edit_dist': 0.2,
                    'display_formula_CDM': 0.8,
                    'table_TEDS': 0.7,
                }
            ),
        ),
        SampleScore(sample_id=2, score=Score(status=ScoreStatus.EXCLUDED)),
    ]

    aggregated = adapter.aggregate_scores(sample_scores)

    normalized_score = next(score for score in aggregated if score.metric_name == 'normalized_score')
    assert normalized_score.num == 1
    assert normalized_score.ids == [1]


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
