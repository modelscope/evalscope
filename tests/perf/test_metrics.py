from evalscope.perf.domain.observation import RequestObservation, TokenSource
from evalscope.perf.metrics.definitions import METRICS, MetricProvenance
from evalscope.perf.results.sqlite import SQLiteResultStore
from evalscope.perf.results.summarize import _metric_values, percentile_stats, summarize_store


def test_metric_provenance_distinguishes_estimated_and_reported() -> None:
    assert METRICS['approx_spec_acceptance_rate'].provenance == MetricProvenance.ESTIMATED
    assert METRICS['draft_acceptance_rate'].provenance == MetricProvenance.SERVER_REPORTED


def test_strict_and_estimated_acceptance_are_separate() -> None:
    observation = RequestObservation(
        run_id='run',
        request_id='request',
        success=True,
        start_time=0,
        first_token_time=1,
        completed_time=3,
        completion_tokens=5,
        chunk_times=[1, 2, 3],
        accepted_draft_tokens=3,
        proposed_draft_tokens=4,
        completion_token_source=TokenSource.SERVER_REPORTED,
    )
    values = _metric_values([observation])
    assert values['decoded_tokens_per_iter'] == [2]
    assert values['approx_spec_acceptance_rate'] == [0.5]
    assert values['draft_acceptance_rate'] == [0.75]


def test_percentiles_use_linear_interpolation() -> None:
    stats = percentile_stats([0, 10])
    assert stats.p50 == 5
    assert stats.p99 == 9.9


def test_summary_cache_hit_rate_is_token_weighted(tmp_path) -> None:
    store = SQLiteResultStore(tmp_path / 'observations.sqlite')
    store.open()
    store.write(
        RequestObservation(
            run_id='run',
            request_id='one',
            success=True,
            start_time=0,
            completed_time=1,
            prompt_tokens=100,
            cached_tokens=50,
        )
    )
    store.write(
        RequestObservation(
            run_id='run',
            request_id='two',
            success=True,
            start_time=0,
            completed_time=1,
            prompt_tokens=10,
            cached_tokens=10,
        )
    )
    summary, _, _, _ = summarize_store(store)
    store.close()
    assert summary.averages['cache_hit_rate'] == 60 / 110


def test_summary_excludes_warmup_observations(tmp_path) -> None:
    store = SQLiteResultStore(tmp_path / 'observations.sqlite')
    store.open()
    store.write(
        RequestObservation(
            run_id='run',
            request_id='warmup',
            success=True,
            is_warmup=True,
            start_time=0,
            completed_time=100,
        )
    )
    store.write(
        RequestObservation(
            run_id='run',
            request_id='measured',
            success=True,
            start_time=0,
            completed_time=1,
        )
    )
    summary, _, _, _ = summarize_store(store)
    store.close()
    assert summary.total == 1
    assert summary.duration_seconds == 1
