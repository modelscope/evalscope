"""Offline coverage for interrupted review reruns and their persisted identities."""

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from evalscope.api.evaluator.cache import CacheManager
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.general_mcq.general_mcq_adapter import GeneralMCQAdapter
from evalscope.config import TaskConfig
from evalscope.evaluator.evaluator import DefaultEvaluator
from evalscope.models.mockllm import MockLLM
from evalscope.run import run_task
from evalscope.utils.io_utils import yaml_to_dict

BENCHMARK = 'general_mcq'
SECOND_BENCHMARK = 'review_recovery_mcq'
MODEL = 'review-recovery-model'


@dataclass
class LocalRun:
    work_dir: Path
    dataset_dir: Path

    def config(
        self,
        *,
        pattern: str = 'B',
        datasets: tuple[str, ...] = (BENCHMARK,),
        subsets: tuple[str, ...] = ('default',),
        resume: bool = False,
        **kwargs: Any,
    ) -> TaskConfig:
        return TaskConfig(
            model=MODEL,
            eval_type='mock_llm',
            datasets=list(datasets),
            dataset_args={
                name: {
                    'local_path': str(self.dataset_dir),
                    'subset_list': list(subsets),
                    'filters': {'regex': {'regex_pattern': pattern}},
                }
                for name in datasets
            },
            eval_batch_size=1,
            no_timestamp=True,
            work_dir=str(self.work_dir),
            use_cache=str(self.work_dir) if resume else None,
            **kwargs,
        )

    @property
    def snapshot_path(self) -> Path:
        return self.work_dir / 'configs' / 'task_config.yaml'

    def identities(self) -> dict[str, Any]:
        return yaml_to_dict(str(self.snapshot_path))['evaluation_identity']['benchmarks']

    def cache_path(self, kind: str, benchmark: str = BENCHMARK, subset: str = 'default') -> Path:
        return self.work_dir / kind / MODEL / f'{benchmark}_{subset}.jsonl'


@pytest.fixture
def local_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalRun:
    dataset_dir = tmp_path / 'data'
    dataset_dir.mkdir()
    row = {'id': 'one', 'question': 'Choose the second option.', 'A': 'first', 'B': 'second', 'answer': 'B'}
    (dataset_dir / 'default_val.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    monkeypatch.setattr(MockLLM, 'default_output', 'B A')
    # The regression concerns durable JSON/YAML artifacts; rendering adds no coverage.
    monkeypatch.setattr('evalscope.report.gen_html_report_file', lambda *args, **kwargs: '')
    return LocalRun(work_dir=tmp_path / 'run', dataset_dir=dataset_dir)


def _fail(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError('injected review failure')


def _forbid_prediction(*args: Any, **kwargs: Any) -> Any:
    pytest.fail('A review rerun must reuse the saved predictions')


def _assert_resume_blocked(local_run: LocalRun, **kwargs: Any) -> None:
    snapshot = local_run.snapshot_path.read_bytes()
    with pytest.raises(ValueError) as error:
        run_task(local_run.config(resume=True, **kwargs))
    assert kwargs.get('datasets', (BENCHMARK,))[0] in str(error.value)
    assert 'rerun_review=True' in str(error.value)
    assert local_run.snapshot_path.read_bytes() == snapshot


@pytest.mark.parametrize('pattern', ['B', 'A'])
def test_failed_review_rerun_blocks_resume_until_forced_recovery(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch, pattern: str
) -> None:
    assert run_task(local_run.config())[BENCHMARK].score == 1
    original_identity = local_run.identities()[BENCHMARK]
    prediction = local_run.cache_path('predictions').read_bytes()
    original_review = local_run.cache_path('reviews').read_bytes()
    monkeypatch.setattr(MockLLM, 'generate', _forbid_prediction)

    with monkeypatch.context() as failure:
        failure.setattr(CacheManager, 'commit_review_reruns', _fail)
        with pytest.raises(RuntimeError, match='injected review failure'):
            run_task(local_run.config(pattern=pattern, resume=True, rerun_review=True))

    _assert_resume_blocked(local_run, pattern=pattern)
    pending = local_run.identities()[BENCHMARK]
    assert pending['review_pending'] is True
    assert pending['cache_source']['fingerprint'] == original_identity['fingerprint']
    assert local_run.cache_path('reviews').read_bytes() == original_review
    assert local_run.cache_path('predictions').read_bytes() == prediction

    result = run_task(local_run.config(pattern=pattern, resume=True, rerun_review=True))
    expected_score = 1 if pattern == 'B' else 0
    assert result[BENCHMARK].score == expected_score
    recovered = local_run.identities()[BENCHMARK]
    assert recovered['review_pending'] is False
    assert recovered['cache_source'] == pending['cache_source']
    assert run_task(local_run.config(pattern=pattern, resume=True))[BENCHMARK].score == expected_score
    assert local_run.identities()[BENCHMARK]['cache_source'] == pending['cache_source']
    assert local_run.cache_path('predictions').read_bytes() == prediction


def test_failed_retries_keep_original_prediction_provenance(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_task(local_run.config())
    original = local_run.identities()[BENCHMARK]
    prediction = local_run.cache_path('predictions').read_bytes()
    monkeypatch.setattr(MockLLM, 'generate', _forbid_prediction)

    with monkeypatch.context() as failure:
        failure.setattr(CacheManager, 'commit_review_reruns', _fail)
        for pattern in ('A', '(A)'):
            with pytest.raises(RuntimeError, match='injected review failure'):
                run_task(local_run.config(pattern=pattern, resume=True, rerun_review=True))
            pending = local_run.identities()[BENCHMARK]
            assert pending['cache_source']['fingerprint'] == original['fingerprint']

    assert run_task(local_run.config(pattern='(A)', resume=True, rerun_review=True))[BENCHMARK].score == 0
    assert local_run.identities()[BENCHMARK]['cache_source']['fingerprint'] == original['fingerprint']
    assert local_run.cache_path('predictions').read_bytes() == prediction


@pytest.mark.parametrize('failure_mode', ['load_error', 'empty_dataset', 'keyboard_interrupt'])
def test_rerun_without_a_report_remains_pending(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch, failure_mode: str
) -> None:
    run_task(local_run.config())
    prediction = local_run.cache_path('predictions').read_bytes()
    review = local_run.cache_path('reviews').read_bytes()

    with monkeypatch.context() as failure:
        config = local_run.config(pattern='A', resume=True, rerun_review=True)
        if failure_mode == 'empty_dataset':
            failure.setattr(GeneralMCQAdapter, 'load_dataset', lambda self: {})
            assert run_task(config)[BENCHMARK] == {}
        elif failure_mode == 'load_error':
            failure.setattr(GeneralMCQAdapter, 'load_dataset', _fail)
            with pytest.raises(RuntimeError, match='injected review failure'):
                run_task(config)
        else:
            def interrupt(*args: Any, **kwargs: Any) -> None:
                raise KeyboardInterrupt

            failure.setattr(CacheManager, 'commit_review_reruns', interrupt)
            with pytest.raises(KeyboardInterrupt):
                run_task(config)

    assert local_run.identities()[BENCHMARK]['review_pending'] is True
    assert local_run.cache_path('predictions').read_bytes() == prediction
    assert local_run.cache_path('reviews').read_bytes() == review
    _assert_resume_blocked(local_run, pattern='A')


def test_report_failure_after_review_commit_does_not_clear_pending(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_task(local_run.config())
    original = local_run.identities()[BENCHMARK]
    monkeypatch.setattr(MockLLM, 'generate', _forbid_prediction)
    with monkeypatch.context() as failure:
        failure.setattr(DefaultEvaluator, 'get_report', _fail)
        with pytest.raises(RuntimeError, match='injected review failure'):
            run_task(local_run.config(pattern='A', resume=True, rerun_review=True))

    assert local_run.identities()[BENCHMARK]['review_pending'] is True
    _assert_resume_blocked(local_run, pattern='A')
    assert run_task(local_run.config(pattern='A', resume=True, rerun_review=True))[BENCHMARK].score == 0
    recovered = local_run.identities()[BENCHMARK]
    assert recovered['review_pending'] is False
    assert recovered['cache_source']['fingerprint'] == original['fingerprint']


def test_partial_benchmark_success_preserves_unselected_pending_identity(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evalscope.api.registry as registry

    get_benchmark = registry.get_benchmark

    def with_alias(name: str, config: TaskConfig | None = None) -> Any:
        if name != SECOND_BENCHMARK:
            return get_benchmark(name, config)
        metadata = copy.deepcopy(BENCHMARK_REGISTRY.get(BENCHMARK))
        metadata.name = name
        if config is not None:
            metadata._update(config.dataset_args.get(name, {}))
        return GeneralMCQAdapter(benchmark_meta=metadata, task_config=config)

    monkeypatch.setattr(registry, 'get_benchmark', with_alias)
    datasets = (BENCHMARK, SECOND_BENCHMARK)
    run_task(local_run.config(datasets=datasets))
    originals = local_run.identities()
    predictions = {name: local_run.cache_path('predictions', name).read_bytes() for name in datasets}
    monkeypatch.setattr(MockLLM, 'generate', _forbid_prediction)
    commit = CacheManager.commit_review_reruns

    def fail_second(manager: CacheManager) -> None:
        if manager.benchmark_name == SECOND_BENCHMARK:
            _fail()
        commit(manager)

    with monkeypatch.context() as failure:
        failure.setattr(CacheManager, 'commit_review_reruns', fail_second)
        with pytest.raises(RuntimeError, match='injected review failure'):
            run_task(local_run.config(datasets=datasets, pattern='A', resume=True, rerun_review=True))

    identities = local_run.identities()
    assert identities[BENCHMARK]['review_pending'] is False
    assert identities[SECOND_BENCHMARK]['review_pending'] is True
    assert run_task(local_run.config(pattern='A', resume=True))[BENCHMARK].score == 0
    assert local_run.identities()[SECOND_BENCHMARK] == identities[SECOND_BENCHMARK]
    _assert_resume_blocked(local_run, pattern='A', datasets=(SECOND_BENCHMARK,))

    result = run_task(local_run.config(datasets=(SECOND_BENCHMARK,), pattern='A', resume=True, rerun_review=True))
    assert result[SECOND_BENCHMARK].score == 0
    recovered = local_run.identities()
    for name in datasets:
        assert recovered[name]['review_pending'] is False
        assert recovered[name]['cache_source']['fingerprint'] == originals[name]['fingerprint']
        assert local_run.cache_path('predictions', name).read_bytes() == predictions[name]


def test_ignored_review_errors_replace_stale_scores_with_empty_cache(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_task(local_run.config())[BENCHMARK].score == 1
    prediction = local_run.cache_path('predictions').read_bytes()
    monkeypatch.setattr(MockLLM, 'generate', _forbid_prediction)
    with monkeypatch.context() as failure:
        failure.setattr(DefaultEvaluator, '_review_task_state', _fail)
        report = run_task(local_run.config(pattern='A', resume=True, rerun_review=True, ignore_errors=True))[BENCHMARK]

    assert report.execution_summary.incomplete is True
    assert report.execution_summary.errored == 1
    assert local_run.cache_path('reviews').read_bytes() == b''
    recovered = local_run.identities()[BENCHMARK]
    assert recovered['review_pending'] is False
    assert run_task(local_run.config(pattern='A', resume=True))[BENCHMARK].score == 0
    assert local_run.identities()[BENCHMARK]['cache_source'] == recovered['cache_source']
    assert local_run.cache_path('predictions').read_bytes() == prediction


def test_failed_snapshot_publication_preserves_old_identity_and_never_starts_review(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_task(local_run.config())
    snapshot = local_run.snapshot_path.read_bytes()
    prediction = local_run.cache_path('predictions').read_bytes()
    review = local_run.cache_path('reviews').read_bytes()

    def fail_write(payload: Any, path: str) -> None:
        Path(path).write_text('partial: [', encoding='utf-8')
        raise OSError('injected snapshot failure')

    def forbid_eval(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('Evaluation must not start when its pending identity cannot be persisted')

    monkeypatch.setattr('evalscope.config.dict_to_yaml', fail_write)
    monkeypatch.setattr(DefaultEvaluator, 'eval', forbid_eval)
    with pytest.raises(OSError, match='injected snapshot failure'):
        run_task(local_run.config(pattern='A', resume=True, rerun_review=True))

    assert local_run.snapshot_path.read_bytes() == snapshot
    assert local_run.cache_path('predictions').read_bytes() == prediction
    assert local_run.cache_path('reviews').read_bytes() == review


def test_failed_completion_snapshot_keeps_rerun_pending(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_task(local_run.config())
    original = local_run.identities()[BENCHMARK]
    prediction = local_run.cache_path('predictions').read_bytes()
    dump_yaml = TaskConfig.dump_yaml

    def fail_completion(
        config: TaskConfig,
        output_dir: str,
        generated_metadata: dict[str, Any] | None = None,
        *,
        strict: bool = False,
    ) -> None:
        identity = generated_metadata['evaluation_identity']['benchmarks'][BENCHMARK]
        if not identity['review_pending']:
            raise OSError('injected completion snapshot failure')
        dump_yaml(config, output_dir, generated_metadata, strict=strict)

    monkeypatch.setattr(MockLLM, 'generate', _forbid_prediction)
    with monkeypatch.context() as failure:
        failure.setattr(TaskConfig, 'dump_yaml', fail_completion)
        with pytest.raises(OSError, match='injected completion snapshot failure'):
            run_task(local_run.config(pattern='A', resume=True, rerun_review=True))

    assert local_run.identities()[BENCHMARK]['review_pending'] is True
    _assert_resume_blocked(local_run, pattern='A')
    assert run_task(local_run.config(pattern='A', resume=True, rerun_review=True))[BENCHMARK].score == 0
    recovered = local_run.identities()[BENCHMARK]
    assert recovered['review_pending'] is False
    assert recovered['cache_source']['fingerprint'] == original['fingerprint']
    assert local_run.cache_path('predictions').read_bytes() == prediction


def test_empty_subset_during_rerun_does_not_keep_stale_reviews(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    subsets = ('default', 'second')
    (local_run.dataset_dir / 'second_val.jsonl').write_bytes(
        (local_run.dataset_dir / 'default_val.jsonl').read_bytes()
    )
    assert run_task(local_run.config(subsets=subsets))[BENCHMARK].score == 1
    predictions = {subset: local_run.cache_path('predictions', subset=subset).read_bytes() for subset in subsets}
    load_dataset = GeneralMCQAdapter.load_dataset

    def load_with_empty_subset(adapter: GeneralMCQAdapter) -> dict[str, Any]:
        datasets = load_dataset(adapter)
        datasets['second'] = datasets['second'].filter(lambda sample: False)
        return datasets

    monkeypatch.setattr(MockLLM, 'generate', _forbid_prediction)
    with monkeypatch.context() as empty_subset:
        empty_subset.setattr(GeneralMCQAdapter, 'load_dataset', load_with_empty_subset)
        assert run_task(local_run.config(subsets=subsets, pattern='A', resume=True, rerun_review=True))[BENCHMARK].score == 0

    assert local_run.identities()[BENCHMARK]['review_pending'] is False
    assert local_run.cache_path('reviews', subset='second').read_bytes() == b''
    assert run_task(local_run.config(subsets=subsets, pattern='A', resume=True))[BENCHMARK].score == 0
    for subset in subsets:
        assert local_run.cache_path('predictions', subset=subset).read_bytes() == predictions[subset]


def test_partial_subset_commit_failure_requires_explicit_recovery(
    local_run: LocalRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    subsets = ('default', 'second')
    (local_run.dataset_dir / 'second_val.jsonl').write_bytes(
        (local_run.dataset_dir / 'default_val.jsonl').read_bytes()
    )
    run_task(local_run.config(subsets=subsets))
    old_reviews = {subset: local_run.cache_path('reviews', subset=subset).read_bytes() for subset in subsets}
    replace = os.replace

    def fail_second_subset(source: str, destination: str) -> None:
        if '.rerun-' in source and destination.endswith('_second.jsonl'):
            raise OSError('injected subset commit failure')
        replace(source, destination)

    monkeypatch.setattr(MockLLM, 'generate', _forbid_prediction)
    with monkeypatch.context() as failure:
        failure.setattr(os, 'replace', fail_second_subset)
        with pytest.raises(OSError, match='injected subset commit failure'):
            run_task(local_run.config(subsets=subsets, pattern='A', resume=True, rerun_review=True))

    assert local_run.cache_path('reviews').read_bytes() != old_reviews['default']
    assert local_run.cache_path('reviews', subset='second').read_bytes() == old_reviews['second']
    _assert_resume_blocked(local_run, subsets=subsets, pattern='A')
    assert run_task(local_run.config(subsets=subsets, pattern='A', resume=True, rerun_review=True))[BENCHMARK].score == 0
    assert local_run.identities()[BENCHMARK]['review_pending'] is False
