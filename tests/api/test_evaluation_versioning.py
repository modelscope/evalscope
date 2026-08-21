"""Tests for native evaluation snapshots, identities, and analysis context."""

from types import SimpleNamespace

import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.config import TaskConfig, load_task_config_snapshot, parse_task_config
from evalscope.evaluation_versioning import (
    BenchmarkEvaluationIdentity,
    EvaluationIdentity,
    FrameworkProvenance,
    ResolvedBenchmarkSpec,
    build_analysis_context,
    build_benchmark_identity,
    build_evaluation_identity,
    cache_source_for_identity,
    legacy_identity_from_config,
    validate_cached_evaluation_identity,
)
from evalscope.report import Report
from evalscope.run import run_task
from evalscope.utils.io_utils import dict_to_yaml


def _task_config(**kwargs) -> TaskConfig:
    return TaskConfig(
        model='test-model',
        eval_type='mock_llm',
        datasets=['demo'],
        dataset_args={'demo': {'local_path': 'custom/demo'}},
        **kwargs,
    )


def _meta(**kwargs) -> BenchmarkMeta:
    return BenchmarkMeta(
        name='demo',
        dataset_id='demo/data',
        eval_split='validation',
        metric_list=['accuracy'],
        description=(
            '## Overview\nA concise benchmark overview.\n\n'
            '## Task Description\n- Task Type: Multiple choice\n\n'
            '## Key Features\nThis must not be sent to analysis.\n\n'
            '## Evaluation Notes\nThis must not be sent to analysis either.'
        ),
        **kwargs,
    )


def test_benchmark_evaluation_version_has_valid_default_and_is_not_doc_metadata() -> None:
    meta = _meta()

    assert meta.evaluation_version == 'v1.0'
    assert 'evaluation_version' not in meta.to_dict()
    assert 'evaluation_version' not in meta.to_string_dict()

    with pytest.raises(ValueError, match='v<major>.<minor>'):
        _meta(evaluation_version='1.0')

    with pytest.raises(ValueError, match='declared by BenchmarkMeta'):
        meta._update({'evaluation_version': 'v1.1'})


def test_snapshot_dump_preserves_raw_dataset_args_and_reparses(tmp_path) -> None:
    config = _task_config()
    spec = ResolvedBenchmarkSpec.from_meta(_meta(), config)
    identity = build_benchmark_identity(spec, 'v1.0', config)

    config.dump_yaml(
        str(tmp_path),
        generated_metadata={
            'resolved_benchmarks': {'demo': spec.model_dump(mode='json')},
            'evaluation_identity': {
                'schema_version': 1,
                'framework': {'evalscope_version': 'test'},
                'benchmarks': {'demo': identity.model_dump(mode='json')},
            },
        },
    )

    snapshot = (tmp_path / 'task_config.yaml').read_text()
    reparsed = parse_task_config(str(tmp_path / 'task_config.yaml'))

    assert 'resolved_benchmarks:' in snapshot
    assert 'evaluation_identity:' in snapshot
    assert reparsed.dataset_args == {'demo': {'local_path': 'custom/demo'}}


def test_identity_changes_only_for_evaluation_semantics() -> None:
    config = _task_config(
        api_key='secret',
        model_args={'api_key': 'nested-secret', 'headers': {'X-API-Key': 'header-secret'}},
    )
    spec = ResolvedBenchmarkSpec.from_meta(_meta(), config)
    first = build_benchmark_identity(spec, 'v1.0', config)

    same_without_secret = _task_config(
        api_key='different-secret',
        model_args={'api_key': 'other-secret', 'headers': {'X-API-Key': 'other-header-secret'}},
    )
    assert first.fingerprint == build_benchmark_identity(spec, 'v1.0', same_without_secret).fingerprint

    assert first.fingerprint != build_benchmark_identity(spec, 'v1.1', config).fingerprint
    assert first.fingerprint != build_benchmark_identity(spec, 'v1.0', _task_config(seed=7)).fingerprint
    assert first.fingerprint != build_benchmark_identity(spec, 'v1.0', _task_config(limit=5)).fingerprint
    changed_prompt = spec.model_copy(update={'prompt_template': 'Changed prompt'})
    assert first.fingerprint != build_benchmark_identity(changed_prompt, 'v1.0', config).fingerprint
    changed_revision = spec.model_copy(update={'dataset_revision': '2026-08-21'})
    assert first.fingerprint != build_benchmark_identity(changed_revision, 'v1.0', config).fingerprint


def test_cache_identity_requires_match_unless_rerun_review_is_explicit() -> None:
    current = BenchmarkEvaluationIdentity(evaluation_version='v1.1', fingerprint='sha256:current')
    previous = BenchmarkEvaluationIdentity(evaluation_version='v1.0', fingerprint='sha256:previous')

    assert cache_source_for_identity(current, current, False, False) is None
    exact_review_source = cache_source_for_identity(current, current, False, True)
    assert exact_review_source is not None
    assert exact_review_source.fingerprint == 'sha256:current'
    with pytest.raises(ValueError, match='rerun_review=True'):
        cache_source_for_identity(current, previous, False, False)

    source = cache_source_for_identity(current, previous, True, True)
    assert source is not None
    assert source.evaluation_version == 'v1.0'
    assert source.fingerprint == 'sha256:previous'
    assert source.inferred_legacy is True
    assert source.reuse_mode == 'rerun_review_override'


def test_unknown_cache_requires_the_same_explicit_review_override() -> None:
    current = BenchmarkEvaluationIdentity(evaluation_version='v1.0', fingerprint='sha256:current')

    with pytest.raises(ValueError, match='previous=unknown'):
        validate_cached_evaluation_identity(None, _identity({'demo': current}), rerun_review=False)

    sources = validate_cached_evaluation_identity(None, _identity({'demo': current}), rerun_review=True)
    assert sources['demo'].evaluation_version == 'unknown'
    assert sources['demo'].fingerprint == 'unknown'


def test_cache_mismatch_is_rejected_before_snapshot_is_overwritten(tmp_path) -> None:
    meta = _meta()
    current_config = _task_config(use_cache=str(tmp_path))
    spec = ResolvedBenchmarkSpec.from_meta(meta, current_config)
    current = build_evaluation_identity({'demo': spec}, {'demo': 'v1.1'}, current_config)
    previous = build_evaluation_identity({'demo': spec}, {'demo': 'v1.0'}, current_config)
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    config_path = config_dir / 'task_config.yaml'
    dict_to_yaml({'evaluation_identity': previous.model_dump(mode='json')}, str(config_path))
    original_snapshot = config_path.read_text()

    previous_snapshot = load_task_config_snapshot(str(config_path))
    with pytest.raises(ValueError, match='Cached evaluation identity does not match'):
        validate_cached_evaluation_identity(previous_snapshot, current, rerun_review=False)

    assert config_path.read_text() == original_snapshot

    forced_config = _task_config(use_cache=str(tmp_path), rerun_review=True)
    forced = build_evaluation_identity({'demo': spec}, {'demo': 'v1.1'}, forced_config)
    sources = validate_cached_evaluation_identity(previous_snapshot, forced, rerun_review=True)
    assert sources['demo'].fingerprint == previous.benchmarks['demo'].fingerprint


def test_native_cache_identity_blocks_mismatched_run_before_snapshot_overwrite(tmp_path) -> None:
    base = {
        'model': 'mock-model',
        'eval_type': 'mock_llm',
        'datasets': ['general_mcq'],
        'dataset_args': {'general_mcq': {'local_path': 'custom_eval/text/mcq', 'subset_list': ['example']}},
        'limit': 1,
        'no_timestamp': True,
        'work_dir': str(tmp_path),
    }
    run_task(TaskConfig(**base))
    run_task(TaskConfig(**base, use_cache=str(tmp_path)))

    snapshot_path = tmp_path / 'configs' / 'task_config.yaml'
    snapshot = snapshot_path.read_text()
    assert 'evaluation_identity:' in snapshot
    assert 'local_path: custom_eval/text/mcq' in snapshot

    with pytest.raises(ValueError, match='Cached evaluation identity does not match'):
        run_task(TaskConfig(**base, use_cache=str(tmp_path), generation_config={'temperature': 0.7}))

    assert snapshot_path.read_text() == snapshot


def test_legacy_full_meta_snapshot_infers_v1_identity() -> None:
    config = _task_config()
    spec = ResolvedBenchmarkSpec.from_meta(_meta(), config)
    previous_config = config.to_dict()
    previous_config['dataset_args'] = {'demo': spec.model_dump(mode='json')}

    identity = legacy_identity_from_config(previous_config, 'demo')

    assert identity is not None
    assert identity.evaluation_version == 'v1.0'
    assert identity.fingerprint == build_benchmark_identity(spec, 'v1.0', config).fingerprint
    assert validate_cached_evaluation_identity(previous_config, _identity({'demo': identity}), rerun_review=False) == {}


def test_analysis_uses_compact_context_without_full_meta_or_perf(monkeypatch) -> None:
    config = _task_config()
    meta = _meta()
    spec = ResolvedBenchmarkSpec.from_meta(meta, config)
    identity = build_benchmark_identity(spec, meta.evaluation_version, config)
    report = Report.from_dict({
        'dataset_name': 'demo',
        'dataset_description': 'full metadata must not be sent',
        'metrics': [{
            'name': 'mean_accuracy',
            'score': 0.8,
            'num': 10,
            'categories': [],
        }],
    })
    report.perf_metrics = {'latency': {'value': 42}}
    context = build_analysis_context(meta, spec, identity, report)
    captured = {}

    class FakeJudge:

        model_id = 'fake-judge'

        def __init__(self, **kwargs) -> None:
            pass

        def generate(self, messages):
            captured['prompt'] = messages[0].content
            return SimpleNamespace(completion='analysis')

    import evalscope.metrics
    monkeypatch.setattr(evalscope.metrics, 'LLMJudge', FakeJudge)

    report.generate_analysis(config, context)

    prompt = captured['prompt']
    assert 'A concise benchmark overview.' in prompt
    assert 'Task Type: Multiple choice' in prompt
    assert '"resolved_benchmark"' in prompt
    assert '"score": 0.8' in prompt
    assert 'This must not be sent to analysis.' not in prompt
    assert 'full metadata must not be sent' not in prompt
    assert 'latency' not in prompt
    assert report.schema_version == 2


def _identity(benchmarks: dict[str, BenchmarkEvaluationIdentity]) -> EvaluationIdentity:
    return EvaluationIdentity(
        framework=FrameworkProvenance(evalscope_version='test'),
        benchmarks=benchmarks,
    )
