"""Runtime benchmark specifications and cache identities for native evaluations."""

import hashlib
import json
import re
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.config import TaskConfig

EVALUATION_IDENTITY_SCHEMA_VERSION = 1
_VERSION_RE = re.compile(r'^v\d+\.\d+$')
_SECRET_KEYS = {
    'api-key',
    'authorization',
    'password',
    'proxy-authorization',
    'secret',
    'token',
    'x-api-key',
    'x-auth-token',
}


class ResolvedBenchmarkSpec(BaseModel):
    """The resolved benchmark settings that affect an evaluation run."""

    name: str
    dataset_id: str
    dataset_hub: Optional[str] = None
    dataset_revision: Optional[str] = None
    eval_split: Optional[str] = None
    train_split: Optional[str] = None
    subset_list: list[str] = Field(default_factory=list)
    default_subset: str = 'default'
    prompt_template: Optional[str] = None
    few_shot_prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    query_template: Optional[str] = None
    few_shot_num: int = 0
    few_shot_random: bool = False
    metric_list: list[Any] = Field(default_factory=list)
    aggregation: str = 'mean'
    primary_metric: Optional[Any] = None
    filters: Optional[Dict[str, Any]] = None
    shuffle: bool = False
    shuffle_choices: bool = False
    max_image_bytes: Optional[Union[int, str]] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    sandbox_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_meta(cls, meta: 'BenchmarkMeta', task_config: 'TaskConfig') -> 'ResolvedBenchmarkSpec':
        primary_metric = meta.primary_metric
        if primary_metric is not None and hasattr(primary_metric, 'model_dump'):
            primary_metric = primary_metric.model_dump(mode='json')
        return cls(
            name=meta.name,
            dataset_id=meta.dataset_id,
            dataset_hub=task_config.dataset_hub,
            dataset_revision=getattr(meta, 'dataset_revision', None),
            eval_split=meta.eval_split,
            train_split=meta.train_split,
            subset_list=list(meta.subset_list),
            default_subset=meta.default_subset,
            prompt_template=meta.prompt_template,
            few_shot_prompt_template=meta.few_shot_prompt_template,
            system_prompt=meta.system_prompt,
            query_template=meta.query_template,
            few_shot_num=meta.few_shot_num,
            few_shot_random=meta.few_shot_random,
            metric_list=meta.metric_list,
            aggregation=meta.aggregation,
            primary_metric=primary_metric,
            filters=dict(meta.filters) if meta.filters is not None else None,
            shuffle=meta.shuffle,
            shuffle_choices=meta.shuffle_choices,
            max_image_bytes=meta.max_image_bytes,
            extra_params=meta.get_extra_params(),
            sandbox_config=meta.sandbox_config,
        )


class CacheSource(BaseModel):
    """The direct cache identity force-reused by ``rerun_review``."""

    evaluation_version: str
    fingerprint: str
    inferred_legacy: bool = False
    prediction_reused: bool = True
    reuse_mode: str = 'rerun_review_override'


class BenchmarkEvaluationIdentity(BaseModel):
    """Published version and exact runtime fingerprint of one benchmark."""

    evaluation_version: str
    fingerprint: str
    cache_source: Optional[CacheSource] = None


class EvaluationIdentity(BaseModel):
    """All generated benchmark identities persisted in ``task_config.yaml``."""

    schema_version: int = EVALUATION_IDENTITY_SCHEMA_VERSION
    benchmarks: Dict[str, BenchmarkEvaluationIdentity] = Field(default_factory=dict)


def cache_source_for_identity(
    current: BenchmarkEvaluationIdentity,
    previous: Optional[BenchmarkEvaluationIdentity],
    inferred_legacy: bool,
    rerun_review: bool,
) -> Optional[CacheSource]:
    """Validate cache reuse and return its direct source when review is rerun.

    Normal cache reuse requires a complete identity match. ``rerun_review`` is
    the sole explicit override: it keeps old predictions and recomputes review
    results under the current evaluation version.
    """
    if previous is not None and previous.fingerprint == current.fingerprint:
        if not rerun_review:
            return None
        return CacheSource(
            evaluation_version=previous.evaluation_version,
            fingerprint=previous.fingerprint,
            inferred_legacy=inferred_legacy,
        )
    if not rerun_review:
        previous_label = previous.fingerprint if previous is not None else 'unknown'
        raise ValueError(
            'Cached evaluation identity does not match the current run '
            f'(previous={previous_label}, current={current.fingerprint}). '
            'Use rerun_review=True to explicitly reuse predictions and recompute reviews.'
        )
    if previous is None:
        return CacheSource(
            evaluation_version='unknown',
            fingerprint='unknown',
            inferred_legacy=True,
        )
    return CacheSource(
        evaluation_version=previous.evaluation_version,
        fingerprint=previous.fingerprint,
        inferred_legacy=inferred_legacy,
    )


def validate_evaluation_version(value: str) -> str:
    """Validate the public benchmark evaluation version."""
    if not _VERSION_RE.fullmatch(value):
        raise ValueError("evaluation_version must use the format 'v<major>.<minor>'")
    return value


def build_benchmark_identity(
    spec: ResolvedBenchmarkSpec,
    evaluation_version: str,
    task_config: 'TaskConfig',
) -> BenchmarkEvaluationIdentity:
    """Build a stable cache identity from effective evaluation semantics."""
    validate_evaluation_version(evaluation_version)
    payload = {
        'evaluation_version': evaluation_version,
        'benchmark': spec.model_dump(mode='json'),
        'task': _fingerprint_task_config(task_config),
    }
    encoded = _canonical_json(_scrub_secrets(payload)).encode('utf-8')
    return BenchmarkEvaluationIdentity(
        evaluation_version=evaluation_version,
        fingerprint=f'sha256:{hashlib.sha256(encoded).hexdigest()}',
    )


def build_evaluation_identity(
    specs: Dict[str, ResolvedBenchmarkSpec],
    versions: Dict[str, str],
    task_config: 'TaskConfig',
) -> EvaluationIdentity:
    """Build the generated identity block written to a native task config."""
    return EvaluationIdentity(
        benchmarks={
            name: build_benchmark_identity(spec, versions[name], task_config)
            for name, spec in specs.items()
        },
    )


def build_generated_evaluation_metadata(
    specs: Dict[str, ResolvedBenchmarkSpec],
    identity: EvaluationIdentity,
) -> Dict[str, Any]:
    """Build output-only metadata stored alongside the user task configuration."""
    return {
        'resolved_benchmarks': {
            name: spec.model_dump(mode='json')
            for name, spec in specs.items()
        },
        'evaluation_identity': identity.model_dump(mode='json'),
    }


def validate_cached_evaluation_identity(
    previous_config: Optional[Dict[str, Any]],
    current_identity: EvaluationIdentity,
    rerun_review: bool,
) -> Dict[str, CacheSource]:
    """Validate native cache reuse and return review-rerun prediction sources.

    ``previous_config`` is deliberately the raw output snapshot: generated
    identity is audit data, not input that alters the current adapter defaults.
    """
    previous_identity = _identity_from_config(previous_config)
    sources: Dict[str, CacheSource] = {}
    for benchmark_name, current in current_identity.benchmarks.items():
        previous = None
        inferred_legacy = False
        if previous_identity is not None:
            previous = previous_identity.benchmarks.get(benchmark_name)
        elif previous_config is not None:
            previous = legacy_identity_from_config(previous_config, benchmark_name)
            inferred_legacy = previous is not None

        source = cache_source_for_identity(current, previous, inferred_legacy, rerun_review)
        if source is not None:
            sources[benchmark_name] = source
    return sources


def legacy_identity_from_config(task_config: Dict[str, Any],
                                benchmark_name: str) -> Optional[BenchmarkEvaluationIdentity]:
    """Infer a v1.0 identity from an older full-meta task config snapshot."""
    raw_spec = task_config.get('dataset_args', {}).get(benchmark_name)
    if not isinstance(raw_spec, dict) or 'dataset_id' not in raw_spec:
        return None
    try:
        spec = ResolvedBenchmarkSpec.model_validate({
            **raw_spec,
            'name': raw_spec.get('name', benchmark_name),
            'dataset_hub': task_config.get('dataset_hub'),
            'dataset_revision': raw_spec.get('dataset_revision'),
        })
    except Exception:
        return None
    payload = {
        'evaluation_version': 'v1.0',
        'benchmark': spec.model_dump(mode='json'),
        'task': _fingerprint_task_mapping(task_config),
    }
    encoded = _canonical_json(_scrub_secrets(payload)).encode('utf-8')
    return BenchmarkEvaluationIdentity(
        evaluation_version='v1.0',
        fingerprint=f'sha256:{hashlib.sha256(encoded).hexdigest()}',
    )


def _identity_from_config(config: Optional[Dict[str, Any]]) -> Optional[EvaluationIdentity]:
    """Parse a generated identity block, treating malformed data as unknown provenance."""
    if not config or 'evaluation_identity' not in config:
        return None
    try:
        return EvaluationIdentity.model_validate(config['evaluation_identity'])
    except Exception:
        return None


def _fingerprint_task_config(task_config: 'TaskConfig') -> Dict[str, Any]:
    return _fingerprint_task_mapping(task_config.to_dict())


def _fingerprint_task_mapping(task_config: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        'model',
        'model_id',
        'model_args',
        'model_task',
        'chat_template',
        'generation_config',
        'eval_type',
        'api_url',
        'limit',
        'repeats',
        'seed',
        'judge',
        'sandbox',
        'agent_config',
    )
    return {key: task_config.get(key) for key in keys}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)


def _scrub_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _scrub_secrets(item) for key, item in value.items() if not _is_secret_key(key)}
    if isinstance(value, list):
        return [_scrub_secrets(item) for item in value]
    return value


def _is_secret_key(key: Any) -> bool:
    """Return whether a mapping key contains transport credentials."""
    normalized = str(key).lower().replace('_', '-')
    return normalized in _SECRET_KEYS or normalized.endswith('-api-key')
