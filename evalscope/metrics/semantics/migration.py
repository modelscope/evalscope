"""Read-old migration for metric identities, report payloads, and persisted semantics."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from evalscope.api.metric.semantics import MetricIdentity, MetricKind, MetricSelector, MetricSemantics
from evalscope.metrics.semantics.catalog import BENCHMARK_METRIC_OVERRIDES, LEGACY_METRIC_MIGRATIONS
from evalscope.metrics.semantics.identity import is_known_dynamic_legacy_name, migrate_legacy_identity
from evalscope.metrics.semantics.resolver import AUDIT_MESSAGE_PREFIX, get_semantics_resolver, select_primary_identity
from evalscope.utils import get_logger

if TYPE_CHECKING:
    from evalscope.report.report import Report

logger = get_logger()

_BUILTIN_META_DIR = Path(__file__).parents[2] / 'benchmarks' / '_meta'


def migrate_legacy_report_identity(metric_name: str, benchmark_name: Optional[str] = None) -> MetricIdentity:
    """Migrate a known v1 name, isolating unknown spellings as diagnostic identities."""
    if metric_name in LEGACY_METRIC_MIGRATIONS or is_known_dynamic_legacy_name(metric_name, benchmark_name):
        return migrate_legacy_identity(metric_name, 'identity', benchmark_name=benchmark_name)
    if re.fullmatch(r'[a-z][a-z0-9_]*', metric_name) and metric_name not in {'score', 'overall', 'total_score'}:
        return MetricIdentity(name=metric_name, aggregation='identity')
    return MetricIdentity(name='legacy_metric', aggregation='identity', dimensions={'original_name': metric_name})


def migrate_legacy_metric_payload(data: Any, benchmark_name: Optional[str] = None) -> Any:
    """Convert one v1 metric dictionary into the persisted v2 shape."""
    if not isinstance(data, dict) or 'identity' in data:
        return data
    migrated = dict(data)
    old_name = migrated.pop('name', 'legacy_metric')
    identity = migrate_legacy_report_identity(old_name, benchmark_name)
    migrated['identity'] = identity.model_dump()
    migrated['legacy_name'] = old_name
    legacy_entry = LEGACY_METRIC_MIGRATIONS.get(old_name)
    if legacy_entry is not None:
        semantics = legacy_entry.resolve(identity.name)
    else:
        semantics = MetricSemantics.diagnostic(old_name)
    migrated.setdefault('semantics', semantics.model_dump())
    return migrated


def migrate_legacy_report_payload(data: Any) -> Any:
    """Convert a v1 report dictionary into the shape validated by ``Report`` v2."""
    if not isinstance(data, dict):
        return data
    migrated = dict(data)
    migrated.pop('num', None)
    metrics = migrated.get('metrics', [])
    has_v2_metrics = bool(metrics) and all(
        hasattr(metric, 'identity') or isinstance(metric, dict) and 'identity' in metric for metric in metrics
    )
    if migrated.get('schema_version') == 2 or has_v2_metrics:
        migrated.setdefault('schema_version', 2)
        return migrated

    dataset_name = migrated.get('dataset_name')
    migrated.pop('score', None)
    migrated['schema_version'] = 2
    migrated['metrics'] = [migrate_legacy_metric_payload(metric, benchmark_name=dataset_name) for metric in metrics]
    return migrated


@lru_cache(maxsize=None)
def _meta_primary_metric(benchmark_name: str) -> Optional[MetricSelector]:
    """Read the primary selector of a built-in benchmark without importing its adapter."""
    path = _BUILTIN_META_DIR / f'{benchmark_name}.json'
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    meta = data.get('meta') if isinstance(data, dict) else None
    if not isinstance(meta, dict):
        return None
    primary = meta.get('primary_metric')
    if isinstance(primary, dict):
        try:
            return MetricSelector.model_validate(primary)
        except ValueError:
            return None
    if not isinstance(primary, str) or not primary:
        return None
    aggregation = meta.get('aggregation')
    identity = migrate_legacy_identity(
        primary, aggregation if isinstance(aggregation, str) else '', benchmark_name=benchmark_name
    )
    return MetricSelector(name=identity.name)


def hydrate_report_semantics(report: 'Report') -> 'Report':
    """Resolve and persist the semantics of a historical report in place."""
    metrics = list(getattr(report, 'metrics', None) or [])
    if not metrics:
        return report

    benchmark_name = getattr(report, 'dataset_name', '') or ''
    active_resolver = get_semantics_resolver()
    selector = _meta_primary_metric(benchmark_name)

    identities = [metric.identity for metric in metrics]
    semantics_by_identity: Dict[str, MetricSemantics] = {}
    for metric in metrics:
        benchmark_override = BENCHMARK_METRIC_OVERRIDES.get((benchmark_name, metric.identity.name))
        legacy_entry = LEGACY_METRIC_MIGRATIONS.get(metric.legacy_name or '')
        if legacy_entry is not None:
            semantics = legacy_entry.resolve(metric.identity.name)
            degraded = False
        elif benchmark_override is not None:
            semantics = benchmark_override.resolve(metric.identity.name)
            degraded = semantics.kind is MetricKind.DIAGNOSTIC
        else:
            resolved = active_resolver.resolve(benchmark_name, metric.identity)
            resolved.log_audit_messages()
            semantics = resolved.semantics
            degraded = resolved.degraded
        semantics_by_identity[metric.identity.key] = semantics
        if not degraded and semantics.kind is not MetricKind.DIAGNOSTIC:
            metric.legacy_name = None

    try:
        primary_identity = select_primary_identity(identities, semantics_by_identity, selector)
    except ValueError as error:
        logger.warning(f'{AUDIT_MESSAGE_PREFIX} legacy primary selector did not match the migrated identities: {error}')
        try:
            primary_identity = select_primary_identity(identities, semantics_by_identity, None)
        except ValueError as fallback_error:
            logger.warning(f'{AUDIT_MESSAGE_PREFIX} legacy report has no unambiguous primary metric: {fallback_error}')
            primary_identity = None
    for metric in metrics:
        metric.semantics = semantics_by_identity[metric.identity.key]
    report.primary_metric_identity = primary_identity
    return report
