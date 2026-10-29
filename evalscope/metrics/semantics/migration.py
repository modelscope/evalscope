"""Read-old migration for metric identities, report payloads, and persisted semantics."""

import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from evalscope.api.metric.semantics import MetricIdentity, MetricKind, MetricSelector, MetricSemantics
from evalscope.metrics.semantics.catalog import LEGACY_METRIC_MIGRATIONS
from evalscope.metrics.semantics.legacy_identity import is_known_legacy_spelling, migrate_legacy_identity
from evalscope.metrics.semantics.primary import PrimarySelectionStatus, read_meta_primary_selector, select_primary
from evalscope.metrics.semantics.resolver import AUDIT_MESSAGE_PREFIX, get_semantics_resolver
from evalscope.utils import get_logger

if TYPE_CHECKING:
    from evalscope.report.report import Report

logger = get_logger()


def migrate_legacy_report_identity(metric_name: str, benchmark_name: Optional[str] = None) -> MetricIdentity:
    """Migrate a known v1 name, isolating unknown spellings as diagnostic identities."""
    if metric_name in LEGACY_METRIC_MIGRATIONS or is_known_legacy_spelling(metric_name, benchmark_name):
        return migrate_legacy_identity(metric_name, 'identity', benchmark_name=benchmark_name)
    if re.fullmatch(r'[a-z][a-z0-9_]*', metric_name) and metric_name not in {'score', 'overall', 'total_score'}:
        return MetricIdentity(name=metric_name, aggregation='identity')
    return MetricIdentity(name='legacy_metric', aggregation='identity', dimensions={'original_name': metric_name})


def migrate_legacy_metric_payload(data: Any, benchmark_name: Optional[str] = None) -> Any:
    """Convert one v1 metric dictionary into the persisted v2 shape."""
    if not isinstance(data, dict):
        return data
    migrated = dict(data)
    migrated.pop('semantic_id', None)
    if 'identity' in migrated:
        semantics = migrated.get('semantics')
        if isinstance(semantics, dict):
            normalized_semantics = dict(semantics)
            role = normalized_semantics.pop('role', None)
            normalized_semantics.pop('contract_version', None)
            if 'kind' not in normalized_semantics and role is not None:
                normalized_semantics['kind'] = 'diagnostic' if role == 'diagnostic' else 'quality'
            migrated['semantics'] = normalized_semantics
        return migrated

    old_name = migrated.pop('name', 'legacy_metric')
    identity = migrate_legacy_report_identity(old_name, benchmark_name)
    migrated['identity'] = identity.model_dump()
    migrated['legacy_name'] = old_name
    semantics = get_semantics_resolver().resolve(benchmark_name or '', identity, old_name).semantics
    migrated.setdefault('semantics', semantics.model_dump())
    return migrated


def migrate_legacy_report_payload(data: Any) -> Any:
    """Convert a v1 report dictionary into the shape validated by ``Report`` v2."""
    if not isinstance(data, dict):
        return data
    migrated = dict(data)
    migrated.pop('num', None)
    migrated.pop('score', None)
    migrated.pop('metric_schema_version', None)
    legacy_primary_name = migrated.pop('primary_metric_name', None)
    metrics = migrated.get('metrics', [])
    role_primary_identity = _legacy_primary_identity(metrics, None)
    dataset_name = migrated.get('dataset_name')
    migrated['schema_version'] = 2
    migrated['metrics'] = [migrate_legacy_metric_payload(metric, benchmark_name=dataset_name) for metric in metrics]

    if migrated.get('primary_metric_identity') is None:
        primary_identity = _legacy_primary_identity(migrated['metrics'], legacy_primary_name) or role_primary_identity
        if primary_identity is not None:
            migrated['primary_metric_identity'] = primary_identity
    return migrated


def _legacy_primary_identity(metrics: Any, legacy_primary_name: Any) -> Optional[Dict[str, Any]]:
    """Recover a persisted primary identity from fields removed from the v2 wire format."""
    if not isinstance(metrics, list):
        return None

    if isinstance(legacy_primary_name, str) and legacy_primary_name:
        matches = [
            metric
            for metric in metrics
            if isinstance(metric, dict)
            and (metric.get('legacy_name') == legacy_primary_name or metric.get('name') == legacy_primary_name)
        ]
        if len(matches) == 1 and isinstance(matches[0].get('identity'), dict):
            # A benchmark override may demote the historical primary to a diagnostic.
            if matches[0].get('semantics', {}).get('kind') != MetricKind.DIAGNOSTIC:
                return matches[0]['identity']

    role_matches = [
        metric.get('identity')
        for metric in metrics
        if isinstance(metric, dict)
        and isinstance(metric.get('semantics'), dict)
        and metric['semantics'].get('role') == 'primary'
    ]
    if len(role_matches) == 1 and isinstance(role_matches[0], dict):
        return role_matches[0]
    return None


def hydrate_report_semantics(
    report: 'Report',
    *,
    selector_for: Callable[[str], Optional[MetricSelector]] = read_meta_primary_selector,
) -> 'Report':
    """Resolve and persist the semantics of a historical report in place.

    Args:
        report: Report parsed from a v1 payload, mutated in place.
        selector_for: Reads the primary selector a benchmark declares. Replaceable so this stays
            testable without a meta cache on disk.

    Returns:
        The same report, with every metric's semantics and the primary identity filled in.
    """
    metrics = list(getattr(report, 'metrics', None) or [])
    if not metrics:
        return report

    benchmark_name = getattr(report, 'dataset_name', '') or ''
    active_resolver = get_semantics_resolver()

    identities = [metric.identity for metric in metrics]
    semantics_by_identity: Dict[str, MetricSemantics] = {}
    for metric in metrics:
        resolved = active_resolver.resolve(benchmark_name, metric.identity, metric.legacy_name)
        resolved.log_audit_messages()
        semantics_by_identity[metric.identity.key] = resolved.semantics
        if resolved.semantics.kind is not MetricKind.DIAGNOSTIC:
            metric.legacy_name = None

    selector = selector_for(benchmark_name)
    selection = select_primary(identities, semantics_by_identity, selector)
    if selection.status is PrimarySelectionStatus.NO_MATCH:
        # A stored identity can miss a dimension the current declaration constrains, yet the report
        # may still hold exactly one scored metric that is plainly its conclusion.
        selection = select_primary(identities, semantics_by_identity, None)
    if selection.identity is None:
        logger.warning(f'{AUDIT_MESSAGE_PREFIX} legacy report has no primary metric: {selection.unavailable_reason}')
    for metric in metrics:
        metric.semantics = semantics_by_identity[metric.identity.key]
    report.primary_metric_identity = selection.identity
    report.primary_metric_unavailable_reason = None if selection.identity else selection.unavailable_reason
    return report
