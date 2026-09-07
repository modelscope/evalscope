"""Report-level primary metric: how one is chosen, and where the declaration is read from.

Selection is one policy applied by three producers -- fresh reports, collection reports and
migrated v1 reports -- so it lives here rather than being restated by each of them. It is
deliberately separate from ``resolver``: resolving what a metric *means* says nothing about which
metric carries a report's conclusion.

``select_primary`` never raises. Ambiguity that a benchmark author must fix is reported through
``PrimarySelection.authoring_error`` and the caller decides whether that is fatal: at generation
time it is, while reading an archived report must not fail over it.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from evalscope.api.metric.semantics import MetricIdentity, MetricKind, MetricSelector, MetricSemantics
from evalscope.metrics.semantics.legacy_identity import migrate_legacy_identity

#: Meta cache of the built-in benchmarks, read instead of importing an adapter.
BUILTIN_META_DIR = Path(__file__).parents[2] / 'benchmarks' / '_meta'

__all__ = ['BUILTIN_META_DIR', 'PrimarySelection', 'read_meta_primary_selector', 'select_primary']


class PrimarySelection(BaseModel):
    """Which identity carries a report's conclusion, or why none does."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    identity: Optional[MetricIdentity] = None
    """The selected primary identity, absent when the report has no single conclusion."""

    unavailable_reason: Optional[str] = None
    """Human readable reason, present exactly when ``identity`` is absent."""

    authoring_error: bool = False
    """Whether the benchmark's own declaration is what made the choice ambiguous."""


def select_primary(
    identities: Sequence[MetricIdentity],
    semantics_by_identity: Mapping[str, MetricSemantics],
    selector: Optional[MetricSelector],
) -> PrimarySelection:
    """Select exactly one report-level primary identity.

    An explicit selector may match no emitted identity when the current sample selection cannot
    compute that metric, which is a fact about the run rather than an authoring mistake. Without a
    selector, implicit selection is allowed only when exactly one non-diagnostic identity exists.

    Args:
        identities: Distinct identities the report will contain.
        semantics_by_identity: Resolved semantics keyed by ``MetricIdentity.key``.
        selector: Declared primary selector, or ``None`` to select implicitly.

    Returns:
        The selection. ``identity`` is ``None`` whenever no single conclusion can be named.
    """
    if selector is not None:
        matches = [identity for identity in identities if selector.matches(identity)]
        if not matches:
            return PrimarySelection(
                unavailable_reason=(
                    f'Primary metric selector {selector.model_dump()} did not match any metric emitted for this '
                    'run. The required observations may be absent from the selected samples.'
                )
            )
        if len(matches) != 1:
            return PrimarySelection(
                unavailable_reason=(
                    f'Primary metric selector {selector.model_dump()} matched {len(matches)} identities; '
                    'expected exactly one.'
                ),
                authoring_error=True,
            )
        if semantics_by_identity[matches[0].key].kind is MetricKind.DIAGNOSTIC:
            return PrimarySelection(
                unavailable_reason=f'Primary metric selector matched diagnostic identity {matches[0].key}.',
                authoring_error=True,
            )
        return PrimarySelection(identity=matches[0])

    graded = [
        identity for identity in identities if semantics_by_identity[identity.key].kind is not MetricKind.DIAGNOSTIC
    ]
    if not graded:
        return PrimarySelection(unavailable_reason='No scored metric was emitted for this run.')
    if len(graded) != 1:
        return PrimarySelection(
            unavailable_reason=(
                f'Benchmark emitted {len(graded)} non-diagnostic metric identities; declare '
                'BenchmarkMeta.primary_metric.'
            ),
            authoring_error=True,
        )
    return PrimarySelection(identity=graded[0])


@lru_cache(maxsize=256)
def read_meta_primary_selector(
    benchmark_name: str,
    meta_dir: Path = BUILTIN_META_DIR,
) -> Optional[MetricSelector]:
    """Read the primary selector a built-in benchmark declares, without importing its adapter.

    Args:
        benchmark_name: Benchmark (dataset) name to look up.
        meta_dir: Directory holding the per-benchmark meta cache.

    Returns:
        The declared selector, or ``None`` when the benchmark is unknown, declares none, or its
        meta cache cannot be read.
    """
    path = meta_dir / f'{benchmark_name}.json'
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
