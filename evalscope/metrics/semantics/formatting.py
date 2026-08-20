"""Metric value formatting driven purely by ``MetricSemantics``.

This module formats the display text of a metric value for every backend surface: the CLI table,
the HTML report and the reports API. The frontend primitive
``evalscope/web/src/domain/metric/metricFormat.ts`` implements the same rules, and both sides
assert against the shared golden samples in ``tests/report/semantics/golden_samples.json`` so one
value renders identically wherever it appears.

Formatting rules
----------------
Only the display fields of ``MetricSemantics`` are consulted (``display_kind``,
``display_multiplier``, ``display_unit``, ``display_precision``). There is no name inference.

- Missing value (``None`` or a non-finite float) -> ``MISSING_PLACEHOLDER`` (``'—'``).
- Missing semantics (``None``) -> diagnostic fallback: the raw value rounded to
  ``DIAGNOSTIC_FALLBACK_PRECISION`` decimals, no unit.
- ``display_kind='percent'`` -> ``value * (display_multiplier or 1)`` rounded to nearest to
  ``display_precision`` decimals, immediately followed by ``display_unit`` (no space before
  ``%``). A ratio in [0, 1] uses ``display_multiplier=100``, an official 0-100 scale uses ``1``.
- ``display_kind='number'`` -> ``value * (display_multiplier or 1)`` rounded to nearest to
  ``display_precision`` decimals, then a single space and ``display_unit``. The multiplier allows
  declared unit conversions such as seconds to milliseconds.

Decimal ties go toward positive infinity, matching JavaScript ``Math.round`` rather than Python's
banker's rounding: ``12.5`` renders as ``13`` and ``-0.5`` renders as ``0`` at precision 0.
Rounded values are rendered in their shortest exact decimal form, matching JavaScript number
stringification: trailing zeros are dropped (``90.0`` -> ``'90'``) and a value that rounds to
zero renders as ``'0'`` (never ``'-0'`` nor ``'0.000'``).

Golden sample schema
--------------------
``tests/report/semantics/golden_samples.json`` is a JSON array of objects, read by
``tests/report/semantics/test_golden_samples.py`` on this side and by
``evalscope/web/src/domain/metric/metricFormat.test.ts`` on the frontend side. It lives under
``tests/`` rather than next to this module so the shipped package carries no test fixture.

.. code-block:: text

    id               str            stable identifier of the sample, unique in the file
    description      str            optional human note; consumers may ignore it
    semantics        object | null  ``MetricSemantics.model_dump(mode='json')`` (snake_case
                                   field names) or ``null`` for the diagnostic fallback path
    value            number | null  the stored metric value, ``null`` for a missing value
    expected_primary str            expected output of ``format_metric_value(value, semantics)``
    expected_raw     str            expected ``FormattedMetric.raw`` of the frontend primitive.
                                   The unscaled text is a frontend-only concern (it backs the
                                   value tooltips), so no backend function produces it.
    identity         object         optional ``MetricIdentity.model_dump(mode='json')``. Present
                                   only on samples that pin the label path.
    expected_label   str            expected output of ``format_metric_label(identity, semantics,
                                   legacy_name)`` and of the frontend
                                   ``formatMetricIdentityLabel``. Required with ``identity``.
    legacy_name      str            optional original v1 spelling a diagnostic label falls back to

Consumers must ignore unknown keys: ``id`` and ``description`` are metadata. Samples deliberately
stay inside the plain-decimal range (no exponential notation) so the expected strings are
unambiguous on both sides.
"""

import math
import re
from collections import Counter
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple, Union

from evalscope.api.metric.semantics import (
    DIAGNOSTIC_FALLBACK_PRECISION,
    MetricDirection,
    MetricDisplayKind,
    MetricIdentity,
    MetricKind,
    MetricSemantics,
)

if TYPE_CHECKING:
    from evalscope.metrics.semantics.resolver import SemanticsResolver

#: Rendered in place of a value that is absent or not finite.
MISSING_PLACEHOLDER = '—'

_DIRECTION_ARROWS = {
    MetricDirection.HIGHER_IS_BETTER: '↑',
    MetricDirection.LOWER_IS_BETTER: '↓',
}

_DIMENSION_DISPLAY_ORDER = {'target': 0, 'level': 1, 'scope': 2, 'ngram': 3, 'variant': 3, 'statistic': 4}

#: First word character of each word, matching the frontend's ``/\b\w/`` label rule.
_WORD_START = re.compile(r'\b\w')

__all__ = [
    'MISSING_PLACEHOLDER',
    'DIAGNOSTIC_FALLBACK_PRECISION',
    'format_metric_label',
    'format_metric_labels',
    'format_metric_value',
    'format_perf_value',
    'is_missing_value',
]


def is_missing_value(value: Any) -> bool:
    """Tell whether ``value`` must render as ``MISSING_PLACEHOLDER``.

    Args:
        value: Candidate metric value.

    Returns:
        ``True`` for ``None``, booleans, non-numeric objects and non-finite floats
        (``nan`` / ``inf``), ``False`` for a finite number.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return not math.isfinite(float(value))


def _round_to_decimal(value: float, precision: int) -> Decimal:
    """Round to nearest with decimal ties going toward positive infinity."""
    quantum = Decimal(1).scaleb(-max(precision, 0))
    # Convert through the shortest decimal representation used by JSON / JavaScript rather than
    # preserving the binary approximation (for example, 1.005 must remain a decimal tie).
    shifted = Decimal(str(value)) / quantum
    return (shifted + Decimal('0.5')).to_integral_value(rounding=ROUND_FLOOR) * quantum


def _format_number(value: float, precision: int) -> str:
    """Render ``value`` rounded to ``precision`` decimals in its shortest exact decimal form."""
    rounded = _round_to_decimal(value, precision)
    if rounded == 0:
        # Collapse '0.000' and '-0' the way JavaScript stringifies 0 and -0.
        return '0'
    return format(rounded.normalize(), 'f')


def _join_unit(text: str, unit: Optional[str], separator: str) -> str:
    """Append ``unit`` to ``text`` using ``separator``, or return ``text`` when there is no unit."""
    if not unit:
        return text
    return f'{text}{separator}{unit}'


def _format_dimension(value: Union[str, int, float, bool]) -> str:
    r"""Render one dimension value for a metric label.

    Only the first letter of each word is raised. ``str.title()`` would also lower the rest, which
    turns an acronym such as ``table_TEDS`` into ``Table Teds`` and makes this diverge from the
    frontend ``formatMetricIdentityLabel``, whose ``/\b\w/`` replacement leaves the tail alone.
    """
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    return _WORD_START.sub(lambda match: match.group(0).upper(), str(value).replace('_', ' '))


def format_metric_label(
    identity: Union[MetricIdentity, str],
    semantics: Optional[MetricSemantics],
    legacy_name: Optional[str] = None,
) -> str:
    """Render one identity through its display name, direction, and meaningful dimensions.

    Diagnostic and unresolved metrics keep their final report name unless their declared
    semantics provide an explicit diagnostic display name.

    Args:
        identity: V2 identity, or a v1 string during migration.
        semantics: Resolved semantics, or ``None`` for an older/unresolved report.
        legacy_name: Original v1 spelling for a diagnostic migration fallback.

    Returns:
        A label such as ``'Accuracy ↑'``, ``'WER ↓'`` or the unchanged final metric name.
    """
    if isinstance(identity, str):
        raw_name = identity
        dimensions = {}
    else:
        raw_name = identity.key
        dimensions = identity.dimensions
    if semantics is None:
        return legacy_name or raw_name
    if semantics.kind is MetricKind.DIAGNOSTIC:
        return semantics.display_name or legacy_name or raw_name
    arrow = _DIRECTION_ARROWS.get(semantics.direction, '')
    label = f'{semantics.metric_name} {arrow}'.strip()
    if dimensions:
        ordered = sorted(dimensions.items(), key=lambda item: (_DIMENSION_DISPLAY_ORDER.get(item[0], 3), item[0]))
        details = ' · '.join(_format_dimension(value) for _, value in ordered)
        label = f'{label} · {details}'
    return label


def format_metric_labels(
    metrics: Iterable[Tuple[Union[MetricIdentity, str], Optional[MetricSemantics]]]
) -> Dict[str, str]:
    """Render all metric labels of one report and disambiguate repeated display names.

    Args:
        metrics: ``(final_metric_name, semantics)`` pairs from one report.

    Returns:
        Final metric name -> display label. Repeated scored labels carry the final name in
        parentheses, for example ``'Accuracy ↑ (mean_fact_acc)'``.
    """
    pairs = list(metrics)
    keys = [identity if isinstance(identity, str) else identity.key for identity, _ in pairs]
    labels = {key: format_metric_label(identity, semantics) for key, (identity, semantics) in zip(keys, pairs)}
    counts = Counter(labels.values())
    return {key: f'{labels[key]} ({key})' if counts[labels[key]] > 1 else labels[key] for key in keys}


def format_metric_value(value: Optional[float], semantics: Optional[MetricSemantics]) -> str:
    """Format the display text of one metric value.

    Args:
        value: Stored metric value. ``None`` or a non-finite float renders as
            ``MISSING_PLACEHOLDER``.
        semantics: Semantics of the metric. ``None`` triggers the diagnostic fallback.

    Returns:
        The display text, for example ``'85.7%'``, ``'1.235 s'`` or ``'—'``.
    """
    if is_missing_value(value):
        return MISSING_PLACEHOLDER

    number = float(value)
    if semantics is None:
        return _format_number(number, DIAGNOSTIC_FALLBACK_PRECISION)

    scaled = number * (semantics.display_multiplier or 1.0)
    if semantics.display_kind == MetricDisplayKind.PERCENT:
        return _join_unit(_format_number(scaled, semantics.display_precision), semantics.display_unit, '')

    return _join_unit(_format_number(scaled, semantics.display_precision), semantics.display_unit, ' ')


def format_perf_value(
    value: Optional[float],
    field_key: str,
    include_unit: bool = True,
    resolver: Optional['SemanticsResolver'] = None,
) -> str:
    """Resolve and format a perf field through the shared semantics registry.

    ``include_unit=False`` is intended for cells whose table header already contains the unit;
    scaling and precision still come from the same semantics declaration.
    """
    if resolver is None:
        # Function-local import preserves the one-way package import graph: resolver imports the
        # formatting-independent perf catalog, while renderers can still use this convenience API.
        from evalscope.metrics.semantics.resolver import get_semantics_resolver
        resolver = get_semantics_resolver()
    semantics = resolver.resolve_perf_field(field_key).semantics
    if not include_unit and semantics.display_unit:
        semantics = semantics.model_copy(update={'display_unit': None})
    return format_metric_value(value, semantics)
