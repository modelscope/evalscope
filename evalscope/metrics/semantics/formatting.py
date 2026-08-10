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
- ``display_kind='percent'`` -> ``value * (display_multiplier or 1)`` rounded half up to
  ``display_precision`` decimals, immediately followed by ``display_unit`` (no space before
  ``%``). A ratio in [0, 1] uses ``display_multiplier=100``, an official 0-100 scale uses ``1``.
- ``display_kind='number'`` -> the value rounded half up to ``display_precision`` decimals,
  then a single space and ``display_unit``. The unit and its space are omitted when
  ``display_unit`` is ``None``.

Rounding is half up (``decimal.ROUND_HALF_UP``), never Python's banker's rounding: ``12.5``
at precision 0 renders as ``13``, and negative ties round away from zero (``-0.5`` -> ``-1``).
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

Consumers must ignore unknown keys: ``id`` and ``description`` are metadata. Samples deliberately
stay inside the plain-decimal range (no exponential notation) so the expected strings are
unambiguous on both sides.
"""

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional

from evalscope.api.metric.semantics import MetricDisplayKind, MetricSemantics

#: Rendered in place of a value that is absent or not finite.
MISSING_PLACEHOLDER = '—'

#: Decimals used when no semantics are available (diagnostic fallback). Shared with
#: ``resolver.diagnostic_fallback``, which builds the semantics this precision belongs to.
DIAGNOSTIC_FALLBACK_PRECISION = 4

__all__ = [
    'MISSING_PLACEHOLDER',
    'DIAGNOSTIC_FALLBACK_PRECISION',
    'format_metric_value',
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
    """Round ``value`` half up into a ``Decimal`` quantized to ``precision`` decimals."""
    quantum = Decimal(1).scaleb(-max(precision, 0))
    # Decimal(float) keeps the exact binary value, so the tie decision matches the frontend,
    # which rounds the same double without an intermediate decimal-string conversion.
    return Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP)


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

    if semantics.display_kind == MetricDisplayKind.PERCENT:
        scaled = number * (semantics.display_multiplier or 1.0)
        return _join_unit(_format_number(scaled, semantics.display_precision), semantics.display_unit, '')

    return _join_unit(_format_number(number, semantics.display_precision), semantics.display_unit, ' ')
