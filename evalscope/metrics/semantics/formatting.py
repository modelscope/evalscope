"""Metric value formatting driven purely by ``MetricSemantics``.

This module is the backend half of the formatting contract. The frontend primitive
``evalscope/web/src/domain/metric/metricFormat.ts`` implements the exact same rules, and both
sides assert against the shared golden samples in ``golden_samples.json`` so the CLI, the HTML
report, the reports API and the Web UI render one value identically.

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
- The raw text (tooltips, exports) is the stored value rounded to ``RAW_VALUE_PRECISION``
  decimals followed by ``raw_unit`` after a space, never scaled by ``display_multiplier``.

Rounding is half up (``decimal.ROUND_HALF_UP``), never Python's banker's rounding: ``12.5``
at precision 0 renders as ``13``, and negative ties round away from zero (``-0.5`` -> ``-1``).
Rounded values are rendered in their shortest exact decimal form, matching JavaScript number
stringification: trailing zeros are dropped (``90.0`` -> ``'90'``) and a value that rounds to
zero renders as ``'0'`` (never ``'-0'`` nor ``'0.000'``).

Golden sample schema
--------------------
``golden_samples.json`` is a JSON array of objects. Both the Python test
(``tests/report/semantics/test_golden_samples.py``) and the vitest test
(``evalscope/web/src/domain/metric/goldenSamples.test.ts``) read this one file.

.. code-block:: text

    id               str            stable identifier of the sample, unique in the file
    description      str            optional human note; consumers may ignore it
    semantics        object | null  ``MetricSemantics.model_dump(mode='json')`` (snake_case
                                   field names) or ``null`` for the diagnostic fallback path
    value            number | null  the stored metric value, ``null`` for a missing value
    expected_primary str            expected output of ``format_metric_value(value, semantics)``
    expected_raw     str            expected output of ``format_raw_metric_value(...)``

Consumers must ignore unknown keys: ``id`` and ``description`` are metadata, only
``semantics`` / ``value`` / ``expected_primary`` / ``expected_raw`` are the assertion contract.
Samples deliberately stay inside the plain-decimal range (no exponential notation) so that the
expected strings are unambiguous on both sides.
"""

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, List, Optional, Union

from evalscope.api.metric.semantics import MetricDisplayKind, MetricRole, MetricSemantics

#: Rendered in place of a value that is absent or not finite.
MISSING_PLACEHOLDER = '—'

#: Decimals used when no semantics are available (diagnostic fallback).
DIAGNOSTIC_FALLBACK_PRECISION = 4

#: Decimals used by the raw (unscaled) representation shown in tooltips and exports.
RAW_VALUE_PRECISION = 4

#: Location of the golden samples shared with the frontend formatting tests.
GOLDEN_SAMPLES_PATH = Path(__file__).with_name('golden_samples.json')

__all__ = [
    'MISSING_PLACEHOLDER',
    'DIAGNOSTIC_FALLBACK_PRECISION',
    'RAW_VALUE_PRECISION',
    'GOLDEN_SAMPLES_PATH',
    'FormattedMetric',
    'GoldenSample',
    'format_metric',
    'format_metric_value',
    'format_raw_metric_value',
    'get_unit_label',
    'is_missing_value',
    'load_golden_samples',
    'round_half_up',
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


def round_half_up(value: float, precision: int) -> float:
    """Round ``value`` to ``precision`` decimals, halves away from zero.

    Args:
        value: Finite number to round.
        precision: Non-negative number of decimals to keep.

    Returns:
        The rounded value.
    """
    return float(_round_to_decimal(value, precision))


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
    """Format the primary display text of one metric value.

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


def format_raw_metric_value(value: Optional[float], semantics: Optional[MetricSemantics]) -> str:
    """Format the unscaled raw text of one metric value.

    The raw text never applies ``display_multiplier``: it shows the stored value with its
    ``raw_unit`` so tooltips and exports can expose what was actually recorded.

    Args:
        value: Stored metric value.
        semantics: Semantics of the metric, or ``None`` when unknown.

    Returns:
        The raw text, for example ``'0.8567'``, ``'1.2346 s'`` or ``'—'``.
    """
    if is_missing_value(value):
        return MISSING_PLACEHOLDER

    text = _format_number(float(value), RAW_VALUE_PRECISION)
    return _join_unit(text, semantics.raw_unit if semantics else None, ' ')


def get_unit_label(semantics: Optional[MetricSemantics]) -> str:
    """Return the display unit of ``semantics``, or an empty string when there is none."""
    if semantics is None or not semantics.display_unit:
        return ''
    return semantics.display_unit


class FormattedMetric(BaseModel):
    """Formatting outcome of one metric value, mirroring the frontend ``FormattedMetric``."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    primary: str
    """Display text driven by the display fields."""

    raw: str
    """Unscaled text of the stored value with its ``raw_unit``."""

    unit_label: str
    """Display unit alone, for table headers and axis labels."""

    is_missing: bool
    """Whether the value was absent or not finite."""

    is_diagnostic_fallback: bool
    """Whether the value must render without a color scale (no semantics, or diagnostic role)."""


def format_metric(value: Optional[float], semantics: Optional[MetricSemantics]) -> FormattedMetric:
    """Format one metric value into every text a UI needs.

    Args:
        value: Stored metric value.
        semantics: Semantics of the metric, or ``None`` when the backend did not provide any.

    Returns:
        The primary and raw texts plus the flags a UI uses to disable color scales.
    """
    return FormattedMetric(
        primary=format_metric_value(value, semantics),
        raw=format_raw_metric_value(value, semantics),
        unit_label=get_unit_label(semantics),
        is_missing=is_missing_value(value),
        is_diagnostic_fallback=semantics is None or semantics.role == MetricRole.DIAGNOSTIC,
    )


class GoldenSample(BaseModel):
    """One entry of ``golden_samples.json``, the backend/frontend formatting contract."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    id: str
    """Stable identifier, unique inside the file."""

    description: str = Field(default='')
    """Human readable note. Not part of the assertion contract."""

    semantics: Optional[MetricSemantics] = Field(default=None)
    """Semantics payload, or ``None`` to exercise the diagnostic fallback."""

    value: Optional[float] = Field(default=None)
    """Stored metric value, or ``None`` to exercise the missing value path."""

    expected_primary: str
    """Expected ``format_metric_value`` output."""

    expected_raw: str
    """Expected ``format_raw_metric_value`` output."""


def load_golden_samples(path: Union[str, Path, None] = None) -> List[GoldenSample]:
    """Load and validate the shared formatting golden samples.

    Args:
        path: Optional override of the samples file. Defaults to ``GOLDEN_SAMPLES_PATH``.

    Returns:
        The parsed samples in file order.

    Raises:
        pydantic.ValidationError: If a sample violates the schema or carries invalid semantics.
    """
    samples_path = Path(path) if path is not None else GOLDEN_SAMPLES_PATH
    with open(samples_path, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    return [GoldenSample(**sample) for sample in payload]
