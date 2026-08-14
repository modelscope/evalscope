"""Unit tests for the public API surface of ``evalscope.metrics.semantics``."""
from typing import List

import evalscope.metrics.semantics as semantics

#: The exact surface, verified against what production imports:
#:
#: - ``get_semantics_resolver`` -- ``evalscope/report/generator.py``
#: - ``hydrate_report_semantics`` -- ``evalscope/report/report.py``
#: - ``attach_perf_semantics`` -- ``evalscope/evaluator/evaluator.py``, ``evalscope/report/report.py``
#: - ``resolve_perf_semantics`` -- ``evalscope/service/blueprints/perf.py``, ``perf_archive.py``
#: - format helpers -- report and perf renderers
#: - ``PrimaryMetricRef`` -- ``evalscope/service/blueprints/reports.py``
#:
#: The catalog tables, the baseline table and the resolver internals are deliberately absent: no
#: production module imports them, so exporting them would widen the maintained API for nothing.
#: Tests import those from their owning module instead.
EXPECTED_EXPORTS: List[str] = [
    'PrimaryMetricRef',
    'attach_perf_semantics',
    'format_metric_label',
    'format_metric_labels',
    'format_metric_value',
    'format_perf_value',
    'get_semantics_resolver',
    'hydrate_report_semantics',
    'resolve_perf_semantics',
]


class TestPublicSurface:
    """The exported surface must stay exactly what production imports.

    This is a policy gate, not a behaviour test: the branch that introduced this package exported
    four catalog tables no production module imported. Pinning the list makes widening the surface
    a deliberate edit.
    """

    def test_surface_is_exactly_the_expected_exports(self) -> None:
        assert sorted(semantics.__all__) == EXPECTED_EXPORTS
