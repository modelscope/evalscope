"""Unit tests for the public API surface of ``evalscope.metrics.semantics``.

Feature: metric-semantics-governance
"""
import ast
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List

import evalscope.metrics.semantics as semantics
from evalscope.metrics.semantics.catalog import lookup_metric_entry

#: Names the rest of the codebase (report generation, service APIs) relies on.
REQUIRED_EXPORTS: List[str] = [
    'METRIC_NAME_SEMANTICS',
    'BENCHMARK_METRIC_OVERRIDES',
    'BENCHMARK_DYNAMIC_METRICS',
    'SEMANTIC_BASELINES',
    'compose_final_metric_name',
    'get_semantics_resolver',
    'hydrate_report_semantics',
    'UndeclaredMetricError',
    'PrimaryMetricRef',
    'format_metric_value',
]

#: Names deliberately kept out of the package surface: they have no production consumer, so
#: exporting them would widen the maintained API for nothing. Tests import them from their
#: owning module instead.
UNEXPORTED_HELPERS: List[str] = [
    'format_metric',
    'FormattedMetric',
    'format_raw_metric_value',
    'get_unit_label',
    'round_half_up',
    'MISSING_PLACEHOLDER',
    'lookup_metric_entry',
    'diagnostic_fallback',
    'is_public_perf_field',
    'builtin_benchmark_names',
    'catalog_entry_location',
    'ResolvedSemantics',
    'SemanticsResolver',
    'SemanticsSource',
]


def _module_level_imports(module: ast.Module) -> Iterator[ast.stmt]:
    """Yield the import statements executed at import time, skipping TYPE_CHECKING blocks."""
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If) and not _is_type_checking_guard(node.test):
            for inner in node.body + node.orelse:
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    yield inner


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Whether an ``if`` test is the ``TYPE_CHECKING`` guard, whose body never runs."""
    if isinstance(test, ast.Name):
        return test.id == 'TYPE_CHECKING'
    return isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'


def _imports_report_at_module_level(path: Path) -> bool:
    """Whether a source file imports ``evalscope.report`` while being imported."""
    module = ast.parse(path.read_text(encoding='utf-8'))
    for node in _module_level_imports(module):
        if isinstance(node, ast.ImportFrom):
            if (node.module or '').startswith('evalscope.report'):
                return True
            continue
        if any(alias.name.startswith('evalscope.report') for alias in node.names):
            return True
    return False


class TestPublicSurface:
    """The exported surface must stay exactly what production imports.

    The two guards that carry the design intent are
    :meth:`test_required_exports_are_present` (nothing production needs may disappear) and
    :meth:`test_helpers_without_a_production_consumer_stay_unexported` (nothing without a consumer
    may creep in). The import checks below catch an ``__all__`` that lists a missing name, or a
    module that exports more than it declares.
    """

    def test_every_exported_name_is_importable_from_the_package(self) -> None:
        for name in semantics.__all__:
            assert hasattr(semantics, name), f'{name} is listed in __all__ but not importable'

    def test_required_exports_are_present(self) -> None:
        missing = [name for name in REQUIRED_EXPORTS if name not in semantics.__all__]
        assert missing == []

    def test_helpers_without_a_production_consumer_stay_unexported(self) -> None:
        leaked = [name for name in UNEXPORTED_HELPERS if name in semantics.__all__]
        assert leaked == []

    def test_star_import_exposes_exactly_all(self) -> None:
        namespace: dict = {}
        exec('from evalscope.metrics.semantics import *', namespace)  # noqa: S102
        exported = {name for name in namespace if not name.startswith('__')}
        assert exported == set(semantics.__all__)

    def test_lookup_helper_returns_none_for_unknown_metric_name(self) -> None:
        assert lookup_metric_entry('a-metric-that-does-not-exist') is None


class TestNoImportCycle:
    """Both import orders must succeed, in a fresh interpreter each time.

    ``evalscope.report.report`` reaches the semantics package through function-local lazy
    imports, so neither module may need the other at import time (requirement 4.1).
    """

    def _import_in_subprocess(self, statement: str) -> None:
        result = subprocess.run([sys.executable, '-c', statement], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_report_first_then_semantics(self) -> None:
        self._import_in_subprocess('import evalscope.report.report; import evalscope.metrics.semantics')

    def test_semantics_first_then_report(self) -> None:
        self._import_in_subprocess('import evalscope.metrics.semantics; import evalscope.report.report')

    def test_top_level_package_still_imports(self) -> None:
        self._import_in_subprocess('import evalscope')

    def test_semantics_modules_never_import_report_at_module_level(self) -> None:
        # ``evalscope/__init__.py`` imports the report package eagerly, so module presence in
        # ``sys.modules`` proves nothing. Check the sources instead: a top-level import of
        # ``evalscope.report`` anywhere in this package would create the cycle.
        package_dir = Path(semantics.__file__).parent
        offenders = [
            str(path.relative_to(package_dir))
            for path in sorted(package_dir.rglob('*.py'))
            if _imports_report_at_module_level(path)
        ]
        assert offenders == []
