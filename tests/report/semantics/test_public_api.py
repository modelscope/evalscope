"""Unit tests for the public API surface of ``evalscope.metrics.semantics``.

Feature: metric-semantics-governance
"""
import ast
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List

import evalscope.metrics.semantics as semantics

#: Names the rest of the codebase (report generation, service APIs, audits) relies on.
REQUIRED_EXPORTS: List[str] = [
    'METRIC_NAME_SEMANTICS',
    'BENCHMARK_METRIC_OVERRIDES',
    'BENCHMARK_DYNAMIC_METRICS',
    'SEMANTIC_BASELINES',
    'lookup_metric_entry',
    'compose_final_metric_name',
    'ResolvedSemantics',
    'SemanticsResolver',
    'SemanticsSource',
    'diagnostic_fallback',
    'get_semantics_resolver',
    'hydrate_report_semantics',
    'MetricSummary',
    'PrimaryMetricRef',
    'SummaryStatus',
    'summarize_primary_metrics',
    'format_metric_value',
]

#: Names deliberately kept out of the package surface: they have no production consumer, so
#: exporting them would widen the maintained API for nothing. Tests import them from
#: ``formatting.py`` directly instead.
UNEXPORTED_HELPERS: List[str] = [
    'format_metric',
    'FormattedMetric',
    'get_unit_label',
    'round_half_up',
    'MISSING_PLACEHOLDER',
]

#: Maintenance scripts that are not imported by the package ``__init__``, so they may import the
#: report layer without creating a cycle on the runtime read path.
STANDALONE_ENTRY_POINTS = frozenset({'audit.py'})


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

    def test_all_is_declared_and_sorted_into_unique_names(self) -> None:
        assert isinstance(semantics.__all__, list)
        assert len(semantics.__all__) == len(set(semantics.__all__))

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
        assert semantics.lookup_metric_entry('a-metric-that-does-not-exist') is None

    def test_declared_metric_names_resolve_through_the_lookup_helper(self) -> None:
        for metric_name, entry in semantics.METRIC_NAME_SEMANTICS.items():
            assert semantics.lookup_metric_entry(metric_name) is entry

    def test_audit_entry_point_is_not_part_of_the_runtime_surface(self) -> None:
        # ``audit.py`` is a maintenance script: importing the package must not pull it in. Checked
        # in a subprocess because the audit tests of this session import the module directly.
        assert 'audit' not in semantics.__all__
        statement = (
            'import sys, evalscope.metrics.semantics; '
            "sys.exit(1 if 'evalscope.metrics.semantics.audit' in sys.modules else 0)"
        )
        result = subprocess.run([sys.executable, '-c', statement], capture_output=True, text=True)
        assert result.returncode == 0, f'importing the semantics package must not import audit.py: {result.stderr}'

    def test_package_is_reimportable(self) -> None:
        assert importlib.reload(semantics).__all__ == semantics.__all__


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
            if path.name not in STANDALONE_ENTRY_POINTS and _imports_report_at_module_level(path)
        ]
        assert offenders == []
