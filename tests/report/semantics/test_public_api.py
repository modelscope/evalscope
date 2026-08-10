"""Unit tests for the public API surface of ``evalscope.metrics.semantics``.

Feature: metric-semantics-governance
"""
import ast
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List

import evalscope.metrics.semantics as semantics

#: The exact surface, verified against what production imports:
#:
#: - ``compose_final_metric_name`` / ``get_semantics_resolver`` -- ``evalscope/report/generator.py``
#: - ``hydrate_report_semantics`` -- ``evalscope/report/report.py``
#: - ``format_metric_value`` -- ``evalscope/report/renderer.py``
#: - ``PrimaryMetricRef`` -- ``evalscope/service/blueprints/reports.py``
#:
#: The catalog tables, the baseline table and the resolver internals are deliberately absent: no
#: production module imports them, so exporting them would widen the maintained API for nothing.
#: Tests import those from their owning module instead.
EXPECTED_EXPORTS: List[str] = [
    'PrimaryMetricRef',
    'compose_final_metric_name',
    'format_metric_value',
    'get_semantics_resolver',
    'hydrate_report_semantics',
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

    This is a policy gate, not a behaviour test: the branch that introduced this package exported
    four catalog tables no production module imported. Pinning the list makes widening the surface
    a deliberate edit.
    """

    def test_surface_is_exactly_the_expected_exports(self) -> None:
        assert sorted(semantics.__all__) == EXPECTED_EXPORTS


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
