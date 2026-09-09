# Copyright (c) Alibaba, Inc. and its affiliates.
"""Focused regression tests for evalscope.utils.import_utils.check_import."""
import sys
from pathlib import Path
from typing import Callable
from unittest import mock

import pytest

from evalscope.utils.import_utils import check_import


@pytest.fixture
def make_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str, str], str]:
    """Create a top-level temp module on sys.path and clean up sys.modules afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    names: list[str] = []

    def _make(name: str, body: str) -> str:
        (tmp_path / f'{name}.py').write_text(body, encoding='utf-8')
        names.append(name)
        return name

    yield _make
    for name in names:
        sys.modules.pop(name, None)


@pytest.fixture
def make_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], str]:
    """Create an empty temp package on sys.path and clean up sys.modules afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    names: list[str] = []

    def _make(name: str) -> str:
        pkg_dir = tmp_path / name
        pkg_dir.mkdir(exist_ok=True)
        (pkg_dir / '__init__.py').write_text('', encoding='utf-8')
        names.append(name)
        return name

    yield _make
    for mod_name in [m for m in sys.modules if m in names or any(m.startswith(n + '.') for n in names)]:
        sys.modules.pop(mod_name, None)


class TestCheckImport:
    def test_missing_module_returns_false(self) -> None:
        assert check_import('definitely_missing_mod_xyz', raise_warning=False) is False

    def test_missing_module_raise_error(self) -> None:
        with pytest.raises(ImportError, match='not found'):
            check_import('definitely_missing_mod_xyz', raise_warning=False, raise_error=True)

    def test_existing_module_returns_true(self) -> None:
        with mock.patch('evalscope.utils.import_utils.logger') as mock_logger:
            assert check_import('os', raise_warning=False) is True
            mock_logger.warning.assert_not_called()

    def test_broken_module_propagates_runtime_error(self, make_module: Callable[[str, str], str]) -> None:
        make_module('brokemod_rg', "raise RuntimeError('boom-on-import')\n")
        with mock.patch('evalscope.utils.import_utils.logger') as mock_logger:
            with pytest.raises(RuntimeError, match='boom-on-import'):
                check_import('brokemod_rg', raise_warning=True)
            mock_logger.warning.assert_not_called()

    def test_module_body_import_error_propagates(self, make_module: Callable[[str, str], str]) -> None:
        make_module('mod_raise_import_err', "raise ImportError('boom-on-import')\n")
        with mock.patch('evalscope.utils.import_utils.logger') as mock_logger:
            with pytest.raises(ImportError) as exc_info:
                check_import('mod_raise_import_err', raise_warning=True)
            assert 'boom-on-import' in str(exc_info.value)
            assert 'not found' not in str(exc_info.value)
            mock_logger.warning.assert_not_called()

    def test_module_internal_missing_dependency_propagates(
        self, make_module: Callable[[str, str], str]
    ) -> None:
        make_module('mod_inner_missing_dep', 'import definitely_missing_internal_dep_xyz\n')
        with mock.patch('evalscope.utils.import_utils.logger') as mock_logger:
            with pytest.raises(ModuleNotFoundError) as exc_info:
                check_import('mod_inner_missing_dep', raise_warning=True)
            assert exc_info.value.name == 'definitely_missing_internal_dep_xyz'
            assert 'mod_inner_missing_dep' not in str(exc_info.value)
            mock_logger.warning.assert_not_called()

    def test_dotted_missing_parent_is_missing(self) -> None:
        with mock.patch('evalscope.utils.import_utils.logger') as mock_logger:
            assert check_import('definitely_missing_parent_xyz.child', raise_warning=True) is False
            assert mock_logger.warning.call_count == 1
            assert 'not found' in mock_logger.warning.call_args[0][0]
        with pytest.raises(ImportError, match='not found'):
            check_import('definitely_missing_parent_xyz.child', raise_warning=False, raise_error=True)

    def test_dotted_missing_child_of_existing_package_is_missing(
        self, make_package: Callable[[str], str]
    ) -> None:
        make_package('hpkg_present')
        with mock.patch('evalscope.utils.import_utils.logger') as mock_logger:
            assert check_import('hpkg_present.missing_child', raise_warning=True) is False
            assert mock_logger.warning.call_count == 1
            assert 'not found' in mock_logger.warning.call_args[0][0]

    def test_dotted_existing_module_returns_true(self) -> None:
        with mock.patch('evalscope.utils.import_utils.logger') as mock_logger:
            assert check_import('os.path', raise_warning=False) is True
            mock_logger.warning.assert_not_called()
