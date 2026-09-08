# Copyright (c) Alibaba, Inc. and its affiliates.
"""Focused regression tests for evalscope.utils.import_utils.check_import."""
import pytest

from evalscope.utils.import_utils import check_import


class TestCheckImport:
    def test_missing_module_returns_false(self) -> None:
        assert check_import('definitely_missing_mod_xyz', raise_warning=False) is False

    def test_missing_module_raise_error(self) -> None:
        with pytest.raises(ImportError, match='not found'):
            check_import('definitely_missing_mod_xyz', raise_warning=False, raise_error=True)

    def test_broken_module_propagates_runtime_error(self, tmp_path, monkeypatch) -> None:
        (tmp_path / 'brokemod_rg.py').write_text('raise RuntimeError(\'boom-on-import\')\n')
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(RuntimeError, match='boom-on-import'):
            check_import('brokemod_rg', raise_warning=False)

    def test_broken_module_not_reported_as_missing(self, tmp_path, monkeypatch, caplog) -> None:
        (tmp_path / 'brokemod_rg2.py').write_text('raise RuntimeError(\'boom-on-import\')\n')
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(RuntimeError):
            check_import('brokemod_rg2', raise_warning=True)
        assert 'not found' not in caplog.text

    def test_existing_module_returns_true(self) -> None:
        assert check_import('os', raise_warning=False) is True
