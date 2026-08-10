import argparse
import pytest

from evalscope.cli.start_service import existing_directory


def test_existing_directory_returns_absolute_path(tmp_path):
    assert existing_directory(str(tmp_path)) == str(tmp_path)


def test_existing_directory_rejects_missing_path(tmp_path):
    missing_path = tmp_path / 'missing'

    with pytest.raises(argparse.ArgumentTypeError, match='output directory does not exist'):
        existing_directory(str(missing_path))
