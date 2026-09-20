import os
import zipfile

import pytest

from evalscope.benchmarks.mvbench.utils import find_archive_member as mvbench_find_archive_member
from evalscope.benchmarks.videomme_v2.utils import find_archive_member as videomme_find_archive_member


def _write_zip(path: str, members: list) -> str:
    with zipfile.ZipFile(path, 'w') as zip_file:
        for name in members:
            zip_file.writestr(name, b'data')
    return path


def test_mvbench_exact_match_not_confused_by_substring(tmp_path):
    # A request for `2.mp4` must not resolve to `12.mp4`, which merely shares the suffix
    # and would otherwise sort first.
    archive = _write_zip(str(tmp_path / 'star.zip'), ['star/12.mp4', 'star/2.mp4'])
    assert mvbench_find_archive_member(archive, 'action_sequence', '2.mp4') == 'star/2.mp4'


def test_mvbench_root_level_exact_match(tmp_path):
    archive = _write_zip(str(tmp_path / 'root.zip'), ['12.mp4', '2.mp4'])
    assert mvbench_find_archive_member(archive, 'action_count', '2.mp4') == '2.mp4'


def test_mvbench_prefers_subset_directory(tmp_path):
    archive = _write_zip(str(tmp_path / 'clevrer.zip'), ['moving_count/v.mp4', 'moving_direction/v.mp4'])
    assert mvbench_find_archive_member(archive, 'moving_direction', 'v.mp4') == 'moving_direction/v.mp4'


def test_mvbench_basename_fallback(tmp_path):
    # No member matches the full relative path, but the basename is unique, so we fall back to it
    # while still ignoring substring-only candidates such as `xS001_rgb.avi`.
    archive = _write_zip(str(tmp_path / 'nturgbd.zip'), ['nturgbd_rgb/S001_rgb.avi', 'nturgbd_rgb/xS001_rgb.avi'])
    assert mvbench_find_archive_member(archive, 'fine_grained_pose', 'S001_rgb.avi') == 'nturgbd_rgb/S001_rgb.avi'


def test_mvbench_missing_video_raises(tmp_path):
    archive = _write_zip(str(tmp_path / 'star.zip'), ['star/1.mp4'])
    with pytest.raises(FileNotFoundError):
        mvbench_find_archive_member(archive, 'action_sequence', '999.mp4')


def test_videomme_exact_match_not_confused_by_substring(tmp_path):
    # Video id `1` normalizes to `001.mp4`; it must not resolve to `1001.mp4`.
    archive = _write_zip(str(tmp_path / '001.zip'), ['data/1001.mp4', 'data/001.mp4'])
    assert videomme_find_archive_member(archive, '1') == 'data/001.mp4'


def test_videomme_missing_video_raises(tmp_path):
    archive = _write_zip(str(tmp_path / '001.zip'), ['data/002.mp4'])
    with pytest.raises(FileNotFoundError):
        videomme_find_archive_member(archive, '1')
