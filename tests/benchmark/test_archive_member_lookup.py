import zipfile

import pytest

from evalscope.benchmarks.mvbench.utils import find_archive_member as mvbench_find_archive_member


def _write_zip(path: str, members: list[str]) -> str:
    with zipfile.ZipFile(path, 'w') as zip_file:
        for name in members:
            zip_file.writestr(name, b'data')
    return path


def test_mvbench_real_ssv2_collision_resolves_correctly(tmp_path):
    # Real regression from the default `action_antonym` subset (ssv2_video.zip).
    # Something-Something-v2 ids are non-zero-padded, variable-length numbers, so the archive
    # legitimately contains both `9741.mp4` and `209741.mp4`. The old suffix lookup matched both
    # and `sorted(...)[0]` returned `ssv2_video/209741.mp4` (the wrong video). The annotation that
    # requests `9741.mp4` must resolve to `ssv2_video/9741.mp4`.
    archive = _write_zip(str(tmp_path / 'ssv2_video.zip'), ['ssv2_video/209741.mp4', 'ssv2_video/9741.mp4'])
    assert mvbench_find_archive_member(archive, 'action_antonym', '9741.mp4') == 'ssv2_video/9741.mp4'


def test_mvbench_prefers_subset_directory(tmp_path):
    archive = _write_zip(str(tmp_path / 'clevrer.zip'), ['moving_count/v.mp4', 'moving_direction/v.mp4'])
    assert mvbench_find_archive_member(archive, 'moving_direction', 'v.mp4') == 'moving_direction/v.mp4'


def test_mvbench_basename_fallback(tmp_path):
    # No member matches the full relative path, but the basename is unique, so we fall back to it
    # while still ignoring substring-only candidates such as `x9741.mp4`.
    archive = _write_zip(str(tmp_path / 'ssv2_video.zip'), ['other/9741.mp4', 'other/x9741.mp4'])
    assert mvbench_find_archive_member(archive, 'action_antonym', '9741.mp4') == 'other/9741.mp4'


def test_mvbench_missing_video_raises(tmp_path):
    archive = _write_zip(str(tmp_path / 'ssv2_video.zip'), ['ssv2_video/9741.mp4'])
    with pytest.raises(FileNotFoundError):
        mvbench_find_archive_member(archive, 'action_antonym', '999999.mp4')
