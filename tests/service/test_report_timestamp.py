import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from evalscope.report import ReportRef
from evalscope.service.blueprints.reports import _extract_timestamp

STARTED = datetime(2026, 3, 1, 9, 30, 0)
RESTORED = datetime(2026, 10, 30, 16, 0, 0)


def _make_run(root: Path, run_id: str, manifest: Optional[dict] = None, mtime: datetime = RESTORED) -> ReportRef:
    """Create a run directory, optionally with a task manifest, and set its modification time."""
    run_dir = root / run_id
    run_dir.mkdir()
    if manifest is not None:
        (run_dir / 'task_info.json').write_text(json.dumps(manifest), encoding='utf-8')
    os.utime(run_dir, (mtime.timestamp(), mtime.timestamp()))
    return ReportRef(run_id=run_id, model_id='model')


def test_manifest_started_at_wins_over_directory_mtime(tmp_path: Path) -> None:
    ref = _make_run(tmp_path, 'eval_1', {'started_at': STARTED.isoformat()})

    assert _extract_timestamp(ref, str(tmp_path)) == STARTED.isoformat()


def test_web_ui_run_id_encodes_the_start_time(tmp_path: Path) -> None:
    ref = _make_run(tmp_path, f'eval_{int(STARTED.timestamp() * 1000)}')

    assert _extract_timestamp(ref, str(tmp_path)) == STARTED.isoformat()


def test_invalid_manifest_start_time_is_ignored(tmp_path: Path) -> None:
    ref = _make_run(tmp_path, f'eval_{int(STARTED.timestamp() * 1000)}', {'started_at': 'not a date'})

    assert _extract_timestamp(ref, str(tmp_path)) == STARTED.isoformat()


def test_dated_directory_names_still_parse(tmp_path: Path) -> None:
    ref = _make_run(tmp_path, '20260423_201338')

    assert _extract_timestamp(ref, str(tmp_path)) == datetime(2026, 4, 23, 20, 13, 38).isoformat()


def test_directory_mtime_is_the_last_resort(tmp_path: Path) -> None:
    for run_id in ('custom-run', 'eval_123'):
        ref = _make_run(tmp_path, run_id)

        assert _extract_timestamp(ref, str(tmp_path)) == RESTORED.isoformat()
