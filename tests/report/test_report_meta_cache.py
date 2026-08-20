import os
import threading
from pathlib import Path
from typing import Optional

from evalscope.report import ReportRef
from evalscope.service.report_meta_cache import (
    build_report_meta_cached,
    clear_report_meta_cache,
    report_ref_fingerprint,
)


def setup_function() -> None:
    clear_report_meta_cache()


def test_fingerprint_tracks_run_directory_mtime(tmp_path: Path) -> None:
    ref = ReportRef(run_id='custom-run', model_id='model')
    model_dir = tmp_path / ref.run_id / 'reports' / ref.model_id
    model_dir.mkdir(parents=True)
    (model_dir / 'report.json').write_text('{}', encoding='utf-8')

    before = report_ref_fingerprint(str(tmp_path), ref)
    run_dir = tmp_path / ref.run_id
    stat = run_dir.stat()
    os.utime(run_dir, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert report_ref_fingerprint(str(tmp_path), ref) != before


def test_newer_concurrent_result_is_not_overwritten(tmp_path: Path) -> None:
    ref = ReportRef(run_id='run', model_id='model')
    old_fingerprint = (('report.json', 1, 1, 1), )
    new_fingerprint = (('report.json', 2, 2, 2), )
    started = threading.Event()
    release = threading.Event()

    def compute_old(_ref: ReportRef, _root: str) -> dict:
        started.set()
        release.wait(timeout=2)
        return {'version': 'old'}

    worker = threading.Thread(
        target=build_report_meta_cached,
        args=(str(tmp_path), ref, old_fingerprint, compute_old),
    )
    worker.start()
    assert started.wait(timeout=2)

    assert build_report_meta_cached(
        str(tmp_path), ref, new_fingerprint, lambda _ref, _root: {'version': 'new'}
    ) == {'version': 'new'}
    release.set()
    worker.join(timeout=2)

    def fail_compute(_ref: ReportRef, _root: str) -> dict:
        raise AssertionError('newer cached result was overwritten')

    assert build_report_meta_cached(str(tmp_path), ref, new_fingerprint, fail_compute) == {'version': 'new'}


def test_failed_metadata_is_retried(tmp_path: Path) -> None:
    ref = ReportRef(run_id='run', model_id='model')
    fingerprint = (('report.json', 1, 1, 1), )
    attempts = 0

    def compute(_ref: ReportRef, _root: str) -> Optional[dict]:
        nonlocal attempts
        attempts += 1
        return None if attempts == 1 else {'ok': True}

    assert build_report_meta_cached(str(tmp_path), ref, fingerprint, compute) is None
    assert build_report_meta_cached(str(tmp_path), ref, fingerprint, compute) == {'ok': True}
    assert attempts == 2
