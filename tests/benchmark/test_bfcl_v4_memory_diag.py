"""Diagnostics for BFCL-v4 memory prereq snapshots (#1100).

Memory categories score by string-matching the model's final text message
against the memorized facts, so a missing or empty prereq snapshot silently
guarantees 0 for every memory entry. These tests pin the warnings that make
that failure visible.
"""
import json
import logging

import pytest

from evalscope.benchmarks.bfcl.v4.utils import _warn_on_empty_memory_snapshots


def _entry(entry_id: str) -> dict:
    return {'id': entry_id}


def _snapshot_dir(model_result_dir, backend: str):
    # mirror get_directory_structure_by_id('memory_{backend}_prereq_...') layout
    return model_result_dir / 'agentic' / 'memory' / backend / 'memory_snapshot'


def _write_snapshot(model_result_dir, backend: str, scenario: str, core: list, archival: list) -> None:
    snapshot_dir = _snapshot_dir(model_result_dir, backend)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / f'{scenario}_final.json').write_text(
        json.dumps({'core_memory': core, 'archival_memory': archival})
    )


def _groups(backend: str, scenario: str) -> dict:
    entry_id = f'memory_{backend}_prereq_0-{scenario}-0'
    return {(backend, scenario): [_entry(entry_id)]}


def _caplog_records(caplog):
    return [r.getMessage() for r in caplog.records]


def test_missing_snapshot_warns(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    _warn_on_empty_memory_snapshots(_groups('kv', 'customer'), tmp_path, logging.getLogger())

    messages = ' '.join(_caplog_records(caplog))
    assert 'no snapshot' in messages
    assert 'kv/customer' in messages


def test_empty_snapshot_warns_with_cause(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    _write_snapshot(tmp_path, 'kv', 'customer', core=[], archival=[])
    _warn_on_empty_memory_snapshots(_groups('kv', 'customer'), tmp_path, logging.getLogger())

    messages = ' '.join(_caplog_records(caplog))
    assert 'no memory entries' in messages
    assert 'memory API call' in messages


def test_populated_snapshot_is_silent(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    _write_snapshot(tmp_path, 'kv', 'customer', core=['Michael likes apples'], archival=[])
    _warn_on_empty_memory_snapshots(_groups('kv', 'customer'), tmp_path, logging.getLogger())

    assert _caplog_records(caplog) == []


def test_corrupt_snapshot_warns(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    snapshot_dir = _snapshot_dir(tmp_path, 'kv')
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / 'customer_final.json').write_text('{not json')
    _warn_on_empty_memory_snapshots(_groups('kv', 'customer'), tmp_path, logging.getLogger())

    messages = ' '.join(_caplog_records(caplog))
    assert 'unreadable' in messages


def test_no_prereq_groups_is_noop(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    _warn_on_empty_memory_snapshots({}, tmp_path, logging.getLogger())

    assert _caplog_records(caplog) == []
