"""High-frequency concurrent-write tests for the persistent jsonl writer in CacheManager.

These tests simulate the real evaluator loop: a :class:`ThreadPoolExecutor` runs
many worker tasks concurrently, and the **main thread** immediately persists
each result as its future completes. On Windows, the old
``dump_jsonl_data`` (open-write-close per record) path triggered ``PermissionError``
due to NTFS file-lock release latency under rapid open/close cycles. The new
persistent-writer path keeps a single handle open and flushes after every write,
which should be immune to that failure mode.
"""
import json
import os
import pytest
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from evalscope.api.evaluator.cache import CacheManager
from evalscope.utils.io_utils import OutputsStructure


def _make_outputs(tmp_dir: str) -> OutputsStructure:
    return OutputsStructure(tmp_dir, is_make=True)


def _make_cm(tmp_dir: str) -> CacheManager:
    return CacheManager(
        outputs=_make_outputs(tmp_dir),
        model_name='mock-model',
        benchmark_name='mock-bench',
    )


class TestPersistentWriterBasics:
    """Low-level behavior of _get_writer / close / flush."""

    def test_get_writer_reuses_same_handle(self, tmp_path):
        cm = _make_cm(str(tmp_path))
        cache_file = os.path.join(str(tmp_path), 'predictions', 'mock-model', 'x.jsonl')

        w1 = cm._get_writer(cache_file)
        w2 = cm._get_writer(cache_file)
        assert w1 is w2, 'same file must return the same writer instance'
        assert len(cm._writers) == 1
        cm.close()

    def test_get_writer_creates_parent_dirs(self, tmp_path):
        cm = _make_cm(str(tmp_path))
        deep = os.path.join(str(tmp_path), 'a', 'b', 'c', 'out.jsonl')
        cm._get_writer(deep)
        assert os.path.isdir(os.path.dirname(deep))
        cm.close()

    def test_close_is_idempotent(self, tmp_path):
        cm = _make_cm(str(tmp_path))
        cache_file = os.path.join(str(tmp_path), 'predictions', 'mock-model', 'x.jsonl')
        cm._get_writer(cache_file).write({'a': 1})
        cm.close()
        cm.close()  # must not raise
        assert cm._writers == {}

    def test_flush_writes_to_disk_immediately(self, tmp_path):
        cm = _make_cm(str(tmp_path))
        cache_file = os.path.join(str(tmp_path), 'predictions', 'mock-model', 'x.jsonl')
        writer = cm._get_writer(cache_file)
        writer.write({'k': 'v'})
        CacheManager._flush_writer(writer)

        # File must be readable while writer is still open.
        with open(cache_file, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines == [{'k': 'v'}]
        cm.close()


class TestHighFrequencyConcurrentWrites:
    """Simulate the evaluator loop: N worker threads finish at near-simultaneous
    times; the main thread persists each result as its future completes.

    This is the scenario that produced the ``PermissionError`` on Windows under
    the old open-write-close-per-record path.
    """

    NUM_SUBSETS = 3
    RECORDS_PER_SUBSET = 50
    WORKER_THREADS = 8

    def _run_pool(self, cm: CacheManager):
        """Mimic evaluator._run_pool: workers produce results, main thread persists."""
        work = [(subset, i) for subset in range(self.NUM_SUBSETS) for i in range(self.RECORDS_PER_SUBSET)]

        def worker(subset_idx, record_idx):
            # Spread completions over a tight window to maximize contention.
            time.sleep((record_idx % self.WORKER_THREADS) * 0.001)
            return subset_idx, record_idx, {'s': subset_idx, 'i': record_idx, 't': time.time()}

        with ThreadPoolExecutor(max_workers=self.WORKER_THREADS) as pool:
            futures = [pool.submit(worker, s, i) for s, i in work]
            for fut in as_completed(futures):
                subset_idx, record_idx, payload = fut.result()
                cache_file = os.path.join(
                    str(cm.outputs.outputs_dir),
                    'predictions',
                    cm.model_name,
                    f'subset_{subset_idx}.jsonl',
                )
                writer = cm._get_writer(cache_file)
                writer.write(payload)
                CacheManager._flush_writer(writer)

    def test_all_records_persisted(self, tmp_path):
        cm = _make_cm(str(tmp_path))
        self._run_pool(cm)
        cm.close()

        pred_dir = os.path.join(str(tmp_path), 'predictions', 'mock-model')
        total = 0
        for subset_idx in range(self.NUM_SUBSETS):
            path = os.path.join(pred_dir, f'subset_{subset_idx}.jsonl')
            with open(path, 'r', encoding='utf-8') as f:
                lines = [json.loads(line) for line in f if line.strip()]
            assert len(lines) == self.RECORDS_PER_SUBSET
            total += len(lines)
        assert total == self.NUM_SUBSETS * self.RECORDS_PER_SUBSET

    def test_one_writer_per_file_regardless_of_write_count(self, tmp_path):
        cm = _make_cm(str(tmp_path))
        self._run_pool(cm)
        # Exactly one writer per subset file, not per record.
        assert len(cm._writers) == self.NUM_SUBSETS
        cm.close()

    def test_no_permission_error_under_burst(self, tmp_path):
        """Force a worst-case burst: all futures complete at once, main thread
        persists them back-to-back without any delay. A PermissionError here
        would indicate the writer is still doing open/close cycles somewhere.
        """
        cm = _make_cm(str(tmp_path))
        cache_file = os.path.join(str(tmp_path), 'predictions', 'mock-model', 'burst.jsonl')

        # Rapid burst of 500 writes to a single file.
        for i in range(500):
            writer = cm._get_writer(cache_file)
            writer.write({'i': i})
            CacheManager._flush_writer(writer)
        cm.close()

        with open(cache_file, 'r', encoding='utf-8') as f:
            assert sum(1 for _ in f) == 500

    def test_old_path_would_open_many_times(self, tmp_path):
        """Document the gap between the old approach (open/close per record) and
        the new one. The old path called jsonlines.open once per record; the new
        path calls it exactly once per distinct file.
        """
        cm = _make_cm(str(tmp_path))

        open_count = {'n': 0}
        real_open = __import__('jsonlines').open

        def counting_open(*args, **kwargs):
            open_count['n'] += 1
            return real_open(*args, **kwargs)

        with patch('evalscope.api.evaluator.cache.jsonl.open', side_effect=counting_open):
            for i in range(100):
                cache_file = os.path.join(str(tmp_path), 'predictions', 'mock-model', 'x.jsonl')
                writer = cm._get_writer(cache_file)
                writer.write({'i': i})
                CacheManager._flush_writer(writer)
            cm.close()

        assert open_count['n'] == 1, (f'persistent writer must open the file exactly once, got {open_count["n"]}')
