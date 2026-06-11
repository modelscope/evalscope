"""High-frequency concurrent-write tests for :class:`JsonlWriter`.

These tests verify that the persistent-writer path used by
:class:`CacheManager` survives the scenario that produced ``PermissionError``
on Windows under the old open-write-close-per-record approach:

- :class:`ThreadPoolExecutor` runs many worker tasks concurrently.
- The **main thread** persists each result as its future completes.
- Rapid open/close cycles on the same file trigger NTFS file-lock
  release latency.

The persistent-writer approach keeps a single handle open and flushes
after every write, which is immune to that failure mode.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from evalscope.api.evaluator.cache import CacheManager
from evalscope.utils.io_utils import JsonlWriter, OutputsStructure


def _make_outputs(tmp_dir: str) -> OutputsStructure:
    return OutputsStructure(tmp_dir, is_make=True)


def _make_cm(tmp_dir: str) -> CacheManager:
    return CacheManager(
        outputs=_make_outputs(tmp_dir),
        model_name='mock-model',
        benchmark_name='mock-bench',
    )


class TestJsonlWriterBasics:
    """Low-level behavior of :class:`JsonlWriter`."""

    def test_write_creates_file(self, tmp_path):
        path = os.path.join(str(tmp_path), 'out.jsonl')
        writer = JsonlWriter(path)
        writer.write({'a': 1})
        writer.close()

        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines == [{'a': 1}]

    def test_write_creates_parent_dirs(self, tmp_path):
        path = os.path.join(str(tmp_path), 'a', 'b', 'c', 'out.jsonl')
        writer = JsonlWriter(path)
        writer.write({'k': 'v'})
        assert os.path.isdir(os.path.dirname(path))
        writer.close()

    def test_close_is_idempotent(self, tmp_path):
        path = os.path.join(str(tmp_path), 'out.jsonl')
        writer = JsonlWriter(path)
        writer.write({'a': 1})
        writer.close()
        writer.close()  # must not raise

    def test_flush_writes_to_disk_immediately(self, tmp_path):
        """After ``write()``, the record must be readable while the writer is still open."""
        path = os.path.join(str(tmp_path), 'out.jsonl')
        writer = JsonlWriter(path)
        writer.write({'k': 'v'})

        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines == [{'k': 'v'}]
        writer.close()

    def test_unicode_content(self, tmp_path):
        path = os.path.join(str(tmp_path), 'out.jsonl')
        writer = JsonlWriter(path)
        writer.write({'text': '你好世界'})
        writer.close()

        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines == [{'text': '你好世界'}]


class TestCacheManagerWriterReuse:
    """Verify CacheManager opens each file exactly once and reuses the handle."""

    def test_get_writer_reuses_same_handle(self, tmp_path):
        cm = _make_cm(str(tmp_path))
        cache_file = os.path.join(str(tmp_path), 'predictions', 'mock-model', 'x.jsonl')

        w1 = cm._get_writer(cache_file)
        w2 = cm._get_writer(cache_file)
        assert w1 is w2, 'same file must return the same writer instance'
        assert len(cm._writers) == 1
        cm.close()

    def test_one_writer_per_file(self, tmp_path):
        """100 writes to the same file must open the file exactly once."""
        cm = _make_cm(str(tmp_path))
        cache_file = os.path.join(str(tmp_path), 'predictions', 'mock-model', 'x.jsonl')

        open_count = {'n': 0}
        real_open = open

        def counting_open(*args, **kwargs):
            open_count['n'] += 1
            return real_open(*args, **kwargs)

        with patch('evalscope.utils.io_utils.open', side_effect=counting_open):
            for i in range(100):
                cm._get_writer(cache_file).write({'i': i})
            cm.close()

        assert open_count['n'] == 1, f'expected exactly 1 open, got {open_count["n"]}'


class TestHighFrequencyConcurrentWrites:
    """Simulate the evaluator loop: N worker threads finish at near-simultaneous
    times; the main thread persists each result as its future completes.
    """

    NUM_SUBSETS = 3
    RECORDS_PER_SUBSET = 50
    WORKER_THREADS = 8

    def _run_pool(self, cm: CacheManager):
        """Mimic evaluator._run_pool: workers produce results, main thread persists."""
        work = [(subset, i) for subset in range(self.NUM_SUBSETS) for i in range(self.RECORDS_PER_SUBSET)]

        def worker(subset_idx, record_idx):
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
                cm._get_writer(cache_file).write(payload)

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
        assert len(cm._writers) == self.NUM_SUBSETS
        cm.close()

    def test_no_permission_error_under_burst(self, tmp_path):
        """Rapid burst of 500 writes to a single file must not raise."""
        cm = _make_cm(str(tmp_path))
        cache_file = os.path.join(str(tmp_path), 'predictions', 'mock-model', 'burst.jsonl')

        for i in range(500):
            cm._get_writer(cache_file).write({'i': i})
        cm.close()

        with open(cache_file, 'r', encoding='utf-8') as f:
            assert sum(1 for _ in f) == 500
