# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for user-initiated stop handling in service subprocess utilities.

Covers the interaction between ``run_in_subprocess`` and ``stop_process``:
a user-initiated stop must surface as :class:`TaskStoppedError` (graceful),
while a genuine child crash must still raise ``RuntimeError``.
"""
import multiprocessing
import threading
import time
import unittest

from evalscope.service.utils.process import (
    TaskStoppedError,
    _user_stopped_tasks,
    register_process,
    run_in_subprocess,
    stop_process,
)


def _sleep_forever() -> str:
    time.sleep(60)
    return 'done'


def _quick_success() -> str:
    return 'ok'


def _crash() -> None:
    raise ValueError('boom')


class TestProcessStop(unittest.TestCase):

    def test_user_stop_raises_task_stopped_error(self):
        """Stopping a running task raises TaskStoppedError, not RuntimeError."""
        task_id = 'test-stop-graceful'
        outcome = {}

        def run() -> None:
            try:
                outcome['result'] = run_in_subprocess(_sleep_forever, task_id=task_id)
            except Exception as e:  # noqa: BLE001 - capture for assertion
                outcome['exc'] = e

        t = threading.Thread(target=run)
        t.start()
        time.sleep(3)  # let the spawned child start sleeping

        self.assertTrue(stop_process(task_id))
        t.join(timeout=30)

        self.assertIsInstance(outcome.get('exc'), TaskStoppedError)

    def test_success_returns_result(self):
        """A normal run returns the child's result unchanged."""
        self.assertEqual(run_in_subprocess(_quick_success, task_id='test-stop-success'), 'ok')

    def test_child_crash_still_raises_runtime_error(self):
        """A genuine child error is not misreported as a user stop."""
        with self.assertRaises(RuntimeError):
            run_in_subprocess(_crash, task_id='test-stop-crash')

    def test_stop_unknown_task_returns_false(self):
        """Stopping a non-existent task is a no-op returning False."""
        self.assertFalse(stop_process('test-stop-nonexistent'))

    def test_stop_after_natural_exit_is_noop(self):
        """Stopping an already-exited (but still registered) process must not
        set the user-stop marker, so a pending success result is not masked."""
        task_id = 'test-stop-already-dead'
        proc = multiprocessing.get_context('spawn').Process(target=_quick_success)
        proc.start()
        proc.join()  # let it exit naturally before the stop request
        register_process(task_id, proc)

        self.assertFalse(stop_process(task_id))
        self.assertNotIn(task_id, _user_stopped_tasks)


if __name__ == '__main__':
    unittest.main()
