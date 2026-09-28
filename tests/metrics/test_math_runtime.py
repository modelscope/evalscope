"""Cross-platform process lifecycle tests, including hard deadlines and recovery."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from evalscope.metrics.math.contracts import MathEvaluationError, MathRequest
from evalscope.metrics.math.runtime import MathWorkerPool


@pytest.fixture
def command(tmp_path: Any) -> Any:
    worker = tmp_path / 'worker.py'
    worker.write_text('''import json, os, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request['prediction'] == 'hang': time.sleep(60)
    if request['prediction'] == 'crash': os._exit(2)
    if request['prediction'] == 'error':
        print(json.dumps({'error':'execution failure'}), flush=True)
    else:
        print(json.dumps({'matched':True, 'extracted':str(os.getpid())}), flush=True)
''')
    return [sys.executable, '-u', str(worker)]


@pytest.mark.parametrize('failure', ['hang','crash','error'])
def test_failed_child_is_reaped_and_replaced(command: Any, failure: Any) -> None:
    with MathWorkerPool(max_workers=1, timeout=0.8, worker_command=command) as pool:
        first = pool.execute(MathRequest(prediction='good')).extracted
        child = next(iter(pool._workers.values())).process
        started = time.monotonic()
        with pytest.raises(MathEvaluationError):
            pool.execute(MathRequest(prediction=failure))
        assert time.monotonic() - started < 3
        assert child.poll() is not None
        recovered = pool.execute(MathRequest(prediction='good'))
        assert recovered.matched and recovered.extracted != first
        children = [worker.process for worker in pool._workers.values()]
    assert all(child.poll() is not None for child in children)


def test_close_cancels_running_and_queued_requests(command: Any) -> None:
    pool = MathWorkerPool(max_workers=1, timeout=5, worker_command=command)
    with ThreadPoolExecutor(max_workers=2) as executor:
        running = executor.submit(pool.execute, MathRequest(prediction='hang'))
        queued = executor.submit(pool.execute, MathRequest(prediction='good'))
        limit = time.monotonic() + 2
        while not pool._workers and time.monotonic() < limit:
            time.sleep(0.01)
        children = [worker.process for worker in pool._workers.values()]
        pool.close()
        for future in [running, queued]:
            with pytest.raises(MathEvaluationError):
                future.result(timeout=2)
    assert all(child.poll() is not None for child in children)


def test_unguarded_script_and_interpreter_exit(tmp_path: Any) -> None:
    script = tmp_path / 'unguarded.py'
    script.write_text('''import json
from evalscope.metrics.math.parser import math_equal
from evalscope.metrics.math import runtime
assert math_equal('2', '2')
print(json.dumps([w.process.pid for w in runtime._pool._workers.values()]))
''')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=30,check=True,env=env)
    children = json.loads(result.stdout.strip().splitlines()[-1])
    assert len(children) == 1
    for pid in children:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.windll.kernel32
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                code = ctypes.c_ulong()
                assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                kernel.CloseHandle(handle)
                assert code.value != 259
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)


def test_pool_bounds(command: Any) -> None:
    for workers in [0,5]:
        with pytest.raises(ValueError):
            MathWorkerPool(max_workers=workers,worker_command=command)


def test_deadline_includes_a_blocked_stdin_write() -> None:
    command = [sys.executable,'-u','-c','import time; time.sleep(60)']
    started = time.monotonic()
    with MathWorkerPool(max_workers=1,timeout=0.5,worker_command=command) as pool:
        with pytest.raises(MathEvaluationError):
            pool.execute(MathRequest(prediction='x' * 1_000_000))
    assert time.monotonic() - started < 3


def test_real_math_timeout_and_recovery() -> None:
    with MathWorkerPool(max_workers=1) as pool:
        assert pool.execute(MathRequest(prediction='2', reference='2')).matched
        child = next(iter(pool._workers.values())).process
        pool.timeout = 0.1
        started = time.monotonic()
        with pytest.raises(MathEvaluationError, match='deadline'):
            pool.execute(MathRequest(prediction=r'2^{2^{30}}', reference='1'))
        assert time.monotonic() - started < 3
        assert child.poll() is not None
        pool.timeout = 15
        assert pool.execute(MathRequest(prediction='2', reference='2')).matched


def test_cancellation_reaps_worker(command: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.metrics.math.runtime import _Worker

    with MathWorkerPool(max_workers=1, worker_command=command) as pool:
        pool.execute(MathRequest(prediction='good'))
        child = next(iter(pool._workers.values())).process
        with monkeypatch.context() as patch:
            def cancel(*args: object, **kwargs: object) -> None:
                raise KeyboardInterrupt

            patch.setattr(_Worker, 'execute', cancel)
            with pytest.raises(KeyboardInterrupt):
                pool.execute(MathRequest(prediction='good'))
        assert child.poll() is not None
        assert pool.execute(MathRequest(prediction='good')).matched


def test_failed_task_session_reaps_workers() -> None:
    from evalscope.metrics.math import runtime
    from evalscope.metrics.math.parser import math_equal

    with pytest.raises(RuntimeError):
        with runtime.math_worker_session():
            assert math_equal('2', '2')
            children = [w.process for w in runtime._pool._workers.values()]
            raise RuntimeError('task failed')
    assert runtime._pool is None
    assert all(child.poll() is not None for child in children)


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_evaluator_failure_reaps_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    from evalscope import TaskConfig, run_task
    from evalscope.evaluator.evaluator import DefaultEvaluator
    from evalscope.metrics.math import runtime
    from evalscope.metrics.math.parser import math_equal

    data = tmp_path / 'data'
    data.mkdir()
    (data / 'test.jsonl').write_text(json.dumps({'question': 'One plus one?', 'answer': 'Reasoning. #### 2'}) + '\n')
    children = []

    def fail_scoring(*args: Any, **kwargs: Any) -> None:
        assert math_equal('2', '2')
        children.extend(worker.process for worker in runtime._pool._workers.values())
        raise failure('evaluation interrupted')

    monkeypatch.setattr(DefaultEvaluator, '_collect_work_items', fail_scoring)
    with pytest.raises(failure, match='evaluation interrupted'):
        run_task(
            TaskConfig(
                model='offline',
                eval_type='mock_llm',
                datasets=['gsm8k'],
                no_timestamp=True,
                work_dir=str(tmp_path / 'run'),
                judge={'strategy': 'rule'},
                dataset_args={
                    'gsm8k': {'local_path': str(data), 'few_shot_num': 0, 'subset_list': ['default']}
                },
            )
        )
    assert children and runtime._pool is None
    assert all(child.poll() is not None for child in children)
