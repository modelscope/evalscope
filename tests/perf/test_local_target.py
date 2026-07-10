import asyncio
import pytest
import sys
from types import SimpleNamespace
from typing import Any, Optional

from evalscope.perf import TargetConfig
from evalscope.perf.domain.errors import PerfRunError
from evalscope.perf.serving.local_target import ManagedTarget


class FakeProcess:

    def __init__(self, command: list, **kwargs: Any) -> None:
        self.command = command
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self) -> Optional[int]:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self) -> int:
        return self.returncode


def test_local_vllm_uses_current_python_and_is_terminated(monkeypatch, tmp_path) -> None:
    processes = []

    def popen(command: list, **kwargs: Any) -> FakeProcess:
        process = FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    monkeypatch.setattr('evalscope.perf.serving.local_target.subprocess.Popen', popen)

    async def run() -> None:
        target = ManagedTarget(
            TargetConfig(model='fake', kind='local_vllm', port=8123),
            str(tmp_path / 'target.log'),
        )
        async with target:
            target.ensure_running()

    asyncio.run(run())
    assert processes[0].command[0] == sys.executable
    assert '--tensor-parallel-size' in processes[0].command
    assert processes[0].terminated


def test_local_target_reports_early_process_exit(tmp_path) -> None:
    target = ManagedTarget(TargetConfig(model='fake'), str(tmp_path / 'target.log'))
    target._process = FakeProcess([])
    target._process.returncode = 2
    with pytest.raises(PerfRunError, match='exited with code 2'):
        target.ensure_running()
