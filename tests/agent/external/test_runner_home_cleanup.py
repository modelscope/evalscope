"""The CLI runners that create a temporary HOME must remove it after ``run``.

Mirrors the ``owns_home_dir`` / ``finally`` cleanup already used by the
claude-code, opencode and deepseek-harness runners.
"""

import os
from typing import Dict, List, Optional, Tuple

import pytest

from evalscope.agent.external.runners.base import BridgeEndpoint, ExternalAgentTask
from evalscope.agent.external.runners.codex import CodexRunner
from evalscope.agent.external.runners.gemini_cli import GeminiCliRunner
from evalscope.agent.external.runners.hermes import HermesRunner
from evalscope.api.agent.types import ExecResult
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner


class FakeEnvironment:
    """Records every exec call; the CLI invocation itself can be made to fail."""

    name = 'fake'

    def __init__(self, cli: str, cli_returncode: int = 0) -> None:
        self._cli = cli
        self._cli_returncode = cli_returncode
        self.calls: List[Tuple[List[str], Optional[float], Optional[Dict[str, str]]]] = []

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        del cwd, input
        self.calls.append((cmd, timeout, env))
        if cmd and cmd[0] == self._cli:
            return ExecResult(returncode=self._cli_returncode, stdout='42\n', stderr='boom', duration=1.0)
        return ExecResult(stdout='42\n')

    def home_dir(self, key: str) -> str:
        """The HOME-like path the runner handed to its CLI invocation."""
        for cmd, _, env in self.calls:
            if cmd and cmd[0] == self._cli:
                assert env is not None
                return env[key]
        raise AssertionError(f'{self._cli} was never invoked: {self.calls}')


@pytest.fixture(autouse=True)
def _release_bridge_loop() -> None:
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _task() -> ExternalAgentTask:
    return ExternalAgentTask(instruction='Reply with 42.', timeout=20.0, metadata={'sample_id': 'sample-1'})


def _bridge() -> BridgeEndpoint:
    return BridgeEndpoint(base_url='http://127.0.0.1:12345', trial_token='trial-secret')


RUNNERS = [
    pytest.param(CodexRunner, 'codex', 'HOME', id='codex'),
    pytest.param(HermesRunner, 'hermes', 'HERMES_HOME', id='hermes'),
    pytest.param(GeminiCliRunner, 'gemini', 'HOME', id='gemini-cli'),
]


@pytest.mark.parametrize('runner_cls, cli, home_key', RUNNERS)
def test_temporary_home_is_removed_after_success(runner_cls, cli, home_key) -> None:
    env = FakeEnvironment(cli)

    AsyncioLoopRunner.run(runner_cls().run(_task(), env, _bridge()))

    home_dir = env.home_dir(home_key)
    assert home_dir.startswith(os.path.join(os.path.realpath('/tmp'), '')) or 'evalscope-' in home_dir
    assert not os.path.exists(home_dir)


@pytest.mark.parametrize('runner_cls, cli, home_key', RUNNERS)
def test_temporary_home_is_removed_after_cli_failure(runner_cls, cli, home_key) -> None:
    env = FakeEnvironment(cli, cli_returncode=1)

    with pytest.raises(RuntimeError, match='exited with code 1'):
        AsyncioLoopRunner.run(runner_cls().run(_task(), env, _bridge()))

    assert not os.path.exists(env.home_dir(home_key))


@pytest.mark.parametrize('runner_cls, cli, home_key', RUNNERS)
def test_user_supplied_home_is_kept(runner_cls, cli, home_key, tmp_path) -> None:
    env = FakeEnvironment(cli)
    marker = tmp_path / 'keep-me'
    marker.write_text('x')

    AsyncioLoopRunner.run(runner_cls(home_override=str(tmp_path)).run(_task(), env, _bridge()))

    assert env.home_dir(home_key) == str(tmp_path)
    assert marker.exists()
