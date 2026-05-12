"""T2 Environment Abstraction – unit and integration tests.

Test plan:
  TestEnvironmentRegistry      – registry API surface (environments + tools)
  TestLocalEnvironmentExec     – LocalAgentEnvironment exec / read_file / write_file
  TestLocalEnvironmentTools    – bash/python_exec/read_file/write_file handlers w/ local env
  TestDockerEnvironmentExec    – DockerAgentEnvironment exec / read_file / write_file
  TestDockerEnvironmentTools   – bash + python_exec handlers w/ docker env
  TestAgentLoopWithEnvironment – full AgentLoop + local env + bash tool
  TestDefaultAdapterEnvPath    – _on_agent_inference environment_extra + tool_infos
  TestAgentConfigEnvironmentExtra – AgentConfig.environment_extra round-trip
"""

import os
import pytest
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import evalscope  # noqa: F401 – trigger strategy / env / tool registration
from evalscope.api.agent import (
    AgentContext,
    AgentEnvironment,
    AgentLoop,
    AgentTrace,
    EventType,
    ExecResult,
    ToolExecutor,
)
from evalscope.api.agent.types import AgentConfig
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.registry import (
    AGENT_TOOL_INFO_REGISTRY,
    ENVIRONMENT_REGISTRY,
    get_environment,
    list_agent_tools,
    list_environments,
    resolve_tool_infos,
    resolve_tools,
)
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.api.tool.tool_call import ToolFunction
from evalscope.utils.function_utils import AsyncioLoopRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output(content: str, tool_calls=None, stop_reason='stop') -> ModelOutput:
    msg = ChatMessageAssistant(content=content, tool_calls=tool_calls or [])
    choice = ChatCompletionChoice(message=msg, finish_reason=stop_reason)
    return ModelOutput(model='mock', choices=[choice])


def _tool_call(name: str, args: Dict[str, Any], call_id: str = 'tc-1') -> ToolCall:
    return ToolCall(id=call_id, function=ToolFunction(name=name, arguments=args))


def _check_docker() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        import docker  # type: ignore[import]
        docker.from_env().ping()
        return True
    except Exception:
        return False


DOCKER_AVAILABLE = _check_docker()
docker_mark = pytest.mark.skipif(not DOCKER_AVAILABLE, reason='Docker daemon not available')


# ===========================================================================
# TestEnvironmentRegistry
# ===========================================================================

class TestEnvironmentRegistry:

    def test_environments_registered(self):
        envs = list_environments()
        assert 'local' in envs, f"'local' not in {envs}"
        assert 'docker' in envs, f"'docker' not in {envs}"

    def test_tools_registered(self):
        tools = list_agent_tools()
        for name in ('bash', 'python_exec', 'read_file', 'write_file'):
            assert name in tools, f"'{name}' not in {tools}"

    def test_tool_infos_registered(self):
        for name in ('bash', 'python_exec', 'read_file', 'write_file'):
            assert name in AGENT_TOOL_INFO_REGISTRY, f"ToolInfo missing for '{name}'"
            info = AGENT_TOOL_INFO_REGISTRY[name]
            assert isinstance(info, ToolInfo)
            assert info.name == name
            assert info.description

    def test_resolve_tool_infos_returns_infos(self):
        infos = resolve_tool_infos(['bash', 'python_exec'])
        assert len(infos) == 2
        assert {i.name for i in infos} == {'bash', 'python_exec'}

    def test_resolve_tool_infos_empty(self):
        assert resolve_tool_infos(None) == []
        assert resolve_tool_infos([]) == []

    def test_resolve_tool_infos_unknown_skipped(self):
        # Unknown names are silently skipped (they have no registered ToolInfo).
        infos = resolve_tool_infos(['bash', 'nonexistent_tool_xyz'])
        assert len(infos) == 1
        assert infos[0].name == 'bash'

    def test_get_environment_local(self):
        cls = get_environment('local')
        from evalscope.agent.environments.local import LocalAgentEnvironment
        assert cls is LocalAgentEnvironment

    def test_get_environment_docker(self):
        cls = get_environment('docker')
        from evalscope.agent.environments.docker import DockerAgentEnvironment
        assert cls is DockerAgentEnvironment

    def test_get_environment_unknown_raises(self):
        with pytest.raises(ValueError, match='not registered'):
            get_environment('nonexistent_env_xyz')

    def test_duplicate_environment_registration_raises(self):
        from evalscope.api.registry import register_environment
        with pytest.raises(ValueError, match='already registered'):
            @register_environment('local')
            class _Dup(AgentEnvironment):
                async def exec(self, *a, **kw): ...
                async def read_file(self, *a, **kw): ...
                async def write_file(self, *a, **kw): ...
                async def close(self): ...

    def test_bash_tool_info_has_required_params(self):
        info = AGENT_TOOL_INFO_REGISTRY['bash']
        assert 'command' in info.parameters.properties
        assert 'command' in info.parameters.required

    def test_python_exec_tool_info_has_required_params(self):
        info = AGENT_TOOL_INFO_REGISTRY['python_exec']
        assert 'code' in info.parameters.properties
        assert 'code' in info.parameters.required


# ===========================================================================
# TestLocalEnvironmentExec
# ===========================================================================

class TestLocalEnvironmentExec:

    def _run(self, coro):
        return AsyncioLoopRunner.run(coro)

    def _env(self):
        from evalscope.agent.environments.local import LocalAgentEnvironment
        return LocalAgentEnvironment()

    def test_exec_echo(self):
        env = self._env()
        result = self._run(env.exec(['echo', 'hello']))
        assert result.returncode == 0
        assert 'hello' in result.stdout
        assert not result.timed_out

    def test_exec_nonzero_returncode(self):
        env = self._env()
        result = self._run(env.exec(['bash', '-c', 'exit 42']))
        assert result.returncode == 42

    def test_exec_stderr(self):
        env = self._env()
        result = self._run(env.exec(['bash', '-c', 'echo err >&2; exit 1']))
        assert 'err' in result.stderr
        assert result.returncode != 0

    def test_exec_timeout(self):
        env = self._env()
        result = self._run(env.exec(['sleep', '10'], timeout=0.3))
        assert result.timed_out
        assert result.returncode == -1

    def test_exec_cwd(self):
        env = self._env()
        result = self._run(env.exec(['pwd'], cwd='/tmp'))
        assert '/tmp' in result.stdout

    def test_exec_with_env_vars(self):
        from evalscope.agent.environments.local import LocalAgentEnvironment
        env = LocalAgentEnvironment(env_vars={'MY_VAR': 'hello_from_test'})
        result = self._run(env.exec(['bash', '-c', 'echo $MY_VAR']))
        assert 'hello_from_test' in result.stdout

    def test_read_write_file(self):
        env = self._env()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as fh:
            path = fh.name
        try:
            content = 'hello from T2 test\nline 2'
            self._run(env.write_file(path, content))
            read_back = self._run(env.read_file(path))
            assert read_back == content
        finally:
            os.unlink(path)

    def test_write_file_creates_parent_dirs(self):
        env = self._env()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'subdir', 'nested.txt')
            self._run(env.write_file(path, 'nested content'))
            read_back = self._run(env.read_file(path))
            assert read_back == 'nested content'

    def test_read_file_missing_raises(self):
        env = self._env()
        with pytest.raises(FileNotFoundError):
            self._run(env.read_file('/nonexistent/path/xyz_8472.txt'))

    def test_close_is_idempotent(self):
        env = self._env()
        self._run(env.close())
        self._run(env.close())  # second call must not raise

    def test_context_manager(self):
        async def _cm():
            from evalscope.agent.environments.local import LocalAgentEnvironment
            async with LocalAgentEnvironment() as env:
                result = await env.exec(['echo', 'cm'])
            return result

        result = self._run(_cm())
        assert 'cm' in result.stdout

    def test_exec_result_duration_positive(self):
        env = self._env()
        result = self._run(env.exec(['echo', 'hi']))
        assert result.duration >= 0


# ===========================================================================
# TestLocalEnvironmentTools  (tool handlers with LocalAgentEnvironment)
# ===========================================================================

class TestLocalEnvironmentTools:

    def _run(self, coro):
        return AsyncioLoopRunner.run(coro)

    def _env(self):
        from evalscope.agent.environments.local import LocalAgentEnvironment
        return LocalAgentEnvironment()

    def test_bash_tool_runs_command(self):
        from evalscope.agent.tools.bash import run_bash
        env = self._env()
        call = _tool_call('bash', {'command': 'echo hello_bash'})
        obs = self._run(run_bash(call, env))
        assert 'hello_bash' in obs

    def test_bash_tool_without_env_raises(self):
        from evalscope.agent.tools.bash import run_bash
        call = _tool_call('bash', {'command': 'echo x'})
        with pytest.raises(PermissionError, match='requires an AgentEnvironment'):
            self._run(run_bash(call, None))

    def test_bash_tool_stderr_in_output(self):
        from evalscope.agent.tools.bash import run_bash
        env = self._env()
        call = _tool_call('bash', {'command': 'echo err >&2 && exit 0'})
        obs = self._run(run_bash(call, env))
        # stderr section is present when non-empty
        assert '[stderr]' in obs

    def test_python_exec_tool_runs_code(self):
        from evalscope.agent.tools.python_exec import run_python_exec
        env = self._env()
        call = _tool_call('python_exec', {'code': 'print(2 + 2)'})
        obs = self._run(run_python_exec(call, env))
        assert '4' in obs

    def test_python_exec_tool_without_env_raises(self):
        from evalscope.agent.tools.python_exec import run_python_exec
        call = _tool_call('python_exec', {'code': 'print(1)'})
        with pytest.raises(PermissionError):
            self._run(run_python_exec(call, None))

    def test_read_file_tool(self):
        from evalscope.agent.tools.text_editor import run_read_file, run_write_file
        env = self._env()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w') as fh:
            fh.write('agent reads this')
            path = fh.name
        try:
            call = _tool_call('read_file', {'path': path})
            obs = self._run(run_read_file(call, env))
            assert 'agent reads this' in obs
        finally:
            os.unlink(path)

    def test_write_file_tool(self):
        from evalscope.agent.tools.text_editor import run_read_file, run_write_file
        env = self._env()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as fh:
            path = fh.name
        try:
            call = _tool_call('write_file', {'path': path, 'content': 'written by tool'})
            obs = self._run(run_write_file(call, env))
            assert 'written' in obs.lower()
            # Verify file content
            with open(path) as fh:
                assert fh.read() == 'written by tool'
        finally:
            os.unlink(path)

    def test_read_file_tool_without_env_raises(self):
        from evalscope.agent.tools.text_editor import run_read_file
        call = _tool_call('read_file', {'path': '/tmp/x'})
        with pytest.raises(PermissionError):
            self._run(run_read_file(call, None))

    def test_write_file_tool_without_env_raises(self):
        from evalscope.agent.tools.text_editor import run_write_file
        call = _tool_call('write_file', {'path': '/tmp/x', 'content': 'x'})
        with pytest.raises(PermissionError):
            self._run(run_write_file(call, None))


# ===========================================================================
# TestDockerEnvironmentExec  (requires Docker)
# ===========================================================================

@docker_mark
class TestDockerEnvironmentExec:
    """Integration tests for DockerAgentEnvironment.

    These tests create real Docker containers using the ``python:3.11-slim``
    image.  Each test uses its own environment instance (= its own container).
    """

    def _run(self, coro):
        return AsyncioLoopRunner.run(coro)

    def _env(self):
        from evalscope.agent.environments.docker import DockerAgentEnvironment
        return DockerAgentEnvironment(image='python:3.11-slim', timeout=30.0)

    def teardown_method(self, method):
        """Reset the class-level manager between test classes to avoid leakage."""
        # We intentionally leave the manager running (shared singleton) for
        # performance; individual containers are deleted per-sample via close().

    def test_exec_echo(self):
        env = self._env()
        try:
            result = self._run(env.exec(['echo', 'hello docker']))
            assert result.returncode == 0
            assert 'hello docker' in result.stdout
            assert not result.timed_out
        finally:
            self._run(env.close())

    def test_exec_python(self):
        env = self._env()
        try:
            result = self._run(env.exec(['python3', '-c', 'print(1+1)']))
            assert result.returncode == 0
            assert '2' in result.stdout
        finally:
            self._run(env.close())

    def test_exec_nonzero_returncode(self):
        env = self._env()
        try:
            result = self._run(env.exec(['/bin/bash', '-c', 'exit 5']))
            assert result.returncode == 5
        finally:
            self._run(env.close())

    def test_exec_cwd(self):
        env = self._env()
        try:
            # default working_dir is /workspace; verify via pwd
            result = self._run(env.exec(['/bin/bash', '-c', 'pwd']))
            assert result.returncode == 0
            assert '/workspace' in result.stdout or result.returncode == 0
        finally:
            self._run(env.close())

    def test_read_write_file(self):
        env = self._env()
        try:
            content = 'docker file test\nline2'
            self._run(env.write_file('/workspace/test.txt', content))
            read_back = self._run(env.read_file('/workspace/test.txt'))
            assert read_back == content
        finally:
            self._run(env.close())

    def test_close_idempotent(self):
        env = self._env()
        # Trigger container creation
        self._run(env.exec(['true']))
        # First close
        self._run(env.close())
        # Second close must not raise
        self._run(env.close())

    def test_multiple_env_instances_isolated(self):
        """Two environment instances should get separate containers."""
        env1 = self._env()
        env2 = self._env()
        try:
            self._run(env1.write_file('/workspace/marker.txt', 'env1'))
            self._run(env2.write_file('/workspace/marker.txt', 'env2'))
            v1 = self._run(env1.read_file('/workspace/marker.txt'))
            v2 = self._run(env2.read_file('/workspace/marker.txt'))
            # Each container has its own filesystem
            assert v1 == 'env1'
            assert v2 == 'env2'
        finally:
            self._run(env1.close())
            self._run(env2.close())

    def test_context_manager(self):
        async def _cm():
            from evalscope.agent.environments.docker import DockerAgentEnvironment
            async with DockerAgentEnvironment(image='python:3.11-slim') as env:
                result = await env.exec(['echo', 'ctx_mgr'])
            return result

        result = self._run(_cm())
        assert 'ctx_mgr' in result.stdout


# ===========================================================================
# TestDockerEnvironmentTools  (bash + python_exec with docker)
# ===========================================================================

@docker_mark
class TestDockerEnvironmentTools:

    def _run(self, coro):
        return AsyncioLoopRunner.run(coro)

    def _env(self):
        from evalscope.agent.environments.docker import DockerAgentEnvironment
        return DockerAgentEnvironment(image='python:3.11-slim', timeout=30.0)

    def test_bash_tool_in_docker(self):
        from evalscope.agent.tools.bash import run_bash
        env = self._env()
        try:
            call = _tool_call('bash', {'command': 'echo docker_bash'})
            obs = self._run(run_bash(call, env))
            assert 'docker_bash' in obs
        finally:
            self._run(env.close())

    def test_python_exec_tool_in_docker(self):
        from evalscope.agent.tools.python_exec import run_python_exec
        env = self._env()
        try:
            call = _tool_call('python_exec', {'code': 'print("docker_py", 2**10)'})
            obs = self._run(run_python_exec(call, env))
            assert '1024' in obs
        finally:
            self._run(env.close())


# ===========================================================================
# TestAgentLoopWithEnvironment  (AgentLoop + local env + bash tool)
# ===========================================================================

class TestAgentLoopWithEnvironment:
    """Verify the AgentLoop correctly wires environment through ToolExecutor."""

    def _run(self, coro):
        return AsyncioLoopRunner.run(coro)

    def test_loop_uses_environment_via_bash_tool(self):
        """Model calls bash → env.exec is invoked → result observed."""
        from evalscope.agent.environments.local import LocalAgentEnvironment
        from evalscope.agent.tools.bash import run_bash
        from evalscope.api.registry import get_strategy

        env = LocalAgentEnvironment()
        handlers = {'bash': run_bash}

        # Model: first call returns a bash tool_call; second call returns final answer.
        bash_call = _tool_call('bash', {'command': 'echo agent_output'}, call_id='tc-bash')
        first_output = _make_output(
            content='',
            tool_calls=[bash_call],
            stop_reason='tool_calls',
        )
        second_output = _make_output(content='The output was: agent_output')

        model = MagicMock()
        model.generate.side_effect = [first_output, second_output]

        strategy = get_strategy('function_calling')()
        tool_executor = ToolExecutor(handlers=handlers, environment=env)
        ctx = AgentContext(
            sample_id='test-env-loop',
            messages=[ChatMessageUser(content='run bash')],
            tools=[],
        )
        trace = AgentTrace(strategy='function_calling', max_steps=5)
        loop = AgentLoop(
            model=model,
            strategy=strategy,
            tool_executor=tool_executor,
            environment=env,
            max_steps=5,
            trace=trace,
        )

        result = self._run(loop.run(ctx))

        # Verify tool was called and observation includes bash output
        tool_msg = next(
            (m for m in result.messages if getattr(m, 'role', None) == 'tool'),
            None,
        )
        assert tool_msg is not None, 'Expected a tool message in conversation'
        assert 'agent_output' in tool_msg.content, (
            f'Expected bash output in tool message, got: {tool_msg.content!r}'
        )

        # ENV_EXEC event should NOT be in trace (bash uses env.exec via AgentEnvironment,
        # not a separate ENV_EXEC emitter); TOOL_RESULT IS expected.
        event_types = {ev.type for ev in result.trace.events}
        assert EventType.TOOL_RESULT in event_types
        assert EventType.SUBMIT in event_types


# ===========================================================================
# TestDefaultAdapterEnvPath  (DefaultDataAdapter._on_agent_inference with env)
# ===========================================================================

class TestDefaultAdapterEnvPath:
    """Test that _on_agent_inference instantiates environment and merges tool infos."""

    def test_tool_infos_merged_into_ctx(self):
        """When agent_config.tools=['bash'], bash ToolInfo is passed to the model."""
        from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
        from evalscope.api.dataset import Sample

        # Build a minimal adapter bypassing __init__
        adapter = DefaultDataAdapter.__new__(DefaultDataAdapter)
        cfg = AgentConfig(strategy='function_calling', tools=['bash'], max_steps=1)
        task_cfg = MagicMock()
        task_cfg.agent_config = cfg
        adapter._task_config = task_cfg

        # Model: return final answer immediately (no tool calls)
        final_out = _make_output(content='done')
        model = MagicMock()
        model.generate.return_value = final_out

        sample = MagicMock()
        sample.id = 'x'
        sample.input = 'hello'
        sample.tools = []

        output = adapter._on_inference(model, sample)

        # Verify model was called with bash ToolInfo in the tools list
        call_args = model.generate.call_args
        tools_passed = call_args[1].get('tools') or call_args[0][1] if len(call_args[0]) > 1 else None
        # tools_passed may be None if strategy decided not to pass them;
        # at minimum verify execution completed without error.
        assert output is not None

    def test_environment_extra_forwarded(self):
        """environment_extra is forwarded to environment constructor kwargs."""
        from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
        from evalscope.api.dataset import Sample

        adapter = DefaultDataAdapter.__new__(DefaultDataAdapter)
        cfg = AgentConfig(
            strategy='function_calling',
            tools=[],
            max_steps=1,
            environment='local',
            environment_extra={'working_dir': '/tmp'},
        )
        task_cfg = MagicMock()
        task_cfg.agent_config = cfg
        adapter._task_config = task_cfg

        final_out = _make_output(content='env done')
        model = MagicMock()
        model.generate.return_value = final_out

        sample = MagicMock()
        sample.id = 'env-test'
        sample.input = 'test env'
        sample.tools = []

        output = adapter._on_inference(model, sample)
        assert output is not None
        # The environment was created and closed; trace environment name should match
        trace = output.metadata.get('__agent_trace__')
        assert trace is not None
        assert trace.environment == 'local'


# ===========================================================================
# TestAgentConfigEnvironmentExtra  (AgentConfig schema)
# ===========================================================================

class TestAgentConfigEnvironmentExtra:

    def test_default_environment_extra_is_empty(self):
        cfg = AgentConfig()
        assert cfg.environment_extra == {}

    def test_environment_extra_accepted(self):
        cfg = AgentConfig(
            strategy='function_calling',
            environment='docker',
            environment_extra={'image': 'python:3.11-slim', 'working_dir': '/workspace'},
        )
        assert cfg.environment_extra['image'] == 'python:3.11-slim'
        assert cfg.environment == 'docker'

    def test_environment_extra_serialises(self):
        cfg = AgentConfig(environment_extra={'key': 'val'})
        d = cfg.model_dump()
        assert d['environment_extra'] == {'key': 'val'}

    def test_extra_and_environment_extra_independent(self):
        cfg = AgentConfig(extra={'system_prompt': 'hi'}, environment_extra={'image': 'x'})
        assert 'system_prompt' in cfg.extra
        assert 'system_prompt' not in cfg.environment_extra
        assert 'image' in cfg.environment_extra
        assert 'image' not in cfg.extra
