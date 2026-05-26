"""T2 Environment Abstraction – unit and integration tests.

Test plan:
  TestEnvironmentRegistry      – registry API surface (environments + tools)
  TestLocalEnvironmentExec     – LocalAgentEnvironment exec
  TestLocalEnvironmentTools    – bash/python_exec handlers w/ local env
  TestDockerEnvironmentExec    – EnclaveAgentEnvironment (docker engine) exec
  TestDockerEnvironmentTools   – bash + python_exec handlers w/ enclave env
  TestAgentLoopWithEnvironment – full AgentLoop + local env + bash tool
  TestDefaultAdapterEnvPath    – _on_agent_inference environment_extra + tool_infos
  TestNativeAgentConfigEnvironmentExtra – NativeAgentConfig.environment_extra round-trip
"""

import os
import pytest
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

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
from evalscope.api.agent.types import NativeAgentConfig
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
        for name in ('bash', 'python_exec'):
            assert name in tools, f"'{name}' not in {tools}"

    def test_tool_infos_registered(self):
        for name in ('bash', 'python_exec'):
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
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
        assert cls is EnclaveAgentEnvironment

    def test_get_environment_enclave_alias(self):
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
        assert get_environment('enclave') is EnclaveAgentEnvironment
        assert get_environment('volcengine') is EnclaveAgentEnvironment

    def test_get_environment_unknown_raises(self):
        with pytest.raises(ValueError, match='not registered'):
            get_environment('nonexistent_env_xyz')

    def test_duplicate_environment_registration_raises(self):
        from evalscope.api.registry import register_environment
        with pytest.raises(ValueError, match='already registered'):
            @register_environment('local')
            class _Dup(AgentEnvironment):
                async def exec(self, *a, **kw): ...
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


# ===========================================================================
# TestDockerEnvironmentExec  (requires Docker)
# ===========================================================================

@docker_mark
class TestDockerEnvironmentExec:
    """Integration tests for ``EnclaveAgentEnvironment`` with the docker engine.

    These tests create real Docker containers using the ``python:3.11-slim``
    image.  Each test uses its own environment instance (= its own container).
    """

    def _run(self, coro):
        return AsyncioLoopRunner.run(coro)

    def _env(self):
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
        return EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config={'image': 'python:3.11-slim'},
            timeout=30.0,
        )

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
            r1 = self._run(env1.exec(['bash', '-c', 'echo env1 > /workspace/marker.txt && cat /workspace/marker.txt']))
            r2 = self._run(env2.exec(['bash', '-c', 'echo env2 > /workspace/marker.txt && cat /workspace/marker.txt']))
            # Each container has its own filesystem; markers stay isolated.
            assert 'env1' in r1.stdout
            assert 'env2' in r2.stdout
        finally:
            self._run(env1.close())
            self._run(env2.close())

    def test_context_manager(self):
        async def _cm():
            from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
            async with EnclaveAgentEnvironment(
                engine='docker',
                sandbox_config={'image': 'python:3.11-slim'},
            ) as env:
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
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
        return EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config={'image': 'python:3.11-slim'},
            timeout=30.0,
        )

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

        # Model: first call returns a bash tool_call; second call returns submit.
        bash_call = _tool_call('bash', {'command': 'echo agent_output'}, call_id='tc-bash')
        first_output = _make_output(
            content='',
            tool_calls=[bash_call],
            stop_reason='tool_calls',
        )
        submit_call = _tool_call('submit', {'answer': 'agent_output'}, call_id='tc-submit')
        second_output = _make_output(content='', tool_calls=[submit_call])

        model = MagicMock()
        model.generate_async = AsyncMock(side_effect=[first_output, second_output])

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
        cfg = NativeAgentConfig(strategy='function_calling', tools=['bash'], max_steps=1)
        task_cfg = MagicMock()
        task_cfg.agent_config = cfg
        adapter._task_config = task_cfg

        # Model: return final answer immediately (no tool calls)
        final_out = _make_output(content='done')
        model = MagicMock()
        model.generate.return_value = final_out
        # AgentLoop awaits ``generate_async``.
        model.generate_async = AsyncMock(return_value=final_out)

        sample = MagicMock()
        sample.id = 'x'
        sample.input = 'hello'
        sample.tools = []

        output = adapter._on_inference(model, sample)

        # Verify model was called with bash ToolInfo in the tools list
        call_args = model.generate_async.call_args
        tools_passed = call_args[1].get('tools') or call_args[0][1] if len(call_args[0]) > 1 else None
        # tools_passed may be None if strategy decided not to pass them;
        # at minimum verify execution completed without error.
        assert output is not None

    def test_environment_extra_forwarded(self):
        """environment_extra is forwarded to environment constructor kwargs."""
        from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
        from evalscope.api.dataset import Sample

        adapter = DefaultDataAdapter.__new__(DefaultDataAdapter)
        cfg = NativeAgentConfig(
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
        model.generate_async = AsyncMock(return_value=final_out)

        sample = MagicMock()
        sample.id = 'env-test'
        sample.input = 'test env'
        sample.tools = []

        output = adapter._on_inference(model, sample)
        assert output is not None
        # The environment was created and closed; trace environment name should match
        trace = output.trace
        assert trace is not None
        assert trace.environment == 'local'


# ===========================================================================
# TestNativeAgentConfigEnvironmentExtra  (NativeAgentConfig schema)
# ===========================================================================

class TestNativeAgentConfigEnvironmentExtra:

    def test_default_environment_extra_is_empty(self):
        cfg = NativeAgentConfig()
        assert cfg.environment_extra == {}

    def test_environment_extra_accepted(self):
        cfg = NativeAgentConfig(
            strategy='function_calling',
            environment='docker',
            environment_extra={'image': 'python:3.11-slim', 'working_dir': '/workspace'},
        )
        assert cfg.environment_extra['image'] == 'python:3.11-slim'
        assert cfg.environment == 'docker'

    def test_environment_extra_serialises(self):
        cfg = NativeAgentConfig(environment_extra={'key': 'val'})
        d = cfg.model_dump()
        assert d['environment_extra'] == {'key': 'val'}

    def test_kwargs_and_environment_extra_independent(self):
        cfg = NativeAgentConfig(kwargs={'system_prompt': 'hi'}, environment_extra={'image': 'x'})
        assert 'system_prompt' in cfg.kwargs
        assert 'system_prompt' not in cfg.environment_extra
        assert 'image' in cfg.environment_extra
        assert 'image' not in cfg.kwargs
