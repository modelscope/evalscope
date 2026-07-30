"""Tests for the OpenEnv backend and its separated service runtimes."""

import asyncio
import pytest
import sys
from types import ModuleType, SimpleNamespace

from evalscope.api.environment import EnvironmentStepResult
from evalscope.api.registry import get_environment_runtime, get_task_environment
from evalscope.environment.backends.openenv import OpenEnvBackend, OpenEnvSession
from evalscope.environment.runtimes.ms_enclave_docker import MsEnclaveDockerRuntime


class FakeGenericEnvClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.calls = []
        self.instances.append(self)

    async def connect(self):
        self.calls.append(('connect', None))

    async def reset(self, **kwargs):
        self.calls.append(('reset', kwargs))
        return SimpleNamespace(observation={'goal': 'test'}, reward=0.0, done=False, metadata={'phase': 'reset'})

    async def step(self, action):
        self.calls.append(('step', action))
        return SimpleNamespace(observation={'text': 'done'}, reward=1.0, done=True, metadata=None)

    async def state(self):
        return {'benchmark': 'miniwob'}

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_openenv(monkeypatch):
    FakeGenericEnvClient.instances.clear()
    openenv = ModuleType('openenv')
    core = ModuleType('openenv.core')
    core.GenericEnvClient = FakeGenericEnvClient
    monkeypatch.setitem(sys.modules, 'openenv', openenv)
    monkeypatch.setitem(sys.modules, 'openenv.core', core)
    monkeypatch.setattr('evalscope.environment.backends.openenv.check_import', lambda *args, **kwargs: True)


def test_openenv_backend_registry_and_validation():
    assert get_task_environment('openenv') is OpenEnvBackend
    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
        EnvironmentStepResult(unexpected=True)
    with pytest.raises(ValueError, match='Unsupported OpenEnv'):
        OpenEnvBackend().create_session(
            base_url='http://localhost:8000',
            config={'provider': 'forbidden'},
        )
    with pytest.raises(ValueError, match='greater than zero'):
        OpenEnvSession(base_url='http://localhost:8000', connect_timeout_s=0)


def test_openenv_session_lifecycle_and_normalization(fake_openenv):
    session = OpenEnvBackend().create_session(
        base_url='http://localhost:8000/',
        config={},
    )

    reset = asyncio.run(session.reset(seed=7))
    step = asyncio.run(session.step({'action_str': 'click("1")'}))
    state = asyncio.run(session.state())
    asyncio.run(session.close())

    client = FakeGenericEnvClient.instances[0]
    assert isinstance(reset, EnvironmentStepResult)
    assert client.kwargs['base_url'] == 'http://localhost:8000'
    assert client.kwargs['mode'] == 'simulation'
    assert 'provider' not in client.kwargs
    assert reset.observation == {'goal': 'test'}
    assert reset.metadata == {'phase': 'reset'}
    assert step.reward == 1.0
    assert step.done is True
    assert state == {'benchmark': 'miniwob'}
    assert client.calls == [
        ('connect', None),
        ('reset', {'seed': 7}),
        ('step', {'action_str': 'click("1")'}),
    ]
    assert client.closed is True


def test_openenv_session_never_calls_from_env(fake_openenv, monkeypatch):
    from_env = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('from_env must not be called'))
    monkeypatch.setattr(FakeGenericEnvClient, 'from_env', from_env, raising=False)
    session = OpenEnvSession(base_url='http://localhost:8000')
    asyncio.run(session.reset())
    asyncio.run(session.close())


def test_ms_enclave_runtime_is_docker_only(monkeypatch):
    calls = []

    class FakeDockerSandboxConfig:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeHandle:

        async def get_info(self):
            return SimpleNamespace(metadata={'container_id': 'container-1'})

        async def close(self):
            calls.append(('close', None))

    class FakeService:

        async def create_sandbox(self, engine, config, manager_config=None):
            calls.append(('create', engine, config, manager_config))
            return FakeHandle()

    package = ModuleType('ms_enclave')
    sandbox = ModuleType('ms_enclave.sandbox')
    model = ModuleType('ms_enclave.sandbox.model')
    model.DockerSandboxConfig = FakeDockerSandboxConfig
    monkeypatch.setitem(sys.modules, 'ms_enclave', package)
    monkeypatch.setitem(sys.modules, 'ms_enclave.sandbox', sandbox)
    monkeypatch.setitem(sys.modules, 'ms_enclave.sandbox.model', model)
    monkeypatch.setattr('evalscope.environment.runtimes.ms_enclave_docker.check_import', lambda *args, **kwargs: True)
    monkeypatch.setattr('evalscope.environment.runtimes.ms_enclave_docker.get_sandbox_service', lambda: FakeService())
    monkeypatch.setattr(MsEnclaveDockerRuntime, '_reserve_port', staticmethod(lambda host: 18123))
    monkeypatch.setattr(MsEnclaveDockerRuntime, '_wait_for_ready', staticmethod(lambda base_url, timeout_s: None))

    runtime = MsEnclaveDockerRuntime()
    handle = asyncio.run(
        runtime.start(
            image='browsergym:latest',
            env_vars={'BROWSERGYM_BENCHMARK': 'miniwob'},
            config={},
        )
    )

    assert get_environment_runtime('ms_enclave_docker') is MsEnclaveDockerRuntime
    assert handle.base_url == 'http://127.0.0.1:18123'
    _, engine, sandbox_config, manager_config = calls[0]
    assert engine.value == 'docker'
    assert sandbox_config.kwargs['ports'] == {'8000/tcp': ('127.0.0.1', 18123)}
    assert sandbox_config.kwargs['tools_config'] == {}
    assert manager_config == {}
    asyncio.run(handle.close())
    assert calls[-1] == ('close', None)
