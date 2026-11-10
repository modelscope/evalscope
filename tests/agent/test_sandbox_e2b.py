"""Local E2B configuration, preparation and output contracts."""

import asyncio
import csv
import gzip
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalscope.api.sandbox import (
    SandboxConfigError,
    SandboxEngine,
    build_sandbox_config,
    get_enclave_types,
    require_docker_engine,
    resolve_engine,
    resolve_task_engine,
    task_sandbox_env_kwargs,
)
from evalscope.api.sandbox.compat import adapt_sandbox_config, memory_limit_to_mib, validate_e2b_fields
from evalscope.api.sandbox.e2b import load_e2b_image_mapping, with_e2b_image_mapping
from evalscope.config import TaskConfig
from evalscope.utils.argument_utils import get_secret_value

ROOT = Path(__file__).resolve().parents[2]


def task_stub(engine: str, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(sandbox=SimpleNamespace(enabled=enabled, engine=engine, manager_config={'api_key': 'placeholder'}))


# Feature: e2b-sandbox, Property 12: Shared engine resolution follows the enabled/image-engine decision table.
@settings(max_examples=100)
@given(st.sampled_from(['docker', 'e2b', 'volcengine', 'volcano', 'volc', 'unknown']), st.booleans(),
       st.booleans(), st.booleans())
def test_engine_table(engine: str, enabled: bool, uppercase: bool, absent: bool) -> None:
    spelling = f' {engine.upper() if uppercase else engine} '
    task = None if absent else task_stub(spelling, enabled)
    invalid = not absent and ((enabled and engine not in ['docker', 'e2b']) or (not enabled and engine == 'e2b'))
    if invalid:
        with pytest.raises(SandboxConfigError):
            resolve_task_engine(task)
        return
    expected = SandboxEngine.E2B if not absent and enabled and engine == 'e2b' else SandboxEngine.DOCKER
    assert resolve_task_engine(task) == expected
    if expected == SandboxEngine.E2B:
        with pytest.raises(SandboxConfigError, match='does not support the e2b sandbox engine'):
            require_docker_engine(task)
    else:
        require_docker_engine(task)
    kwargs = task_sandbox_env_kwargs(task)
    assert kwargs['engine'] == expected
    assert ('manager_config' in kwargs) == (expected == SandboxEngine.E2B)
    if expected == SandboxEngine.E2B:
        assert kwargs['manager_config'] == task.sandbox.manager_config
        assert kwargs['manager_config'] is not task.sandbox.manager_config


def test_engine_types() -> None:
    from ms_enclave.sandbox.model import E2BSandboxConfig, SandboxType
    assert resolve_engine(' E2B ') == SandboxEngine.E2B
    assert get_enclave_types(SandboxEngine.E2B)[:2] == (SandboxType.E2B, E2BSandboxConfig)
    assert resolve_task_engine(SimpleNamespace(sandbox=None)) == SandboxEngine.DOCKER
    for alias in ['volcano', 'volc', 'volcengine']:
        assert resolve_engine(alias) == SandboxEngine.VOLCENGINE
    from evalscope.api.mixin.code_execution_sandbox_mixin import EnclaveCodeExecutionBackend
    mixin = EnclaveCodeExecutionBackend.__new__(EnclaveCodeExecutionBackend)
    mixin._task_config = task_stub('volcengine')
    assert mixin._resolve_engine() == SandboxEngine.VOLCENGINE


# Feature: e2b-sandbox, Property 13: Field adaptation matches an independent reference conversion.
@settings(max_examples=100)
@given(st.integers(0, 10000), st.floats(min_value=0.1, max_value=64, allow_nan=False), st.booleans(),
       st.one_of(st.none(), st.integers(1, 8000)), st.one_of(st.none(), st.floats(0, 8000)), st.booleans())
def test_field_rules(memory: int, cpu: float, internet: bool, explicit: int | None,
                     duration: float | None, reject: bool) -> None:
    config = dict(image='org/image', memory_limit=f'{memory}m', cpu_limit=cpu, network_enabled=internet,
                  command='sleep infinity', platform='linux/amd64', cap_add=['SYS_ADMIN'], docker_executor_workers=8,
                  volumes={'/host': {'bind': '/target'}} if reject else {}, privileged=False, network='bridge')
    if explicit is not None:
        config.update(memory_mb=explicit, cpu_count=explicit, sandbox_timeout=explicit, user='custom')
    assert adapt_sandbox_config(SandboxEngine.DOCKER, config, expected_duration=duration) == config
    if reject:
        with pytest.raises(SandboxConfigError, match='volumes'):
            adapt_sandbox_config(SandboxEngine.E2B, config)
        return
    result = adapt_sandbox_config(SandboxEngine.E2B, config, expected_duration=duration)
    expected = dict(image='org/image', memory_mb=explicit or max(1, memory), cpu_count=explicit or math.ceil(cpu),
                    allow_internet_access=internet, user='custom' if explicit else 'root')
    if duration is not None:
        expected['sandbox_timeout'] = min(math.ceil(duration) + 600, explicit) if explicit else math.ceil(duration) + 600
    elif explicit:
        expected['sandbox_timeout'] = explicit
    assert result == expected


# Feature: e2b-sandbox, Property 14: Docker memory sizes round upward to whole MiB.
@settings(max_examples=100)
@given(st.integers(0, 2**60), st.sampled_from(['', 'k', 'K', 'm', 'M', 'g', 'G']))
def test_memory_units(value: int, unit: str) -> None:
    factor = {'': 1, 'k': 1024, 'm': 1024**2, 'g': 1024**3}[unit.lower()]
    assert memory_limit_to_mib(f'{value}{unit}') == max(1, (value * factor + 2**20 - 1) // 2**20)


@pytest.mark.parametrize('value', ['1.5g', '-1m', '1t', 'abc', '', '1 mb'])
def test_invalid_memory(value: str) -> None:
    with pytest.raises(SandboxConfigError, match='memory_limit'):
        memory_limit_to_mib(value)


def test_field_errors_and_defaults() -> None:
    with pytest.raises(SandboxConfigError, match='entrypoint.*privileged'):
        build_sandbox_config(SandboxEngine.E2B, dict(image='image', privileged=True, entrypoint=['sh']))
    with pytest.raises(SandboxConfigError, match='image.*template'):
        build_sandbox_config(SandboxEngine.E2B, {})
    with pytest.raises(SandboxConfigError, match='unknown'):
        build_sandbox_config(SandboxEngine.E2B, dict(image='image', unknown=True))
    assert build_sandbox_config(SandboxEngine.E2B, dict(image='image', privileged=False)).user == 'root'
    assert build_sandbox_config(SandboxEngine.E2B, dict(image='image', user='user')).user == 'user'
    validate_e2b_fields({})
    config = dict(image='image', command='ignored', platform='ignored', pull_progress=True,
                  pull_progress_interval=1, remove_on_exit=True, docker_executor_workers=8, cap_add=['SYS_ADMIN'])
    assert adapt_sandbox_config(SandboxEngine.E2B, config) == {'image': 'image', 'user': 'root'}


def mapping_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location('generate_e2b_mapping', ROOT / 'scripts/generate_e2b_image_mapping.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Feature: e2b-sandbox, Property 20: Mapping generation round-trips and is byte-reproducible.
@settings(max_examples=100)
@given(st.dictionaries(st.from_regex('[a-z]{1,10}', fullmatch=True), st.text(alphabet='abc/:-0123', min_size=1), max_size=15),
       st.booleans())
def test_mapping_roundtrip(mapping: dict[str, str], bom: bool) -> None:
    module = mapping_script()
    rows = [{'源镜像': k, '转换后镜像': v, '数据集': 'SWE-bench Verified', 'extra': 'ignored'} for k, v in mapping.items()]
    rows += [{'源镜像': 'empty', '转换后镜像': '', '数据集': 'SWE-bench Verified'},
             {'源镜像': 'excluded', '转换后镜像': 'image', '数据集': 'Other'}]
    with tempfile.TemporaryDirectory() as root:
        csv_path, out = Path(root) / 'input.csv', Path(root) / 'mapping.gz'
        with csv_path.open('w', encoding='utf-8-sig' if bom else 'utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['源镜像', '转换后镜像', '数据集', 'extra'])
            writer.writeheader()
            writer.writerows(rows)
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            result = module.build_mapping(csv.DictReader(stream), module.DEFAULT_DATASETS)
        assert result == dict(sorted(mapping.items()))
        module.write_mapping(out, result)
        first = out.read_bytes()
        module.write_mapping(out, result)
        assert out.read_bytes() == first
        with gzip.open(out, 'rt') as stream:
            assert json.load(stream) == mapping
        if mapping:
            key = next(iter(mapping))
            with pytest.raises(ValueError, match=key):
                module.build_mapping(rows + [{'源镜像': key, '转换后镜像': mapping[key] + 'x', '数据集': 'SWE-bench Verified'}], module.DEFAULT_DATASETS)


def test_mapping_cli_errors(tmp_path: Path) -> None:
    path = tmp_path / 'input.csv'
    path.write_text('wrong\nvalue\n')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_e2b_image_mapping.py'), '--csv', str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 1 and str(path) in result.stderr and '源镜像' in result.stderr


def test_bundled_mapping_injection() -> None:
    load_e2b_image_mapping.cache_clear()
    original = {'api_key': 'placeholder'}
    first, second = with_e2b_image_mapping(original), with_e2b_image_mapping(original)
    assert len(first['image_mapping']) == 3518
    assert first['image_mapping'] is second['image_mapping']
    assert load_e2b_image_mapping.cache_info().misses == 1
    assert original == {'api_key': 'placeholder'}


# Feature: e2b-sandbox, Property 15: Serialized sandbox settings never contain raw API keys.
@settings(max_examples=100)
@given(st.text(alphabet='abcdef0123456789', min_size=12, max_size=32), st.integers(0, 4))
def test_task_secret_serialization(secret: str, depth: int) -> None:
    payload = {'api_key': secret}
    for _ in range(depth):
        payload = {'nested': [payload]}
    task = TaskConfig(model='mock', eval_type='mock_llm', datasets=[], sandbox={
        'manager_config': payload, 'default_config': payload}, sandbox_manager_config=payload)
    assert secret not in json.dumps(task.to_dict())
    assert secret not in str(task)
    assert secret not in repr(task)
    assert secret not in repr(task.sandbox)
    assert get_secret_value(task.sandbox.manager_config) == payload
    assert get_secret_value(task._to_update_dict()['sandbox_manager_config']) == payload
    from evalscope.evaluation_versioning import _fingerprint_task_mapping, _scrub_secrets
    raw = task.to_dict()
    raw['sandbox'] = task.sandbox.model_dump(mode='json')
    raw['sandbox_manager_config'] = payload
    assert _scrub_secrets(_fingerprint_task_mapping(task.to_dict())) == _scrub_secrets(_fingerprint_task_mapping(raw))
    with tempfile.TemporaryDirectory() as root:
        task.dump_yaml(root, strict=True)
        assert all(secret not in path.read_text() for path in Path(root).rglob('*.yaml'))


@pytest.mark.parametrize('sandbox,error', [
    ({'enabled': False}, 'enabled=True'),
    ({'manager_config': {'unknown': 1}}, 'unknown'),
    ({'manager_config': {'build_mode': 'micro'}}, 'micro'),
    ({'default_config': {'unknown': 1}}, 'unknown'),
    ({'default_config': {'volumes': {'host': {}}}}, 'volumes'),
])
def test_task_validation(sandbox: dict, error: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('E2B_API_KEY', 'environment-placeholder')
    with pytest.raises(ValueError, match=error):
        TaskConfig(model='mock', eval_type='mock_llm', sandbox={'enabled': True, 'engine': 'e2b', **sandbox})


def test_task_missing_key_and_read_only_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('E2B_API_KEY', raising=False)
    with pytest.raises(ValueError, match='E2B_API_KEY'):
        TaskConfig(model='mock', eval_type='mock_llm', sandbox={'enabled': True, 'engine': 'e2b'})
    monkeypatch.setenv('E2B_API_KEY', 'environment-placeholder')
    task = TaskConfig(model='mock', eval_type='mock_llm', sandbox={'enabled': True, 'engine': 'e2b'})
    assert task.sandbox.manager_config == {}


def test_missing_e2b_types_and_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.sandbox import e2b as e2b_module

    monkeypatch.setenv('E2B_API_KEY', 'placeholder')
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'e2b', None)  # `import e2b` fails as if the SDK were not installed
        with pytest.raises(ImportError, match=r"pip install 'ms-enclave\[e2b\]'"):
            TaskConfig(model='mock', eval_type='mock_llm', sandbox={'enabled': True, 'engine': 'e2b'})
    monkeypatch.setattr(e2b_module, 'get_enclave_types', MagicMock(side_effect=ImportError('upgrade ms-enclave[e2b]')))
    with pytest.raises(ImportError, match='upgrade'):
        TaskConfig(model='mock', eval_type='mock_llm', sandbox={'enabled': True, 'engine': 'e2b'})


def test_external_agent_rejected_before_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('E2B_API_KEY', 'placeholder')
    with pytest.raises(ValueError, match='ExternalAgentConfig'):
        TaskConfig(model='mock', eval_type='mock_llm', sandbox={'enabled': True, 'engine': 'e2b'},
                   agent_config={'mode': 'external', 'framework': 'mock'})


@pytest.mark.parametrize('engine', ['docker', 'volcengine'])
def test_non_e2b_task_does_not_import_backend(engine: str) -> None:
    script = f"""
import sys
from evalscope.config import TaskConfig
TaskConfig(model='mock', eval_type='mock_llm', sandbox={{'enabled': True, 'engine': {engine!r}}})
assert 'ms_enclave.sandbox.boxes.e2b_connection' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True)


# Feature: e2b-sandbox, Property 16: Preparation, execution and uploads share a single sandbox handle.
@settings(max_examples=100)
@given(st.lists(st.sampled_from(['prepare', 'exec', 'upload']), min_size=1, max_size=10))
def test_prepare_idempotent(operations: list[str]) -> None:
    async def run() -> None:
        from ms_enclave.sandbox.model import ExecutionStatus

        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
        with pytest.MonkeyPatch.context() as patch:
            handle = SimpleNamespace(sandbox_id='fake', close=AsyncMock(), put_dir=AsyncMock(return_value=True),
                execute_tool=AsyncMock(return_value=SimpleNamespace(status=ExecutionStatus.SUCCESS, output='ok', error='', execution_time=0)))
            service = SimpleNamespace(create_sandbox=AsyncMock(return_value=handle))
            patch.setattr('evalscope.agent.environments.enclave.get_sandbox_service', lambda: service)
            env = EnclaveAgentEnvironment()
            for op in operations:
                if op == 'prepare':
                    await env.prepare()
                elif op == 'exec':
                    await env.exec(['echo', 'hi'])
                else:
                    await env.put_dir('/source', '/target')
            assert service.create_sandbox.call_count == 1
            await env.close()
            await env.close()
            handle.close.assert_awaited_once()
    asyncio.run(run())


# Preparation errors propagate before model calls; the evaluator owns ignore_errors.
@settings(max_examples=100)
@given(st.text(min_size=1, max_size=60), st.sampled_from([RuntimeError, TimeoutError, ConnectionError, SandboxConfigError]))
def test_prepare_failure_propagates(reason: str, error_type: type[Exception]) -> None:
    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.api.benchmark.adapters import AgentLoopAdapter
    from evalscope.api.dataset import Sample

    adapter = AgentLoopAdapter(
        benchmark_meta=BenchmarkMeta(name='failure-test', dataset_id='fake', metric_list=['accuracy']),
        task_config=TaskConfig(model='mock', eval_type='mock_llm'),
    )
    error = error_type(reason)
    environment = SimpleNamespace(prepare=AsyncMock(side_effect=error), close=AsyncMock())
    adapter.build_environment = lambda sample: environment
    adapter._resolve_strategy = lambda sample, ac: None
    adapter._resolve_tools = lambda sample, ac: ({}, [])
    model = SimpleNamespace(name='mock', generate=MagicMock(side_effect=AssertionError('model must not run')))
    sample = Sample(id=0, input='task', target='target')
    with pytest.raises(error_type) as caught:
        adapter._on_inference(model, sample)
    assert caught.value.args == error.args
    environment.close.assert_awaited_once()
    model.generate.assert_not_called()


@pytest.mark.parametrize('owned', [False, True])
@pytest.mark.parametrize('cleanup_fails', [False, True])
@pytest.mark.parametrize('error_type', [RuntimeError, SandboxConfigError])
def test_prepare_failure_preserves_error_and_ownership(
    owned: bool, cleanup_fails: bool, error_type: type[Exception]
) -> None:
    from evalscope.api.agent import prepare_environment

    error = error_type('preparation failed')
    env = SimpleNamespace(prepare=AsyncMock(side_effect=error),
                          close=AsyncMock(side_effect=RuntimeError('cleanup failed') if cleanup_fails else None))
    with pytest.raises(error_type) as caught:
        prepare_environment(env, sample_id=0, close_on_failure=owned)
    assert caught.value is error
    assert env.close.await_count == int(owned)


def test_environment_defaults() -> None:
    from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
    env = EnclaveAgentEnvironment(engine='e2b', sandbox_config={'image': 'image'},
                                  interpreter=('bash', '-lc'), expected_duration=1800)
    assert env._interpreter == ['bash', '-c']
    assert env._sandbox_config.user == 'root'
    assert set(env._sandbox_config.tools_config) == {'shell_executor', 'python_executor'}
    assert env._sandbox_config.sandbox_timeout == 2400
    for engine in ['docker', 'e2b']:
        env = EnclaveAgentEnvironment(engine=engine, sandbox_config={'image': 'image'}, interpreter=('sh', '-c'))
        assert env._interpreter == ['sh', '-c']
    env = EnclaveAgentEnvironment(interpreter=('bash', '-lc'))
    assert env._interpreter == ['bash', '-lc']
    with pytest.raises(SandboxConfigError, match='volumes'):
        EnclaveAgentEnvironment(engine='e2b', sandbox_config={'image': 'image', 'volumes': {'host': {}}})
    env = EnclaveAgentEnvironment(engine='e2b', sandbox_config={'image': 'image'})
    with pytest.raises(SandboxConfigError, match='volumes'):
        env.merge_sandbox_config({'volumes': {'host': {}}})
    assert 'volumes' not in env._sandbox_config_dict


def test_e2b_service_mapping_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    from ms_enclave.sandbox.model import E2BSandboxManagerConfig

    from evalscope.api.sandbox import SandboxService
    from evalscope.api.sandbox import service as service_module
    managers = []
    class FakeManager:
        def __init__(self, config):
            self.config = config
            self.start, self.stop = AsyncMock(), AsyncMock()
            managers.append(self)
    monkeypatch.setattr(service_module, 'get_enclave_types', lambda engine: (None, None, FakeManager, E2BSandboxManagerConfig))
    service = SandboxService()
    original = {'api_key': 'placeholder'}
    async def run():
        first = await service._get_or_create_manager(SandboxEngine.E2B, original)
        second = await service._get_or_create_manager(SandboxEngine.E2B, original)
        assert first is second and len(first.config.image_mapping) == 3518
        assert 'image_mapping' not in str(service._managers.keys())
    from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
    try:
        AsyncioLoopRunner.run(run())
    finally:
        service.shutdown_all()
    assert original == {'api_key': 'placeholder'}
    managers[0].stop.assert_awaited_once()


def test_service_shutdown_cancels_real_manager_capacity_waiter(monkeypatch: pytest.MonkeyPatch) -> None:
    from e2b import AsyncSandbox
    from ms_enclave.sandbox.model import E2BSandboxConfig

    from evalscope.api.sandbox import SandboxService

    remote = SimpleNamespace(sandbox_id='remote-test', kill=AsyncMock())
    create = AsyncMock(return_value=remote)
    monkeypatch.setattr(AsyncSandbox, 'create', create)
    service = SandboxService()
    config = {'api_key': 'placeholder', 'max_sandboxes': 1}
    waiting = threading.Event()

    async def resolve(*args: object, **kwargs: object) -> str:
        waiting.set()
        return 'explicit'

    async def run() -> None:
        owner_loop = service._runtime.loop
        manager = await service.get_or_create_manager(SandboxEngine.E2B, config)
        sandbox_config = E2BSandboxConfig(template='explicit')
        await service.create_sandbox(SandboxEngine.E2B, sandbox_config, config)
        monkeypatch.setattr(manager._templates, 'resolve', resolve)
        queued = asyncio.create_task(service.create_sandbox(SandboxEngine.E2B, sandbox_config, config))
        try:
            assert await asyncio.to_thread(waiting.wait, 1)
            await asyncio.wait_for(service.shutdown_all_async(), 1)
            with pytest.raises(asyncio.CancelledError):
                await queued
            stats = await manager.get_stats()
            assert stats['total_sandboxes'] == 0 and stats['running'] is False
            create.assert_awaited_once()
            remote.kill.assert_awaited_once()
        finally:
            if not owner_loop.is_closed():
                await asyncio.wrap_future(service._runtime.submit(manager.stop()))
            await asyncio.gather(queued, return_exceptions=True)
            await service.shutdown_all_async()

    asyncio.run(run())


def test_manager_cache_distinguishes_secret_values() -> None:
    from pydantic import SecretStr

    from evalscope.api.sandbox.service import _freeze
    assert _freeze({'api_key': SecretStr('first')}) != _freeze({'api_key': SecretStr('second')})
    assert _freeze({'api_key': SecretStr('first')}) == _freeze({'api_key': 'first'})


def test_e2b_pool_rejects_impossible_live_limit() -> None:
    from evalscope.api.sandbox import SandboxService
    manager = SimpleNamespace(config=SimpleNamespace(max_sandboxes=2), initialize_pool=AsyncMock())
    service = SandboxService()
    try:
        with pytest.raises(SandboxConfigError, match='pool_size=3.*max_sandboxes=2'):
            asyncio.run(service._initialize_pool(SandboxEngine.E2B, manager, 3, object()))
        manager.initialize_pool.assert_not_awaited()
    finally:
        service.shutdown_all()


def test_structural_engine_boundary() -> None:
    import ast
    for path in (ROOT / 'evalscope').rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != 'e2b' and not alias.name.startswith('e2b.') for alias in node.names), path
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module != 'e2b' and not (node.module or '').startswith('e2b.'), path
            if 'benchmarks' in path.parts and isinstance(node, ast.Compare):
                assert not any(isinstance(n, ast.Attribute) and n.attr == 'E2B' for n in ast.walk(node)), path
                assert not any(isinstance(n, ast.Constant) and n.value == 'e2b' for n in ast.walk(node)), path


@pytest.mark.parametrize('engine', ['docker', 'e2b', 'volcengine'])
@pytest.mark.parametrize('site', ['swe', 'pro', 'atlas', 'wide', 'gaia', 'office', 'job', 'gdp', 'skills'])
def test_migrated_environment_sites(site: str, engine: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.dataset import Sample
    from evalscope.benchmarks.gaia.gaia_adapter import GaiaAdapter
    from evalscope.benchmarks.gdpval.gdpval_adapter import GDPvalAdapter
    from evalscope.benchmarks.job_bench.job_bench_adapter import JobBenchAdapter
    from evalscope.benchmarks.officeqa.officeqa_adapter import OfficeQAAdapter
    from evalscope.benchmarks.skillsbench.skillsbench_adapter import SkillsBenchAdapter
    from evalscope.benchmarks.swe_atlas_qna.swe_atlas_qna_adapter import SweAtlasQnAAdapter
    from evalscope.benchmarks.swe_bench.swe_bench_agentic_adapter import SWEBenchVerifiedAgenticAdapter
    from evalscope.benchmarks.swe_bench_pro.swe_bench_pro_agentic_adapter import SWEBenchProAgenticAdapter
    from evalscope.benchmarks.wide_search.wide_search_adapter import WideSearchAdapter
    classes = dict(swe=SWEBenchVerifiedAgenticAdapter, pro=SWEBenchProAgenticAdapter, atlas=SweAtlasQnAAdapter,
                   wide=WideSearchAdapter, gaia=GaiaAdapter, office=OfficeQAAdapter, job=JobBenchAdapter,
                   gdp=GDPvalAdapter, skills=SkillsBenchAdapter)
    adapter = classes[site].__new__(classes[site])
    adapter._benchmark_meta = SimpleNamespace(name=site)
    adapter._task_config = task_stub(engine)
    adapter._task_config.agent_config = SimpleNamespace(environment='docker')
    adapter._task_sandbox_config = lambda: {}
    adapter._native_command_timeout = lambda: 120
    adapter.working_dir = '/testbed' if site == 'swe' else '/app'
    adapter.force_arch = ''
    adapter._host_files_dir = None
    adapter._corpus_dir = str(tmp_path)
    adapter.reset_to_base_commit = adapter.fail_on_source_modification = False
    adapter.command_timeout = 60
    adapter.docker_image_default = adapter.docker_image = 'image'
    adapter._current_output_dir = str(tmp_path)
    sample = Sample(id=0, input='task', target='target', metadata={
        'docker_image': 'image', 'instance_id': 'demo', 'artifact_dir': str(tmp_path), 'image_tag': 'image',
        'working_dir': '/app', 'frontmatter': {'environment': {'network_mode': 'no-network'}}})
    seen = []
    def env(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(**kwargs)
    monkeypatch.setattr('evalscope.agent.environments.enclave.EnclaveAgentEnvironment', env)
    monkeypatch.setattr('evalscope.benchmarks.swe_atlas_qna.swe_atlas_qna_adapter.SweAtlasEnclaveEnvironment', env)
    if site == 'skills':
        monkeypatch.setattr('evalscope.benchmarks.skillsbench.skillsbench_adapter.EnclaveAgentEnvironment', env)
    unsupported = site in ['gaia', 'office', 'job', 'gdp', 'skills']
    if site == 'gdp':
        adapter._ensure_docker_image = MagicMock(side_effect=AssertionError('must not build before rejecting remote engine'))
        if engine == 'docker':
            adapter._ensure_docker_image.side_effect = None
        monkeypatch.setattr('evalscope.benchmarks.gdpval.gdpval_adapter.build_reference_volumes', lambda sample: {})
    def build():
        if site == 'skills':
            # load() must reject remote engines before touching any task Dockerfile.
            if engine != 'docker':
                return adapter.load()
            return adapter._build_environment(sample)
        return adapter.build_environment(sample)
    if engine == 'volcengine' or (engine == 'e2b' and unsupported):
        with pytest.raises(SandboxConfigError, match=engine):
            build()
        assert not seen
        if site == 'gdp':
            adapter._ensure_docker_image.assert_not_called()
    else:
        build()
        assert len(seen) == 1 and seen[0]['engine'] == resolve_engine(engine)
        assert ('manager_config' in seen[0]) == (engine == 'e2b')
        if site in ['swe', 'pro', 'atlas']:
            assert seen[0]['interpreter'] == ['bash', '-lc'] or seen[0]['interpreter'] == ('bash', '-lc')
        if site == 'gaia':
            assert 'volumes' not in seen[0]['sandbox_config']
