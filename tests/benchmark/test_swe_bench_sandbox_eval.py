"""Official harness protocol checks and opt-in real provider smoke tests."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalscope.api.agent import ExecResult
from evalscope.api.sandbox import SandboxConfigError, SandboxEngine
from evalscope.benchmarks.swe_bench import utils as swe
from evalscope.benchmarks.swe_bench_pro import utils as pro
from evalscope.config import TaskConfig


class FakeEnv:
    def __init__(self, *, apply: tuple[bool, ...] = (True, True, True), failure: str = '', **kwargs: Any) -> None:
        self.kwargs, self.apply, self.failure = kwargs, apply, failure
        self.calls = []
        self.uploads = []
        self.closed = 0

    async def prepare(self) -> None:
        if self.failure == 'prepare':
            raise RuntimeError('prepare failed')

    async def put_dir(self, source: str | Path, target: str) -> None:
        if self.failure == 'upload':
            raise RuntimeError('upload failed')
        self.uploads.append((target, {p.name: p.read_text() for p in Path(source).iterdir()}))

    async def exec(self, command: list[str], **kwargs: Any) -> ExecResult:
        self.calls.append((command, kwargs))
        text = command[-1]
        for index, prefix in enumerate(swe.GIT_APPLY_CMDS):
            if text.startswith(prefix + ' /tmp/'):
                return ExecResult(returncode=0 if self.apply[index] else 1, stdout=f'apply {index}')
        if command[0] == 'cat':
            if text.endswith('output.json'):
                if self.failure == 'output':
                    return ExecResult(returncode=1, stderr='missing output')
                return ExecResult(returncode=0, stdout=json.dumps({'tests': [{'name': 'test', 'status': 'PASSED'}]}))
            return ExecResult(returncode=0, stdout='log output')
        if text == '/bin/bash /tmp/eval.sh 2>&1' or text == '/workspace/entryscript.sh':
            return ExecResult(returncode=-1 if self.failure == 'timeout' else 0,
                              stdout='test output', timed_out=self.failure == 'timeout')
        return ExecResult(returncode=0, stdout='diff')

    async def close(self) -> None:
        self.closed += 1


def install_harness(patch: pytest.MonkeyPatch, env: FakeEnv, resolved: bool = True, report_failure: bool = False) -> dict:
    spec = SimpleNamespace(instance_id='demo', instance_image_key='org/image:latest', platform='linux/amd64',
                           eval_script='echo test', docker_specs={'run_args': {'cap_add': ['SYS_ADMIN']}})
    seen = {}
    def make_test_spec(instance: dict, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return spec
    def make_env(**kwargs: Any) -> FakeEnv:
        env.kwargs = kwargs
        return env
    def get_report(**kwargs: Any) -> dict:
        if report_failure:
            raise RuntimeError('report failed')
        seen['report'] = kwargs
        return {'demo': {'resolved': resolved}}
    patch.setattr('swebench.harness.test_spec.test_spec.make_test_spec', make_test_spec)
    patch.setattr('swebench.harness.grading.get_eval_report', get_report)
    patch.setattr(swe, 'EnclaveAgentEnvironment', make_env)
    patch.setattr(pro, 'EnclaveAgentEnvironment', make_env)
    return seen


# Feature: e2b-sandbox, Property 18: Patch application tries the ordered prefix through the first success.
@settings(max_examples=100, deadline=None)
@given(st.tuples(st.booleans(), st.booleans(), st.booleans()))
def test_patch_order(outcomes: tuple[bool, ...]) -> None:
    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as root:
        env = FakeEnv(apply=outcomes)
        install_harness(patch, env)
        result = swe.eval_instance({'instance_id': 'demo'}, 'patch', task_config=None, log_dir=root)
        from swebench.harness.constants import APPLY_PATCH_FAIL, DOCKER_PATCH
        expected_count = next((i + 1 for i, success in enumerate(outcomes) if success), 3)
        apply_commands = [command[-1] for command, _ in env.calls if any(command[-1].startswith(c) for c in swe.GIT_APPLY_CMDS)]
        assert apply_commands == [f'{cmd} {DOCKER_PATCH}' for cmd in swe.GIT_APPLY_CMDS[:expected_count]]
        assert result['resolved'] == any(outcomes)
        assert result['completed'] == any(outcomes)
        if not any(outcomes):
            assert APPLY_PATCH_FAIL in result['report']['error'] and 'apply 2' in result['report']['error']
            assert not any(command[-1] == '/bin/bash /tmp/eval.sh 2>&1' for command, _ in env.calls)
        assert env.closed == 1


def test_harness_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = FakeEnv()
    seen = install_harness(monkeypatch, env)
    result = swe.eval_instance({'instance_id': 'demo'}, 'patch', task_config=None, log_dir=str(tmp_path))
    assert result == {'completed': True, 'resolved': True, 'report': {'demo': {'resolved': True}}}
    assert env.uploads == [('/tmp', {'patch.diff': 'patch', 'eval.sh': 'echo test'})]
    config = env.kwargs['sandbox_config']
    assert config['platform'] == 'linux/amd64' and config['command'] == 'tail -f /dev/null'
    assert config['cap_add'] == ['SYS_ADMIN']
    assert config['cpu_limit'] is None and config['memory_limit'] is None
    assert 'volumes' not in config
    assert env.kwargs['expected_duration'] == 1800
    assert len([c for c, _ in env.calls if c[-1] == 'git -c core.fileMode=false diff']) == 2
    assert any(c[-1] == '/bin/bash /tmp/eval.sh 2>&1' and k['timeout'] == 1800 for c, k in env.calls)
    assert seen['report']['include_tests_status'] is True
    from swebench.harness.constants import LOG_TEST_OUTPUT
    assert (tmp_path / 'swebench_log/demo' / LOG_TEST_OUTPUT).read_text() == 'test output'
    spec = SimpleNamespace(instance_image_key='image', platform='linux/amd64', docker_specs={})
    assert 'cap_add' not in swe._eval_sandbox_config(None, spec)


def test_docker_grading_preserves_unrestricted_defaults() -> None:
    from evalscope.api.sandbox import build_sandbox_config

    spec = SimpleNamespace(instance_image_key='image', platform='linux/amd64', docker_specs={})
    config = build_sandbox_config(SandboxEngine.DOCKER, swe._eval_sandbox_config(None, spec))
    assert config.memory_limit is None and config.cpu_limit is None
    task = SimpleNamespace(sandbox=SimpleNamespace(enabled=False, engine='docker',
                                                  default_config={'memory_limit': '8g', 'cpu_limit': 4}))
    config = build_sandbox_config(SandboxEngine.DOCKER, swe._eval_sandbox_config(task, spec))
    assert config.memory_limit == '8g' and config.cpu_limit == 4
    task.sandbox.enabled, task.sandbox.engine = True, 'e2b'
    task.sandbox.default_config = {}
    config = build_sandbox_config(SandboxEngine.E2B, swe._eval_sandbox_config(task, spec))
    assert config.memory_mb == 4096 and config.cpu_count == 2


@pytest.mark.parametrize('agentic', [False, True])
def test_swe_missing_runtime_fails_at_initialization(monkeypatch: pytest.MonkeyPatch, agentic: bool) -> None:
    import importlib.util

    from evalscope.api.benchmark import BenchmarkMeta
    from evalscope.benchmarks.swe_bench.swe_bench_adapter import SWEBenchVerifiedAdapter
    from evalscope.benchmarks.swe_bench.swe_bench_agentic_adapter import SWEBenchVerifiedAgenticAdapter

    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None if name == 'ms_enclave' else original(name))
    cls = SWEBenchVerifiedAgenticAdapter if agentic else SWEBenchVerifiedAdapter
    with pytest.raises(ImportError, match='ms_enclave.*evalscope\\[swe_bench\\]'):
        cls(benchmark_meta=BenchmarkMeta(name='swe-test', dataset_id='fake'), task_config=TaskConfig())


def test_remote_arch_and_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = FakeEnv()
    seen = install_harness(monkeypatch, env)
    task = SimpleNamespace(sandbox=SimpleNamespace(enabled=True, engine='e2b', manager_config={}, default_config={}))
    swe.eval_instance({'instance_id': 'demo'}, 'patch', task_config=task, log_dir=str(tmp_path), force_arch='arm64')
    assert seen['arch'] == 'x86_64' and env.kwargs['engine'] == SandboxEngine.E2B
    assert env.kwargs['expected_duration'] == 1800
    assert swe.resolve_swebench_arch('demo', 'arm64', SandboxEngine.DOCKER) == 'arm64'
    monkeypatch.setattr(swe, 'EnclaveAgentEnvironment', lambda **kw: (_ for _ in ()).throw(SandboxConfigError('bad config')))
    with pytest.raises(SandboxConfigError, match='bad config'):
        swe.eval_instance({'instance_id': 'demo'}, '', task_config=None, log_dir=str(tmp_path))


PRO_METADATA = {'instance_id': 'demo', 'base_commit': 'abc', 'fail_to_pass': "['test']", 'pass_to_pass': '[]',
                'test_patch': 'official test patch'}
PRO_RESOURCES = {'run_script': 'echo test', 'parser_script': 'print(1)', 'base_dockerfile': '', 'instance_dockerfile': ''}


@pytest.mark.parametrize('pro_case', [False, True])
def test_grading_preparation_error_propagates(
    pro_case: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeEnv(failure='prepare')
    install_harness(monkeypatch, env)
    with pytest.raises(RuntimeError, match='prepare failed'):
        if pro_case:
            pro.eval_instance(PRO_METADATA, 'patch', PRO_RESOURCES, 'image', task_config=None, log_dir=str(tmp_path))
        else:
            swe.eval_instance({'instance_id': 'demo'}, 'patch', task_config=None, log_dir=str(tmp_path))
    assert env.closed == 1
    assert env.uploads == [] and env.calls == []


def test_pro_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = FakeEnv()
    install_harness(monkeypatch, env)
    result = pro.eval_instance(PRO_METADATA, 'patch', PRO_RESOURCES, 'image', task_config=None, log_dir=str(tmp_path))
    assert result['completed'] and result['resolved']
    assert env.uploads[0][0] == '/workspace'
    assert set(env.uploads[0][1]) == {'patch.diff', 'run_script.sh', 'parser.py', 'entryscript.sh'}
    assert env.uploads[1] == ('/tests', {'test_patch.patch': PRO_METADATA['test_patch']})
    assert env.kwargs['sandbox_config']['working_dir'] == '/app'
    assert 'volumes' not in env.kwargs['sandbox_config']
    assert env.kwargs['expected_duration'] == 3600
    assert env.closed == 1
    workspace = tmp_path / 'swe_bench_pro_log/demo/workspace'
    assert all((workspace / name).is_file() for name in ['stdout.log', 'stderr.log', 'output.json'])
    assert (workspace.parent / 'container.log').is_file()


def test_pro_test_patch_upload_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock
    env = FakeEnv()
    install_harness(monkeypatch, env)
    env.put_dir = AsyncMock(side_effect=[None, RuntimeError('test patch upload failed')])
    result = pro.eval_instance(PRO_METADATA, 'patch', PRO_RESOURCES, 'image', task_config=None, log_dir=str(tmp_path))
    assert not result['completed'] and not result['resolved']
    assert 'test patch upload failed' in result['report']['error']
    assert env.closed == 1 and env.calls == []


# Feature: e2b-sandbox, Property 19: Infrastructure failures cannot become resolved evaluations.
@settings(max_examples=100, deadline=None)
@given(st.sampled_from(['upload', 'apply', 'timeout', 'report', 'output', '']), st.booleans())
def test_eval_fail_closed(failure: str, resolved: bool) -> None:
    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as root:
        env = FakeEnv(apply=(False, False, False) if failure == 'apply' else (True, True, True), failure=failure)
        install_harness(patch, env, resolved, report_failure=failure == 'report')
        if failure != 'output':
            result = swe.eval_instance({'instance_id': 'demo'}, 'patch', task_config=None, log_dir=root)
            assert result['completed'] == (not failure)
            assert result['resolved'] == (not failure and resolved)
            assert ('error' in result['report']) == bool(failure)
            assert env.closed == 1


        if failure in ['upload', 'timeout', 'output', '']:
            env.closed = 0
            result = pro.eval_instance(PRO_METADATA, 'patch', PRO_RESOURCES, 'image', task_config=None, log_dir=root)
            assert result['completed'] == (not failure)
            assert result['resolved'] == (not failure)
            assert ('error' in result['report']) == bool(failure)
            assert env.closed == 1


SMOKE_IDS = ['astropy__astropy-12907', 'django__django-11099', 'axios__axios-4731']
SMOKE_DATASETS = ['princeton-nlp/SWE-bench_Verified', 'princeton-nlp/SWE-bench_Lite', 'SWE-bench/SWE-bench_Multilingual']


@pytest.mark.parametrize('name', [
    'swe_bench_verified', 'swe_bench_verified_mini', 'swe_bench_lite', 'swe_bench_verified_agentic',
    'swe_bench_verified_mini_agentic', 'swe_bench_lite_agentic', 'swe_bench_multilingual_agentic', 'swe_bench_pro',
    'swe_atlas_qna', 'wide_search', 'gaia', 'gdpval', 'job_bench', 'officeqa', 'skillsbench', 'cybench',
])
def test_evaluation_versions_unchanged(name: str) -> None:
    from evalscope.api.registry import BENCHMARK_REGISTRY
    from evalscope.benchmarks import load_benchmark
    load_benchmark(name)
    # Pro v1.1 fixes missing official test-patch staging; all other versions match 5df5e509.
    assert BENCHMARK_REGISTRY[name].evaluation_version == ('v1.1' if name == 'swe_bench_pro' else 'v1.0')


def load_smoke_case(instance_id: str) -> dict:
    """Read an official instance through the normal dataset hub and cache."""
    from evalscope.api.dataset.hub import DatasetHub
    dataset = SMOKE_DATASETS[SMOKE_IDS.index(instance_id)]
    return next(row for row in DatasetHub(dataset).load('test') if row['instance_id'] == instance_id)


def provider_task() -> TaskConfig:
    """Build an E2B task using only environment-provided endpoints and credentials."""
    manager = {key: os.environ[var] for key, var in [('api_url', 'E2B_API_URL'), ('domain', 'E2B_DOMAIN')] if os.environ.get(var)}
    return TaskConfig(model='mock', eval_type='mock_llm', sandbox={
        'enabled': True, 'engine': 'e2b', 'manager_config': manager, 'default_config': {'sandbox_timeout': 3600}})


def run_provider_case(instance_id: str, task: TaskConfig, log_dir: Path) -> dict:
    """Assert gold and empty-patch outcomes using the implemented evaluator."""
    row = load_smoke_case(instance_id)
    results = {}
    for kind, prediction in [('gold', row['patch']), ('empty', '')]:
        result = swe.eval_instance(row, prediction, task_config=task, log_dir=str(log_dir / kind))
        assert result['resolved'] == (kind == 'gold'), result
        results[kind] = result
    return results


@pytest.mark.skipif(os.getenv('EVALSCOPE_DOCKER_SMOKE') != '1', reason='Set EVALSCOPE_DOCKER_SMOKE=1 for real Docker grading')
@pytest.mark.parametrize('instance_id', SMOKE_IDS)
def test_docker_consistency_smoke(instance_id: str, tmp_path: Path) -> None:
    baseline_file = Path(__file__).with_name('fixtures') / 'swe_bench_docker_baseline.json'
    if not baseline_file.is_file():
        pytest.skip('Generate the pre-change Docker fixture using spec task 10.1 first')
    baseline = json.loads(baseline_file.read_text())
    row = load_smoke_case(instance_id)
    for kind, prediction in [('gold', row['patch']), ('empty', '')]:
        result = swe.eval_instance(row, prediction, task_config=None, force_arch='x86_64', log_dir=str(tmp_path / kind))
        assert result['resolved'] == (kind == 'gold'), result
        assert result['report'][instance_id] == baseline[instance_id][kind]['report'][instance_id]


@pytest.mark.skipif(os.getenv('EVALSCOPE_E2B_SMOKE') != '1' or not os.getenv('E2B_API_KEY') or bool(os.getenv('E2B_API_URL')),
                    reason='Set official E2B smoke credentials without custom endpoints')
def test_official_smoke(tmp_path: Path) -> None:
    from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
    from evalscope.api.sandbox import task_sandbox_env_kwargs
    task = provider_task()
    run_provider_case(SMOKE_IDS[0], task, tmp_path)
    env = EnclaveAgentEnvironment(**task_sandbox_env_kwargs(task), sandbox_config={
        'image': swe.get_remote_docker_image_from_id(SMOKE_IDS[0], engine=SandboxEngine.E2B), 'sandbox_timeout': 3601})
    async def run() -> None:
        try:
            with pytest.raises(RuntimeError, match='cannot be greater than.*Hobby'):
                await env.prepare()
        finally:
            await env.close()
    asyncio.run(run())


@pytest.mark.skipif(os.getenv('EVALSCOPE_E2B_SMOKE') != '1' or not all(os.getenv(v) for v in ['E2B_API_KEY', 'E2B_API_URL', 'E2B_DOMAIN']),
                    reason='Set Aliyun E2B smoke credentials and endpoint variables')
def test_aliyun_smoke(tmp_path: Path) -> None:
    from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
    from evalscope.api.sandbox import task_sandbox_env_kwargs
    task = provider_task()
    for instance_id in SMOKE_IDS[:2]:
        run_provider_case(instance_id, task, tmp_path / instance_id)
    env = EnclaveAgentEnvironment(**task_sandbox_env_kwargs(task), sandbox_config={
        'image': swe.get_remote_docker_image_from_id(SMOKE_IDS[0], engine=SandboxEngine.E2B), 'working_dir': '/testbed'},
        interpreter=('bash', '-lc'))
    async def run() -> None:
        try:
            await env.prepare()
            python = await env.exec(['python', '--version'])
            status = await env.exec(['git', 'status'])
            assert python.returncode == 0 and 'Python 3.9.' in python.stdout
            assert status.returncode == 0
        finally:
            await env.close()
    asyncio.run(run())
