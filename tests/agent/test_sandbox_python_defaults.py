from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.mixin.code_execution_sandbox_mixin import EnclaveCodeExecutionBackend
from evalscope.api.sandbox import resolve_sandbox_config
from evalscope.benchmarks.wide_search.wide_search_adapter import WideSearchAdapter
from evalscope.config import SandboxTaskConfig

DEFAULTS = {'image': 'python:3.11-slim', 'tools_config': {'python_executor': {}}}
ALIYUN = {'api_key': 'placeholder', 'domain': 'cn-hangzhou.sandbox.aliyuncs.com'}


@pytest.fixture(autouse=True)
def clear_endpoint_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('E2B_API_URL', 'E2B_DOMAIN'):
        monkeypatch.delenv(name, raising=False)


def sandbox_config(**overrides: Any) -> SandboxTaskConfig:
    return SandboxTaskConfig(enabled=True, engine='e2b', manager_config=ALIYUN, **overrides)


def test_aliyun_default_python_uses_base_without_mutating_settings() -> None:
    defaults = deepcopy(DEFAULTS)
    sandbox = sandbox_config(default_config={'sandbox_timeout': 600, 'env_vars': {'KEY': 'value'}})
    before = sandbox.model_dump()

    config = resolve_sandbox_config(defaults, sandbox)

    assert config == {
        'template': 'base',
        'tools_config': DEFAULTS['tools_config'],
        'sandbox_timeout': 600,
        'env_vars': {'KEY': 'value'},
    }
    assert defaults == DEFAULTS
    assert sandbox.model_dump() == before


@pytest.mark.parametrize(
    'overrides',
    [
        {'image': 'custom:latest'},
        {'image': 'python:3.11-slim'},
        {'template': 'custom-template'},
        {'image': 'custom:latest', 'template': 'custom-template'},
    ],
)
def test_explicit_images_and_templates_take_precedence(overrides: dict[str, Any]) -> None:
    assert resolve_sandbox_config(DEFAULTS, sandbox_config(default_config=overrides)) == {**DEFAULTS, **overrides}


@pytest.mark.parametrize(
    'source',
    ['python:3.11-slim', 'docker.io/python:3.11-slim', ' python:3.11-slim ', ' docker.io/python:3.11-slim '],
)
def test_custom_image_mapping_takes_precedence(source: str) -> None:
    sandbox = sandbox_config()
    sandbox.manager_config = {**ALIYUN, 'image_mapping': {source: 'custom-registry/python:3.11'}}
    assert resolve_sandbox_config(DEFAULTS, sandbox) == DEFAULTS


@pytest.mark.parametrize('engine,enabled', [('docker', True), ('volcengine', True), ('e2b', False)])
def test_other_engines_and_disabled_sandboxes_keep_python_image(engine: str, enabled: bool) -> None:
    sandbox = SandboxTaskConfig(engine=engine, enabled=enabled)
    assert resolve_sandbox_config(DEFAULTS, sandbox) == DEFAULTS
    assert resolve_sandbox_config(DEFAULTS, None) == DEFAULTS


@pytest.mark.parametrize('domain', ['e2b.app', 'sandbox.example.com'])
def test_non_aliyun_services_keep_python_image(domain: str) -> None:
    sandbox = SandboxTaskConfig(enabled=True, engine='e2b', manager_config={'api_key': 'placeholder', 'domain': domain})
    assert resolve_sandbox_config(DEFAULTS, sandbox) == DEFAULTS


def test_aliyun_endpoint_environment_variables_are_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('E2B_API_URL', 'https://api.cn-hangzhou.sandbox.aliyuncs.com')
    monkeypatch.setenv('E2B_DOMAIN', 'cn-hangzhou.sandbox.aliyuncs.com')
    sandbox = SandboxTaskConfig(enabled=True, engine='e2b', manager_config={'api_key': 'placeholder'})
    assert resolve_sandbox_config(DEFAULTS, sandbox)['template'] == 'base'


@pytest.mark.parametrize('defaults', [{'image': 'custom:latest'}, {**DEFAULTS, 'template': 'benchmark-template'}])
def test_other_benchmark_environments_are_unchanged(defaults: dict[str, Any]) -> None:
    assert resolve_sandbox_config(defaults, sandbox_config()) == defaults


def test_code_execution_pool_receives_automatic_template() -> None:
    backend = EnclaveCodeExecutionBackend(
        SimpleNamespace(sandbox_config=DEFAULTS), SimpleNamespace(sandbox=sandbox_config(), eval_batch_size=1)
    )
    assert backend._resolve_sandbox_config_dict()['template'] == 'base'


def test_widesearch_environment_receives_automatic_template() -> None:
    adapter = WideSearchAdapter.__new__(WideSearchAdapter)
    adapter._task_config = SimpleNamespace(sandbox=sandbox_config())
    sample = Sample(input='question', metadata={'instance_id': 'python-default'})

    environment = adapter.build_environment(sample)

    assert environment._sandbox_config.template == 'base'
    assert environment._sandbox_config.image is None
    assert environment._sandbox_config.allow_internet_access is True
