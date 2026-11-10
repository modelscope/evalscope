from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from evalscope.api.mixin import llm_judge_mixin
from evalscope.api.model import model as model_module
from evalscope.api.model.generate_config import GenerateConfig
from evalscope.api.sandbox import SandboxEngine, SandboxService, build_sandbox_config
from evalscope.config import SandboxTaskConfig, TaskConfig
from evalscope.models.utils.openai import openai_completion_params
from evalscope.models.utils.openai_responses import openai_response_params
from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.utils.argument_utils import get_secret_value


@pytest.mark.parametrize('source', ['nested', 'typed', 'legacy'])
def test_sandbox_keys_use_secretstr_for_display_and_updates(source: str) -> None:
    payload = {'api_key': 'sandbox-test-secret', 'nested': [{'api_key': 'nested-test-secret'}], 'empty': {'api_key': ''}}
    settings = {'manager_config': payload, 'default_config': payload}
    kwargs = {'sandbox_manager_config': payload} if source == 'legacy' else {
        'sandbox': SandboxTaskConfig(**settings) if source == 'typed' else settings,
    }
    task = TaskConfig(model='test', eval_type='mock_llm', **kwargs)
    assert isinstance(task.sandbox.manager_config['api_key'], SecretStr)
    assert get_secret_value(task.sandbox.manager_config) == payload
    assert 'sandbox-test-secret' not in repr(task)
    assert 'nested-test-secret' not in task.model_dump_json()
    assert task.to_dict()['sandbox']['manager_config']['empty']['api_key'] == ''
    clone = TaskConfig(model='test', eval_type='mock_llm')
    clone.update(task)
    assert get_secret_value(clone.sandbox.manager_config) == payload
    clone.update({'sandbox': {'manager_config': {'api_key': 'updated-test-secret'}}})
    assert isinstance(clone.sandbox.manager_config['api_key'], SecretStr)
    assert get_secret_value(clone.sandbox.manager_config)['api_key'] == 'updated-test-secret'
    assert 'updated-test-secret' not in str(clone)
    assert payload['api_key'] == 'sandbox-test-secret'


@pytest.mark.parametrize('engine', [SandboxEngine.DOCKER, SandboxEngine.E2B])
def test_sandbox_runtime_config_receives_raw_env_values(engine: SandboxEngine) -> None:
    task = TaskConfig(model='test', eval_type='mock_llm', sandbox={
        'enabled': True, 'engine': engine.value,
        'default_config': {'image': 'python:3.11-slim', 'env_vars': {'api_key': 'runtime-test-secret'}},
        'manager_config': {'api_key': 'manager-test-secret'},
    })
    config = build_sandbox_config(engine, task.sandbox.default_config)
    assert config.env_vars['api_key'] == 'runtime-test-secret'
    assert isinstance(task.sandbox.default_config['env_vars']['api_key'], SecretStr)


def test_sandbox_manager_receives_raw_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.api.sandbox import service as service_module

    manager = MagicMock()
    monkeypatch.setattr(service_module, 'get_enclave_types', lambda engine: (None, None, manager, None))
    sandbox = SandboxTaskConfig(manager_config={'api_key': 'manager-test-secret'})
    service = SandboxService()
    try:
        service._construct_manager(SandboxEngine.VOLCENGINE, sandbox.manager_config)
        manager.assert_called_once_with(api_key='manager-test-secret')
        assert isinstance(sandbox.manager_config['api_key'], SecretStr)
    finally:
        service.shutdown_all()


def test_perf_arguments_masks_api_key_and_authorization_header():
    args = Arguments(
        model='test-model',
        url='http://localhost:8080/v1/chat/completions',
        api_key='secret-token',
        wandb_api_key='wandb-token',
        swanlab_api_key='swanlab-token',
    )

    args_dict = args.to_dict()
    args_text = str(args)

    assert args_dict['api_key'] == '**********'
    assert args_dict['wandb_api_key'] == '**********'
    assert args_dict['swanlab_api_key'] == '**********'
    assert args_dict['headers']['Authorization'] == '**********'
    assert 'secret-token' not in args_text
    assert 'Bearer secret-token' not in args_text
    assert 'wandb-token' not in args_text
    assert 'swanlab-token' not in args_text
    assert '**********' in args_text


def test_perf_runtime_headers_use_raw_secret_value():
    args = Arguments(
        model='test-model',
        url='http://localhost:8080/v1/chat/completions',
        api_key='secret-token',
    )

    runtime_headers = AioHttpClient._get_runtime_headers(args.headers)

    assert runtime_headers['Authorization'] == 'Bearer secret-token'


def test_task_config_string_omits_api_keys():
    task_config = TaskConfig(
        model='test-model',
        api_url='http://localhost:8080/v1/chat/completions',
        api_key='secret-token',
        datasets=['gsm8k'],
        judge={'models': {
            'model_id': 'judge-model',
            'api_key': 'judge-secret-token',
            'model_args': {'nested': {'api_key': 'nested-judge-secret-token'}},
        }},
    )

    task_config_text = str(task_config)
    task_config_dict = task_config.to_dict()

    assert 'secret-token' not in task_config_text
    assert 'judge-secret-token' not in task_config_text
    assert 'nested-judge-secret-token' not in task_config_text
    assert task_config_dict['api_key'] == '**********'
    judge_model = task_config_dict['judge']['models'][0]
    assert judge_model['api_key'] == '**********'
    assert judge_model['model_args']['nested']['api_key'] == '**********'


def test_task_config_string_omits_extra_auth_headers():
    task_config = TaskConfig(
        model='test-model',
        api_url='http://localhost:8080/v1/chat/completions',
        datasets=['gsm8k'],
        generation_config={'extra_headers': {
            'Authorization': 'Bearer eval-secret-token'
        }},
    )

    task_config_text = str(task_config)
    task_config_dict = task_config.to_dict()

    assert 'eval-secret-token' not in task_config_text
    assert task_config_dict['generation_config']['extra_headers']['Authorization'] == '**********'


def test_get_model_with_task_config_uses_raw_secret_value(monkeypatch):
    captured = {}

    def fake_get_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(model_module, 'get_model', fake_get_model)
    task_config = TaskConfig(
        model='test-model',
        api_url='http://localhost:8080/v1/chat/completions',
        api_key='secret-token',
        datasets=['gsm8k'],
    )

    model_module.get_model_with_task_config(task_config)

    assert captured['api_key'] == 'secret-token'


def test_llm_judge_mixin_uses_raw_nested_judge_model_config(monkeypatch):
    captured = {}

    class FakeLLMJudge:

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_judge_mixin, 'LLMJudge', FakeLLMJudge)
    task_config = TaskConfig(
        model='test-model',
        api_url='http://localhost:8080/v1/chat/completions',
        datasets=['gsm8k'],
        judge={'strategy': 'llm', 'models': {
            'model_id': 'judge-model',
            'api_key': 'judge-secret-token',
            'model_args': {'nested': {'api_key': 'nested-judge-secret-token'}},
        }},
    )

    judges = llm_judge_mixin.LLMJudgeMixin(benchmark_meta=object(), task_config=task_config).init_llm_judges()

    assert judges[0].judge_id == 'judge-model'
    assert captured['api_key'] == 'judge-secret-token'
    assert captured['model_args']['nested']['api_key'] == 'nested-judge-secret-token'


def test_openai_params_use_raw_extra_auth_headers():
    config = GenerateConfig(extra_headers={'Authorization': 'Bearer eval-secret-token'})

    chat_params = openai_completion_params('test-model', config, tools=False)
    response_params = openai_response_params('test-model', config, tools=False)

    assert chat_params['extra_headers']['Authorization'] == 'Bearer eval-secret-token'
    assert response_params['extra_headers']['Authorization'] == 'Bearer eval-secret-token'


def test_task_config_update_revalidates_generation_config_headers():
    task_config = TaskConfig(
        model='test-model',
        api_url='http://localhost:8080/v1/chat/completions',
        datasets=['gsm8k'],
    )

    task_config.update({'generation_config': {'extra_headers': {'Authorization': 'Bearer update-secret-token'}}})

    task_config_text = str(task_config)
    task_config_dict = task_config.to_dict()
    runtime_headers = openai_completion_params('test-model', task_config.generation_config,
                                               tools=False)['extra_headers']

    assert 'update-secret-token' not in task_config_text
    assert task_config_dict['generation_config']['extra_headers']['Authorization'] == '**********'
    assert runtime_headers['Authorization'] == 'Bearer update-secret-token'
