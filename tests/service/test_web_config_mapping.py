from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evalscope.config import SandboxTaskConfig, TaskConfig
from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import ApiRegistry
from evalscope.service.app import create_app
from evalscope.service.blueprints.eval import _build_task_config
from evalscope.service.utils import log


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from evalscope.service.blueprints import eval as eval_routes
    from evalscope.service.blueprints import perf
    monkeypatch.setattr(log, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(eval_routes, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(perf, 'OUTPUT_DIR', str(tmp_path))
    return create_app(outputs=str(tmp_path)).test_client()


def test_all_eval_fields_survive_actual_config_parsing(client, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    def execute(func: object, config: TaskConfig, **kwargs: object) -> dict:
        captured['config'] = config
        return {'general_qa': {'score': 1}}
    monkeypatch.setattr('evalscope.service.blueprints.eval.run_in_subprocess', execute)
    monkeypatch.setattr('evalscope.service.blueprints.eval._build_result_table', lambda path: '')
    payload = {
        'model': 'same-model', 'datasets': ['general_qa', 'gsm8k'], 'api_url': 'http://localhost:18101/v1',
        'api_key': 'fake-key', 'alias': 'server-A', 'limit': .25, 'eval_batch_size': 2, 'repeats': 3,
        'generation_config': {'stream': False, 'timeout': 2.5, 'temperature': .85, 'top_p': .95, 'max_tokens': 32, 'top_k': -1},
        'dataset_args': {'general_qa': {'local_path': '/tmp/qa.jsonl'}},
    }
    response = client.post('/api/v1/eval/invoke', json=payload, headers={'EvalScope-Task-Id': 'eval_config'})
    assert response.status_code == 200
    config = captured['config']
    assert config.model == payload['model']
    assert config.datasets == payload['datasets']
    assert config.api_url == payload['api_url']
    assert config.api_key.get_secret_value() == 'fake-key'
    assert config.limit == .25 and config.repeats == 3 and config.eval_batch_size == 2
    assert config.dataset_args == payload['dataset_args']
    for key, value in payload['generation_config'].items():
        assert getattr(config.generation_config, key) == value
    assert config.timeout is None and config.stream is None
    assert 'alias' not in TaskConfig.model_fields


def test_all_perf_fields_survive_actual_config_parsing(client, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    def execute(func: object, config: Arguments, **kwargs: object) -> dict:
        captured['config'] = config
        return {}
    monkeypatch.setattr('evalscope.service.blueprints.perf.run_in_subprocess', execute)
    payload = {
        'model': 'same-model', 'api': 'openai', 'url': 'http://localhost:18101/v1', 'api_key': 'fake-key', 'alias': 'server-A',
        'parallel': [1, 2], 'number': [10, 20], 'rate': .5, 'max_tokens': 32, 'min_tokens': 0, 'dataset': 'openqa',
        'max_prompt_length': 100, 'min_prompt_length': 0, 'tokenizer_path': '/tmp/tokenizer',
    }
    response = client.post('/api/v1/perf/invoke', json=payload, headers={'EvalScope-Task-Id': 'perf_config'})
    assert response.status_code == 200
    config = captured['config']
    for key, value in payload.items():
        if key == 'url':
            assert config.url == value + '/chat/completions'
        elif key == 'rate':
            assert config.rate == [.5]
        elif key == 'api_key':
            assert config.api_key.get_secret_value() == 'fake-key'
        else:
            assert getattr(config, key) == value
    request_config = config.model_copy(update={'tokenizer_path': None})
    plugin = ApiRegistry.get_class(config.api)(request_config)
    body = plugin.build_request([{'role': 'user', 'content': '2+2?'}])
    assert body['max_tokens'] == 32 and body['min_tokens'] == 0


@pytest.mark.parametrize('api', ['openai', 'local'])
def test_every_perf_ui_option_has_http_and_request_implementation(api: str) -> None:
    args = Arguments(model='same-model', api=api, number=1)
    plugin = ApiRegistry.get_class(api)
    assert plugin.process_request is not ApiPluginBase.process_request
    assert plugin(args).build_request([{'role': 'user', 'content': '2+2?'}]) is not None
    if api == 'local':
        assert args.url == 'http://127.0.0.1:8877/v1/chat/completions'


@pytest.mark.parametrize('overrides', [
    {'api': 'dashscope'}, {'api': 'unknown'}, {'dataset': 'unknown'}, {'rate': 0}, {'rate': -2},
    {'parallel': [1, 2], 'number': [1]}, {'max_tokens': 1.5}, {'min_tokens': 9, 'max_tokens': 8},
    {'min_prompt_length': 20, 'max_prompt_length': 10}, {'dataset': 'random'}, {'number': [0]},
])
def test_invalid_perf_values_fail_before_execution(client, monkeypatch: pytest.MonkeyPatch, overrides: dict) -> None:
    execute = Mock()
    monkeypatch.setattr('evalscope.service.blueprints.perf.run_in_subprocess', execute)
    response = client.post('/api/v1/perf/invoke', json={'model': 'same-model', 'url': 'http://localhost/v1', **overrides},
                           headers={'EvalScope-Task-Id': 'perf_invalid'})
    assert response.status_code == 400, response.get_json()
    execute.assert_not_called()


@pytest.mark.parametrize('overrides', [
    {'limit': 2.5}, {'repeats': 1.5}, {'eval_batch_size': 1.5}, {'datasets': ['general_qa'], 'dataset_args': {'general_qa': []}},
    {'generation_config': {'max_tokens': 1.5}}, {'generation_config': {'top_k': 1.5}},
    {'generation_config': {'timeout': 0}}, {'generation_config': {'top_p': 1.5}},
])
def test_invalid_eval_values_fail_before_execution(client, monkeypatch: pytest.MonkeyPatch, overrides: dict) -> None:
    execute = Mock()
    monkeypatch.setattr('evalscope.service.blueprints.eval.run_in_subprocess', execute)
    response = client.post('/api/v1/eval/invoke', json={'model': 'same-model', 'api_url': 'http://localhost/v1', 'datasets': ['general_qa'], **overrides},
                           headers={'EvalScope-Task-Id': 'eval_invalid'})
    assert response.status_code == 400
    execute.assert_not_called()


@pytest.mark.parametrize('engine', ['docker', 'volcengine'])
def test_sandbox_options_map_to_framework_config(engine: str) -> None:
    overrides = {'enabled': True, 'engine': engine, 'pool_size': 2}
    if engine == 'docker':
        overrides['manager_config'] = {'base_url': 'http://sandbox:8000'}
        overrides['default_config'] = {'image': 'python:3.12'}
    task = _build_task_config({'model': 'same-model', 'api_url': 'http://localhost/v1', 'sandbox': overrides})
    config = task.sandbox
    assert config.engine == engine and config.pool_size == 2
    if engine == 'docker':
        assert config.manager_config == overrides['manager_config']
        assert config.default_config == overrides['default_config']
    else:
        assert config.manager_config == {} and config.default_config == {}


def test_local_protocol_does_not_require_remote_url(client, monkeypatch: pytest.MonkeyPatch) -> None:
    execute = Mock(return_value={})
    monkeypatch.setattr('evalscope.service.blueprints.perf.run_in_subprocess', execute)
    response = client.post('/api/v1/perf/invoke', json={'model': 'same-model', 'api': 'local'},
                           headers={'EvalScope-Task-Id': 'perf_local_config'})
    assert response.status_code == 200
    args = execute.call_args.args[1]
    assert args.api == 'local' and args.url == 'http://127.0.0.1:8877/v1/chat/completions'


def test_positive_rate_html_summary_formats_configuration_as_configuration() -> None:
    from evalscope.perf.utils.report.summary import format_summary_rows
    columns = [{'key': 'concurrency'}, {'key': 'request_rate'}]
    rows = [{'values': {'concurrency': 2, 'request_rate': 2.5}}]
    assert format_summary_rows(columns, rows) == [['2', '2.5']]


def test_sweep_configuration_shows_all_executed_values() -> None:
    from evalscope.perf.utils.report.summary import build_run_config
    runs = [
        SimpleNamespace(parallel=1, number=2, rate=None, summary=SimpleNamespace(request_rate=2.5)),
        SimpleNamespace(parallel=2, number=4, rate=None, summary=SimpleNamespace(request_rate=2.5)),
    ]
    config = build_run_config({'parallel': 1, 'number': 2, 'rate': 2.5}, runs)
    assert config == {'parallel': '1, 2', 'number': '2, 4', 'rate': '2.5, 2.5'}


def test_backend_defaults_used_by_optional_ui_fields() -> None:
    args = Arguments(model='same-model')
    config = _build_task_config({'model': 'same-model', 'api_url': 'http://localhost/v1'})
    assert args.max_tokens == 2048
    assert args.max_prompt_length == 131072
    assert args.rate == [-1]
    assert config.eval_batch_size == 8


@pytest.mark.parametrize('limit', [0, .25, 1, 20])
def test_limit_matches_count_fraction_and_unlimited_semantics(limit: int | float) -> None:
    config = _build_task_config({'model': 'same-model', 'api_url': 'http://localhost/v1', 'limit': limit})
    assert config.limit == (None if limit == 0 else limit)
