# Copyright (c) Alibaba, Inc. and its affiliates.
"""Operator-enabled plugins in the service: listing, request validation and workers."""

from pathlib import Path

import pytest

flask = pytest.importorskip('flask')

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.plugins import PluginNotFoundError
from evalscope.service.api_models import BenchmarksResponse, PluginsResponse
from evalscope.service.app import create_app
from evalscope.service.utils import log

from .conftest import full_plugin_source


@pytest.fixture
def plugin_prefix(fake_plugins) -> str:
    fake_plugins.install(
        'company-audit', full_plugin_source(fake_plugins.prefix), distribution='company-audit', version='0.4.2'
    )
    fake_plugins.install('not-enabled', 'raise RuntimeError("must not be imported")\n')
    return fake_plugins.prefix


@pytest.fixture
def client(plugin_prefix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from evalscope.service.blueprints import eval as eval_routes
    from evalscope.service.blueprints import perf as perf_routes

    for module in (log, eval_routes, perf_routes):
        monkeypatch.setattr(module, 'OUTPUT_DIR', str(tmp_path))
    return create_app(outputs=str(tmp_path), plugins=['company-audit']).test_client()


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Capture the config handed to the task worker instead of spawning it."""
    seen = {}

    def execute(func: object, config: object, **kwargs: object) -> dict:
        seen['config'] = config
        return {}

    monkeypatch.setattr('evalscope.service.blueprints.eval.run_in_subprocess', execute)
    monkeypatch.setattr('evalscope.service.blueprints.eval._build_result_table', lambda path: '')
    monkeypatch.setattr('evalscope.service.blueprints.eval._all_results_empty', lambda result: False)
    monkeypatch.setattr('evalscope.service.blueprints.perf.run_in_subprocess', execute)
    return seen


def _eval_payload(prefix: str, **overrides: object) -> dict:
    payload = {
        'model': 'dummy',
        'api_url': 'http://127.0.0.1:8901/v1',
        'eval_type': f'{prefix}_model_api',
        'datasets': [f'{prefix}_benchmark', 'gsm8k'],
    }
    payload.update(overrides)
    return payload


def test_service_lists_only_enabled_plugins(client, plugin_prefix: str) -> None:
    response = client.get('/api/v1/plugins')

    assert response.status_code == 200
    body = PluginsResponse.model_validate(response.get_json())
    [plugin] = body.plugins
    assert (plugin.name, plugin.distribution, plugin.version) == ('company-audit', 'company-audit', '0.4.2')
    assert plugin.registrations['model_api'] == [f'{plugin_prefix}_model_api']
    assert plugin.registrations['perf_api'] == [f'{plugin_prefix}_perf_api']


def test_benchmark_catalogue_includes_enabled_plugin_benchmarks(client, plugin_prefix: str) -> None:
    for query in ('', '?all=true'):
        body = BenchmarksResponse.model_validate(client.get(f'/api/v1/eval/benchmarks{query}').get_json())
        [entry] = [entry for entry in body.text if entry.plugin]
        assert entry.name == f'{plugin_prefix}_benchmark'
        assert entry.plugin == 'company-audit'
        assert entry.metrics == [f'{plugin_prefix}_metric']
        assert entry.description.en.sections['evaluation_notes'] == '- Always scores 1.'


def test_eval_request_runs_with_the_service_plugin_list(client, captured: dict, plugin_prefix: str) -> None:
    response = client.post(
        '/api/v1/eval/invoke', json=_eval_payload(plugin_prefix), headers={'EvalScope-Task-Id': 'eval_plugin'}
    )

    assert response.status_code == 200, response.get_json()
    config: TaskConfig = captured['config']
    assert config.plugins == ['company-audit']
    assert config.eval_type == f'{plugin_prefix}_model_api'


@pytest.mark.parametrize(
    'overrides, message',
    [
        ({'plugins': ['not-enabled']}, 'Plugins not enabled on this service: not-enabled'),
        ({'eval_type': 'no_such_api'}, "Unknown eval_type 'no_such_api'"),
        ({'datasets': ['no_such_benchmark']}, 'Unknown datasets: no_such_benchmark'),
    ],
    ids=['plugin-not-enabled', 'unknown-model-api', 'unknown-benchmark'],
)
def test_eval_request_rejects_names_outside_the_enabled_set(
    client, captured: dict, plugin_prefix: str, overrides: dict, message: str
) -> None:
    response = client.post(
        '/api/v1/eval/invoke',
        json=_eval_payload(plugin_prefix, **overrides),
        headers={'EvalScope-Task-Id': 'eval_rejected'},
    )

    assert response.status_code == 400
    assert message in response.get_json()['error']
    assert 'config' not in captured


def test_perf_request_runs_with_the_service_plugin_list(client, captured: dict, plugin_prefix: str) -> None:
    payload = {'model': 'dummy', 'api': f'{plugin_prefix}_perf_api', 'url': 'http://127.0.0.1:8901/v1/chat'}

    response = client.post('/api/v1/perf/invoke', json=payload, headers={'EvalScope-Task-Id': 'perf_plugin'})

    assert response.status_code == 200, response.get_json()
    config: PerfArguments = captured['config']
    assert config.plugins == ['company-audit']
    assert config.api == f'{plugin_prefix}_perf_api'


def test_perf_request_cannot_enable_another_plugin(client, captured: dict) -> None:
    payload = {'model': 'dummy', 'url': 'http://127.0.0.1:8901/v1', 'plugins': ['not-enabled']}

    response = client.post('/api/v1/perf/invoke', json=payload, headers={'EvalScope-Task-Id': 'perf_rejected'})

    assert response.status_code == 400
    assert 'Plugins not enabled on this service: not-enabled' in response.get_json()['error']


def test_service_without_plugins_exposes_none(fake_plugins) -> None:
    client = create_app().test_client()

    assert client.get('/api/v1/plugins').get_json() == {'plugins': []}


def test_service_does_not_start_with_a_missing_plugin(fake_plugins) -> None:
    with pytest.raises(PluginNotFoundError, match="'missing' is not installed"):
        create_app(plugins=['missing'])
