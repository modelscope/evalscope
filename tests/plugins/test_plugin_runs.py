# Copyright (c) Alibaba, Inc. and its affiliates.
"""Plugins in TaskConfig / perf Arguments, run records, and the spawned-worker path."""

import pickle
from pathlib import Path

import pytest

from evalscope import plugins
from evalscope.api.registry import MODEL_APIS
from evalscope.config import TaskConfig, parse_task_config
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.run import run_task
from evalscope.utils.io_utils import yaml_to_dict

from .conftest import full_plugin_source


def test_task_config_loads_plugins_and_records_versions(fake_plugins, tmp_path: Path) -> None:
    prefix = fake_plugins.prefix
    fake_plugins.install('company-audit', full_plugin_source(prefix), distribution='company-audit', version='0.4.2')

    config = TaskConfig(
        plugins='company-audit', model='dummy', eval_type=f'{prefix}_model_api', datasets=[f'{prefix}_benchmark']
    )
    config.dump_yaml(str(tmp_path))

    snapshot = yaml_to_dict(str(tmp_path / 'task_config.yaml'))
    assert snapshot['plugins'] == ['company-audit']
    [record] = snapshot['plugin_packages']
    assert record['name'] == 'company-audit'
    assert record['distribution'] == 'company-audit'
    assert record['version'] == '0.4.2'
    assert record['registrations']['model_api'] == [f'{prefix}_model_api']
    # The recorded snapshot can be re-run as a config; the generated record is not an input field.
    assert parse_task_config(snapshot).plugins == ['company-audit']


def test_task_config_without_plugins_records_nothing(tmp_path: Path) -> None:
    TaskConfig(model='dummy', eval_type='mock_llm', datasets=['gsm8k']).dump_yaml(str(tmp_path))

    snapshot = yaml_to_dict(str(tmp_path / 'task_config.yaml'))
    assert snapshot['plugins'] == []
    assert 'plugin_packages' not in snapshot


def test_perf_arguments_load_plugins_and_record_versions(fake_plugins) -> None:
    prefix = fake_plugins.prefix
    fake_plugins.install('company-audit', full_plugin_source(prefix), distribution='company-audit', version='0.4.2')

    args = PerfArguments(
        plugins=['company-audit'], model='dummy', api=f'{prefix}_perf_api', url='http://127.0.0.1:8000/v1/chat'
    )
    record = args.to_dict()

    assert record['plugins'] == ['company-audit']
    assert record['plugin_packages'][0]['version'] == '0.4.2'
    assert record['plugin_packages'][0]['registrations']['perf_api'] == [f'{prefix}_perf_api']
    assert 'plugin_packages' not in PerfArguments(model='dummy', url='http://127.0.0.1:8000/v1').to_dict()
    # A recorded benchmark_args.json can be replayed as input.
    assert PerfArguments(**record).plugins == ['company-audit']


def test_unpickled_config_reloads_plugins_in_a_fresh_worker(fake_plugins, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = fake_plugins.prefix
    fake_plugins.install('company-audit', full_plugin_source(prefix))
    config = TaskConfig(
        plugins=['company-audit'], model='dummy', eval_type=f'{prefix}_model_api', datasets=['gsm8k'], dry_run=True
    )
    payload = pickle.dumps(config)

    # Simulate a spawned worker: nothing is loaded and unpickling skips TaskConfig validation.
    tracked = plugins._tracked_registries()
    for kind, mapping in tracked:
        for key in [key for key in dict.keys(mapping) if key.startswith(prefix)]:
            dict.__delitem__(mapping, key)
    monkeypatch.setattr(plugins, '_loaded', {})
    for name in [name for name in list(__import__('sys').modules) if name.startswith(prefix)]:
        __import__('sys').modules.pop(name)
    assert f'{prefix}_model_api' not in MODEL_APIS

    run_task(pickle.loads(payload))

    assert f'{prefix}_model_api' in MODEL_APIS
    assert [info.name for info in plugins.loaded_plugins()] == ['company-audit']
