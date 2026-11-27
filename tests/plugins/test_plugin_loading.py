# Copyright (c) Alibaba, Inc. and its affiliates.
"""Opt-in loading of plugins declared in the ``evalscope.plugins`` entry point group."""

import argparse
import sys
import threading

import pytest

from evalscope import plugins
from evalscope.api.registry import BENCHMARK_REGISTRY, METRIC_REGISTRY, MODEL_APIS
from evalscope.arguments import check_eval_type
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.registry import ApiRegistry
from evalscope.plugins import (
    PluginConflictError,
    PluginLoadError,
    PluginNotFoundError,
    add_plugin_argument,
    available_plugins,
    load_plugins,
)

from .conftest import full_plugin_source


def test_installed_plugin_is_imported_only_when_enabled(fake_plugins) -> None:
    prefix = fake_plugins.prefix
    module = fake_plugins.install('company-audit', full_plugin_source(prefix), distribution='company-audit', version='0.4.2')

    installed = {info.name: info for info in available_plugins()}
    assert installed['company-audit'].distribution == 'company-audit'
    assert installed['company-audit'].version == '0.4.2'
    assert module not in sys.modules
    assert f'{prefix}_model_api' not in MODEL_APIS

    [info] = load_plugins(['company-audit'])

    assert module in sys.modules
    assert info.distribution == 'company-audit' and info.version == '0.4.2' and info.module == module
    assert info.registrations == {
        'benchmark': [f'{prefix}_benchmark'],
        'model_api': [f'{prefix}_model_api'],
        'metric': [f'{prefix}_metric'],
        'perf_api': [f'{prefix}_perf_api'],
    }
    assert f'{prefix}_model_api' in MODEL_APIS
    assert f'{prefix}_metric' in METRIC_REGISTRY
    assert BENCHMARK_REGISTRY.is_registered(f'{prefix}_benchmark')
    assert ApiRegistry.get_class(f'{prefix}_perf_api').__module__ == module


def test_other_installed_plugins_stay_unloaded(fake_plugins) -> None:
    enabled = fake_plugins.install('enabled', 'ENABLED = True\n')
    skipped = fake_plugins.install('skipped', 'raise RuntimeError("must not be imported")\n')

    load_plugins(['enabled'])

    assert enabled in sys.modules
    assert skipped not in sys.modules
    assert [info.name for info in plugins.loaded_plugins()] == ['enabled']


def test_loading_is_idempotent_and_thread_safe(fake_plugins) -> None:
    prefix = fake_plugins.prefix
    # Re-running the registrations would raise, so a second import would surface as a conflict.
    fake_plugins.install('once', f"from evalscope.api.registry import register_model_api\nregister_model_api('{prefix}_once')(object)\n")
    errors = []

    def load() -> None:
        try:
            load_plugins(['once'])
        except Exception as e:  # pragma: no cover - reported through the assertion below
            errors.append(e)

    threads = [threading.Thread(target=load) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert load_plugins(['once', 'once'])[0].registrations == {'model_api': [f'{prefix}_once']}


def test_unknown_plugin_lists_installed_ones(fake_plugins) -> None:
    fake_plugins.install('company-audit', 'X = 1\n')

    with pytest.raises(PluginNotFoundError, match="Plugin 'missing' is not installed.*company-audit"):
        load_plugins(['missing'])


def test_import_failure_is_reported_and_rolled_back(fake_plugins) -> None:
    prefix = fake_plugins.prefix
    module = fake_plugins.install(
        'broken',
        f"""
from evalscope.api.registry import register_model_api
from evalscope.perf.plugin.registry import register_dataset

register_model_api('{prefix}_partial')(object)
register_dataset('{prefix}_partial_dataset')(object)
raise RuntimeError('missing provider SDK')
""",
        distribution='broken-plugin',
        version='9.9.9',
    )

    with pytest.raises(PluginLoadError) as excinfo:
        load_plugins(['broken'])

    message = str(excinfo.value)
    assert "plugin 'broken' (broken-plugin 9.9.9" in message
    assert 'RuntimeError: missing provider SDK' in message
    assert not isinstance(excinfo.value, PluginConflictError)
    assert f'{prefix}_partial' not in MODEL_APIS
    assert module not in sys.modules
    assert plugins.loaded_plugins() == []


@pytest.mark.parametrize(
    'source, expected',
    [
        ("from evalscope.api.registry import register_model_api\nregister_model_api('openai_api')(object)\n",
         "Model API 'openai_api' is already registered"),
        ("from evalscope.api.registry import register_metric\nregister_metric(name='acc')(object)\n",
         "Metric 'acc' is already registered"),
        ("from evalscope.perf.plugin.registry import register_api\nregister_api('openai')(object)\n",
         "perf_api 'openai'"),
    ],
    ids=['model-api', 'metric', 'perf-api'],
)
def test_name_conflict_with_builtin_is_rejected(fake_plugins, source: str, expected: str) -> None:
    fake_plugins.install('clashing', source, distribution='clashing-plugin', version='1.0.0')

    with pytest.raises(PluginConflictError, match=expected) as excinfo:
        load_plugins(['clashing'])

    assert "plugin 'clashing' (clashing-plugin 1.0.0" in str(excinfo.value)
    assert ApiRegistry.get_class('openai') is OpenaiPlugin


def test_benchmark_conflict_with_builtin_is_rejected(fake_plugins) -> None:
    fake_plugins.install(
        'clashing',
        """
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.registry import register_benchmark

@register_benchmark(BenchmarkMeta(name='gsm8k', dataset_id='unused'))
class Shadow(DefaultDataAdapter):
    pass
""",
    )

    with pytest.raises(PluginConflictError, match='Benchmark gsm8k already registered'):
        load_plugins(['clashing'])


def test_name_conflict_between_plugins_is_rejected(fake_plugins) -> None:
    prefix = fake_plugins.prefix
    source = f"from evalscope.perf.plugin.registry import register_api\nregister_api('{prefix}_shared')(type('P', (), {{}}))\n"
    fake_plugins.install('first', source, distribution='first-plugin')
    fake_plugins.install('second', source, distribution='second-plugin')

    load_plugins(['first'])
    with pytest.raises(PluginConflictError, match=f"perf_api '{prefix}_shared'"):
        load_plugins(['second'])

    assert ApiRegistry.get_class(f'{prefix}_shared').__module__ == fake_plugins.module_name('first')


def test_entry_point_name_provided_twice_is_ambiguous(fake_plugins) -> None:
    fake_plugins.install('dup', 'X = 1\n', distribution='dist-a')
    fake_plugins.install('dup', 'X = 2\n', distribution='dist-b', module=fake_plugins.module_name('other'))

    with pytest.raises(PluginLoadError, match="'dup' is provided by more than one distribution"):
        load_plugins(['dup'])


def test_plugin_flag_accepts_repeats_and_lists() -> None:
    parser = argparse.ArgumentParser()
    add_plugin_argument(parser)

    assert parser.parse_args([]).plugins is None
    assert parser.parse_args(['--plugin', 'a', '--plugin', 'b', 'c']).plugins == ['a', 'b', 'c']


def test_eval_type_accepts_model_api_registered_by_enabled_plugin(fake_plugins) -> None:
    prefix = fake_plugins.prefix
    fake_plugins.install('company-audit', full_plugin_source(prefix))

    with pytest.raises(ValueError, match='invalid choice'):
        check_eval_type(f'{prefix}_model_api')
    load_plugins(['company-audit'])
    check_eval_type(f'{prefix}_model_api')
    check_eval_type('openai_api')
    check_eval_type(None)


@pytest.mark.parametrize('names', [[''], ['  '], [1]])
def test_invalid_plugin_names_are_rejected(names) -> None:
    with pytest.raises(ValueError, match='non-empty strings'):
        load_plugins(names)
