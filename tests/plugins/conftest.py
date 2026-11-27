# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fixtures that install fake plugin distributions on ``sys.path`` for a single test."""

import importlib
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Iterator, Optional

import pytest

from evalscope import plugins


class FakePlugins:
    """Writes importable plugin modules plus ``*.dist-info`` metadata under one directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.prefix = f'fake_evalscope_plugin_{uuid.uuid4().hex[:8]}'

    def module_name(self, suffix: str) -> str:
        """Return a module name unique to this test, so imports never leak across tests."""
        return f'{self.prefix}_{suffix}'

    def install(
        self,
        entry_point: str,
        source: str,
        *,
        distribution: str = 'fake-plugin',
        version: str = '1.2.3',
        module: Optional[str] = None,
    ) -> str:
        """Install a distribution whose ``evalscope.plugins`` entry point imports ``source``."""
        module = module or self.module_name(entry_point.replace('-', '_'))
        (self.root / f'{module}.py').write_text(textwrap.dedent(source), encoding='utf-8')
        self.declare(entry_point, module, distribution=distribution, version=version)
        return module

    def declare(self, entry_point: str, module: str, *, distribution: str, version: str) -> None:
        """Write ``*.dist-info`` metadata declaring an entry point for an existing module."""
        dist_info = self.root / f'{distribution.replace("-", "_")}_{uuid.uuid4().hex[:6]}-{version}.dist-info'
        dist_info.mkdir()
        (dist_info / 'METADATA').write_text(
            f'Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n', encoding='utf-8'
        )
        (dist_info / 'entry_points.txt').write_text(
            f'[{plugins.ENTRY_POINT_GROUP}]\n{entry_point} = {module}\n', encoding='utf-8'
        )
        importlib.invalidate_caches()


@pytest.fixture
def fake_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakePlugins]:
    """Isolate plugin state: fresh load cache, registries and modules restored after the test."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(plugins, '_loaded', {})
    tracked = plugins._tracked_registries()
    before = plugins._snapshot(tracked)
    modules_before = set(sys.modules)
    factory = FakePlugins(tmp_path)
    try:
        yield factory
    finally:
        plugins._rollback(tracked, before)
        for name in set(sys.modules) - modules_before:
            if name.startswith(factory.prefix) or name.startswith('evalscope_example_plugin'):
                sys.modules.pop(name, None)
        importlib.invalidate_caches()


#: Source of a plugin that registers one component of every kind the dashboard offers.
FULL_PLUGIN = '''
from typing import List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.metric import Metric
from evalscope.api.registry import register_benchmark, register_metric, register_model_api
from evalscope.models.mockllm import MockLLM
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.registry import register_api


@register_model_api('{prefix}_model_api')
class FakeModelAPI(MockLLM):
    pass


@register_metric(name='{prefix}_metric')
class FakeMetric(Metric):
    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        return [1.0 for _ in predictions]


@register_benchmark(
    BenchmarkMeta(
        name='{prefix}_benchmark',
        evaluation_version='v1.0',
        pretty_name='Fake Plugin Benchmark',
        dataset_id='unused',
        description='## Overview\\nA fake benchmark.\\n\\n## Evaluation Notes\\n- Always scores 1.',
        metric_list=['{prefix}_metric'],
    )
)
class FakeAdapter(DefaultDataAdapter):
    def record_to_sample(self, record) -> Sample:
        return Sample(input=record['q'], target=record['a'])


@register_api('{prefix}_perf_api')
class FakePerfPlugin(OpenaiPlugin):
    pass
'''


def full_plugin_source(prefix: str) -> str:
    """Return :data:`FULL_PLUGIN` with registered names made unique by ``prefix``."""
    return FULL_PLUGIN.replace('{prefix}', prefix)
