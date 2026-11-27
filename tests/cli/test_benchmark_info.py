# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evalscope.api.registry import get_benchmark
from evalscope.cli.benchmark_info import BenchmarkInfoCMD


def test_markdown_fallback_uses_adapter_metadata(capsys) -> None:
    adapter = get_benchmark('mmlu')

    with patch('evalscope.utils.doc_utils.load_benchmark_data', return_value={}):
        BenchmarkInfoCMD(Namespace())._display_markdown(adapter)

    output = capsys.readouterr().out
    assert '# MMLU' in output
    assert '## Overview' in output


@pytest.mark.parametrize('tag,expected', [(None, ['alpha', 'beta']), (['coding'], ['beta']), (['missing'], [])])
def test_list_json_is_parseable_and_filters_tags(capsys, tag, expected) -> None:
    from evalscope.api.registry import BENCHMARK_REGISTRY

    def meta(tags: list[str]) -> SimpleNamespace:
        return SimpleNamespace(tags=tags, data_adapter=None, pretty_name=None, metric_list=['acc'], subset_list=['test'])

    with patch.dict(BENCHMARK_REGISTRY, {'beta': meta(['Coding']), 'alpha': meta([])}, clear=True):
        BenchmarkInfoCMD(Namespace(tag=tag, format='json'))._list_benchmarks()

    records = json.loads(capsys.readouterr().out)
    assert [record['name'] for record in records] == expected
    if records:
        assert records[0]['metrics'] == 1
        assert records[0]['subsets'] == 1
        assert isinstance(records[0]['tags'], list)


def test_list_markdown_uses_a_markdown_table(capsys) -> None:
    from evalscope.api.registry import BENCHMARK_REGISTRY

    meta = SimpleNamespace(tags=['Coding'], data_adapter=None, pretty_name='Demo', metric_list=[], subset_list=[])
    with patch.dict(BENCHMARK_REGISTRY, {'demo': meta}, clear=True):
        BenchmarkInfoCMD(Namespace(tag=None, format='markdown'))._list_benchmarks()
    output = capsys.readouterr().out
    assert '| Name' in output
    assert '| demo' in output
    assert 'Total: 1 benchmarks' in output


def test_list_default_preserves_text_table(capsys) -> None:
    from evalscope.api.registry import BENCHMARK_REGISTRY

    meta = SimpleNamespace(tags=[], data_adapter=None, pretty_name=None, metric_list=[], subset_list=[])
    with patch.dict(BENCHMARK_REGISTRY, {'demo': meta}, clear=True):
        BenchmarkInfoCMD(Namespace(tag=None, format='text'))._list_benchmarks()
    output = capsys.readouterr().out
    assert 'demo' in output
    assert 'Other' in output
    assert 'Total: 1 benchmarks' in output
    assert '| Name' not in output
