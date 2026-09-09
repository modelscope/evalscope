# Copyright (c) Alibaba, Inc. and its affiliates.

from argparse import Namespace
from unittest.mock import patch

from evalscope.api.registry import get_benchmark
from evalscope.cli.benchmark_info import BenchmarkInfoCMD


def test_markdown_fallback_uses_adapter_metadata(capsys) -> None:
    adapter = get_benchmark('mmlu')

    with patch('evalscope.utils.doc_utils.load_benchmark_data', return_value={}):
        BenchmarkInfoCMD(Namespace())._display_markdown(adapter)

    output = capsys.readouterr().out
    assert '# MMLU' in output
    assert '## Overview' in output
