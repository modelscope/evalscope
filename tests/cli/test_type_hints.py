# Copyright (c) Alibaba, Inc. and its affiliates.

from argparse import ArgumentParser
from typing import get_type_hints

import pytest

from evalscope.cli.base import ArgumentParserWithSubParsers, CLICommand
from evalscope.cli.benchmark_info import BenchmarkInfoCMD
from evalscope.cli.start_app import StartAppCMD
from evalscope.cli.start_eval import EvalCMD
from evalscope.cli.start_perf import PerfBenchCMD
from evalscope.cli.start_service import ServiceCMD


@pytest.mark.parametrize('command', [CLICommand, BenchmarkInfoCMD, StartAppCMD, EvalCMD, PerfBenchCMD, ServiceCMD])
def test_cli_define_args_type_hints_are_runtime_resolvable(command: type[CLICommand]) -> None:
    assert get_type_hints(command.define_args)['parsers'] is ArgumentParserWithSubParsers


def test_subparser_protocol_type_hints_are_runtime_resolvable() -> None:
    assert get_type_hints(ArgumentParserWithSubParsers.add_parser)['return'] is ArgumentParser
