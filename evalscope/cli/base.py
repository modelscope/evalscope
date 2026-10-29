# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Any, Protocol


class ArgumentParserWithSubParsers(Protocol):
    def add_parser(self, name: str, **kwargs: Any) -> ArgumentParser: ...


class CLICommand(ABC):
    """
    Base class for command line tool.

    """

    @staticmethod
    @abstractmethod
    def define_args(parsers: ArgumentParserWithSubParsers) -> None:
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        raise NotImplementedError()
