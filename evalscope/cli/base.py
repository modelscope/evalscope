# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from argparse import ArgumentParser


class ArgumentParserWithSubParsers(Protocol):
    def add_parser(self, name, **kwargs) -> 'ArgumentParser': ...


class CLICommand(ABC):
    """
    Base class for command line tool.

    """

    @staticmethod
    @abstractmethod
    def define_args(parsers: 'ArgumentParserWithSubParsers'):
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        raise NotImplementedError()
