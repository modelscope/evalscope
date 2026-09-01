# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


class CLICommand(ABC):
    """
    Base class for command line tool.

    """

    @staticmethod
    @abstractmethod
    def define_args(parsers: '_SubParsersAction[ArgumentParser]'):
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        raise NotImplementedError()
