# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any


class BaseTask(object):

    def __init__(self, **kwargs):
        ...

    def run(self, **kwargs) -> Any:
        raise NotImplementedError(
            'run() method must be implemented in your subclass.')
