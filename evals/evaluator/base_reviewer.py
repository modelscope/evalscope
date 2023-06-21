# Copyright (c) Alibaba, Inc. and its affiliates.


class BaseReviewer(object):

    def __init__(self, **kwargs):
        ...

    def run(self, *args, **kwargs):
        """
        Run pairwise battles with given models.
        """
        raise NotImplementedError(
            'run() method must be implemented in your subclass.')
