# Copyright (c) Alibaba, Inc. and its affiliates.

from evals import Eval


class EvalTask(object):

    def __init__(self, eval_cls: Eval):
        self._eval_cls = eval_cls

        if not isinstance(self._eval_cls, Eval):
            raise TypeError('eval_cls must be an instance of evals.Eval')

    def run(self):
        ...

    def get_samples(self):
        ...

    def get_model(self):
        ...

    def get_batches(self):
        ...

    def run_inference(self):
        ...

    def gen_report(self):
        ...

    def dump_result(self):
        ...
