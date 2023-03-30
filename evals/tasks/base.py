# Copyright (c) Alibaba, Inc. and its affiliates.

from evals import Eval
from evals.predictors import Predictor


class EvalTask(object):

    def __init__(self, predictor: Predictor, eval_cls: Eval, **kwargs):
        # TODO: task_name to be added in registry

        self._predictor = predictor
        self._eval_cls = eval_cls

        if not isinstance(self._predictor, Predictor):
            raise TypeError('predictor must be an instance of evals.predictors.Predictor')

        if not isinstance(self._eval_cls, Eval):
            raise TypeError('eval_cls must be an instance of evals.Eval')

    def run(self):

        # 1. get samples

        # 2. get_model meta info

        # 3. get batches (or add_batches) --P1

        # 4. run inference
        result_dict = self.run_inference(**input_args)

        # 5. gen report

        # 6. dump result

        ...

    def get_samples(self):
        ...

    def get_model(self):
        ...

    def get_batches(self):
        ...

    def run_inference(self, **input_args) -> dict:
        result_dict = self._predictor(**input_args)
        return result_dict

    def gen_report(self):
        ...

    def dump_result(self):
        ...
