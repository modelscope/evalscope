# Copyright (c) Alibaba, Inc. and its affiliates.
from evals import Eval, ModelMeta


class UnitTestEval(Eval):
    """
    Unit test methods for evaluating capabilities of LLM models, like coding skills, etc.
    """

    def __init__(self, eval_name: str, model_meta: ModelMeta, predicted_samples: str, **kwargs):
        super().__init__(eval_name, model_meta, predicted_samples, **kwargs)

    def get_predicted_samples(self):
        pass

    def get_metrics(self):
        pass

    def eval_samples(self):
        pass

    def run(self):
        pass

    def unit_test_coding(self):
        pass
