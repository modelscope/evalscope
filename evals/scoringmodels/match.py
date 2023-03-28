# Copyright (c) Alibaba, Inc. and its affiliates.
from evals import Eval, ModelMeta


class MatchEval(Eval):
    def __init__(self, eval_name: str, model_meta: ModelMeta, predicted_samples: str, **kwargs):
        super().__init__(eval_name, model_meta, predicted_samples, **kwargs)

    def get_predicted_samples(self):
        pass

    def get_metrics(self):
        pass

    def eval_single_sample(self):
        pass

    def eval_all_samples(self):
        pass

    def run(self):
        pass
