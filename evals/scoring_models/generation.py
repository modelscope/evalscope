# Copyright (c) Alibaba, Inc. and its affiliates.

from evals.evaluate import Evaluate
from evals.utils.utils import jsonl_to_list


class GenerationEvaluate(Evaluate):
    """
    GenerationEval is a subclass of Eval, which is used to evaluate the generation task of LLMs.
    One of scoring models.
    """

    def __init__(self, metrics: list, **kwargs):
        super(GenerationEvaluate, self).__init__(metrics, **kwargs)

    def eval_samples(self):
        pass

    def run(self, predicted_samples_file: str):
        res_list = []

        # TODO: to be implemented
        res_list = jsonl_to_list(predicted_samples_file)

        return res_list

