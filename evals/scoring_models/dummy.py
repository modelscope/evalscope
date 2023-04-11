# Copyright (c) Alibaba, Inc. and its affiliates.

from evals.evaluate import Evaluate
from evals.utils.utils import jsonl_to_list


class DummyEvaluate(Evaluate):
    """
    DummyEvaluate is used to evaluate the dummy task of LLMs.
    """

    def __init__(self, metrics: list, **kwargs):
        super(DummyEvaluate, self).__init__(metrics=metrics, **kwargs)

    def eval_samples(self):
        ...

    def run(self, predicted_samples_file: str):
        """
        Nothing to do with evaluating but load the predicted samples to list.
        """
        return jsonl_to_list(predicted_samples_file)
