# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from llmuses.evaluate import Evaluate
from llmuses.utils.utils import jsonl_to_list


class DummyEvaluate(Evaluate):
    """
    DummyEvaluate is used to evaluate the dummy task of LLMs.
    """

    def __init__(self, metrics: list, **kwargs):
        super(DummyEvaluate, self).__init__(metrics=metrics, **kwargs)

    def eval_samples(self):
        ...

    def run(self, prompts: Union[str, list]):
        """
        Nothing to do with evaluating but load the predicted samples to list.
        """
        if isinstance(prompts, list):
            return prompts
        return jsonl_to_list(prompts)
