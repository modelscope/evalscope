# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from llmuses.constants import MetricMembers
from llmuses.evaluate import Evaluate
from llmuses.metrics.math_accuracy import run_math_eval
from llmuses.utils.logger import get_logger
from llmuses.utils.utils import jsonl_to_list

logger = get_logger()


class MathEvaluate(Evaluate):
    """
    MathEvaluate is used to evaluate the math task of LLMs.

    Args:
        metrics: A list of metrics to evaluate. e.g. ['math_accuracy']

    """

    def __init__(self, metrics: list, **kwargs):
        super(MathEvaluate, self).__init__(metrics=metrics, **kwargs)

    def eval_samples(self, data_list: list):

        # TODO: registry to be added
        for metric in self.metrics:
            if metric == MetricMembers.MATH_ACCURACY.value:
                md_level = self.kwargs.pop('md_level', 2)
                run_math_eval(data_list, md_level=md_level)
            else:
                raise ValueError(f'Unsupported metric: {metric}')

    def run(self, prompts: Union[str, list]) -> list:
        """
        Load the predicted samples and evaluate them for math task.
        """
        # TODO: res to be added
        res_list = []

        if isinstance(prompts, str):
            prompts = jsonl_to_list(prompts)
        self.eval_samples(prompts)
        logger.info('Math evaluation finished.')

        return res_list
