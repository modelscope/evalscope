# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from evals.constants import MetricMembers
from evals.evaluate import Evaluate
from evals.metrics.code_metric import run_code_eval
from evals.utils.utils import jsonl_to_list
from evals.utils.logger import get_logger

logger = get_logger()


class CodeEvaluate(Evaluate):
    """
    CodeEvaluate is used to evaluate coding skill of LLMs.

    Args:
        metrics: A list of metrics to evaluate. e.g. ['code_pass_k']

    """

    def __init__(self, metrics: list, **kwargs):
        super(CodeEvaluate, self).__init__(metrics=metrics, **kwargs)

    def eval_samples(self, data_list: list):

        # TODO: registry to be added
        for metric in self.metrics:
            if metric == MetricMembers.CODE_PASS_K.value:
                k = self.kwargs.pop('k', 4)
                md_level = self.kwargs.pop('md_level', 2)
                run_code_eval(data_list, k, md_level)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

    def run(self, prompts: Union[str, list]) -> list:
        """
        Load the predicted samples and evaluate them for coding task.
        """
        # TODO: res to be added
        res_list = []

        if isinstance(prompts, str):
            prompts = jsonl_to_list(prompts)
        self.eval_samples(prompts)
        logger.info(f"Coding evaluation finished.")

        return res_list
