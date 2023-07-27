# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from llmuses.constants import MetricMembers
from llmuses.evaluate import Evaluate
from llmuses.metrics.rouge_metric import run_rouge_eval
from llmuses.utils.logger import get_logger
from llmuses.utils.utils import jsonl_to_list

logger = get_logger()


class CommonGenerationEvaluate(Evaluate):
    """
    GenerationEval is a subclass of Eval, which is used to evaluate common generation tasks of LLMs,
    like machine translation, summarization, poetry generating etc.

    Args:
        metrics: A list of metrics to evaluate. e.g. ['rouge']
    """

    def __init__(self, metrics: list, **kwargs):
        super(CommonGenerationEvaluate, self).__init__(
            metrics=metrics, **kwargs)
        logger.info(f'input config kwargs: {kwargs}')

    def eval_samples(self, data_list: list):

        # TODO: registry to be added
        for metric in self.metrics:
            if metric == MetricMembers.ROUGE.value:
                md_level = self.kwargs.pop('md_level', 2)
                report_metric_key = self.kwargs.pop('report_metric_key',
                                                    'rouge-l-f')
                run_rouge_eval(
                    data_list,
                    md_level=md_level,
                    report_metric_key=report_metric_key)
            else:
                raise ValueError(f'Unsupported metric: {metric}')

    def run(self, prompts: Union[str, list]) -> list:
        """
        Load the predicted samples and evaluate them for common generating task.
        """
        # TODO: res to be added
        res_list = []

        if isinstance(prompts, str):
            prompts = jsonl_to_list(prompts)
        self.eval_samples(prompts)
        logger.info('Common generating evaluation finished.')

        return res_list
