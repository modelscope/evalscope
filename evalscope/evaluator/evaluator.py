# Copyright (c) Alibaba, Inc. and its affiliates.

import abc
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from evalscope.api.dataset import Dataset, DatasetDict
from evalscope.api.evaluator import Evaluator, TaskState
from evalscope.api.evaluator.cache import ModelResult, ReviewResult
from evalscope.api.metric import AggScore, SampleScore
from evalscope.report import Report

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.model import Model
    from evalscope.config import TaskConfig
    from evalscope.utils.io_utils import OutputsStructure

from evalscope.utils.logger import get_logger

logger = get_logger()


class DefaultEvaluator(Evaluator):
    """
    Default Evaluator for running evaluations on benchmarks.

    Args:
        benchmark: The data adapter for loading and processing data.
        model: The model to be evaluated.
        outputs: The output structure for saving evaluation results.
        task_config: The task configuration.
    """

    def __init__(
        self,
        benchmark: 'DataAdapter',
        model: 'Model',
        outputs: 'OutputsStructure' = None,
        task_config: 'TaskConfig' = None,
    ):
        self.benchmark = benchmark
        self.model = model
        self.outputs = outputs
        self.task_config = task_config

    def eval(self) -> Report:
        """Run the evaluation process."""
        # Load the dataset and evaluate each subset
        dataset: DatasetDict = self.benchmark.load_dataset()
        agg_score_dict = defaultdict(list)
        for subset, data in dataset.items():
            subset_score = self.evaluate_subset(subset, data)
            agg_score_dict[subset] = subset_score
        # Generate the report based on aggregated scores
        report = self.get_report(agg_score_dict)
        return report

    def evaluate_subset(self, subset: str, data: Dataset) -> List[AggScore]:
        """Evaluate a subset of the dataset."""
        task_states = self.get_answers(subset, data)
        sample_scores = self.get_reviews(subset, task_states)
        agg_scores = self.benchmark.aggregate_scores(sample_scores=sample_scores)
        return agg_scores

    def get_answers(self, subset: str, dataset: Dataset) -> List[TaskState]:
        """Get the evaluation answers."""
        task_state_list = []
        model_result_list = []

        for sample in dataset:
            logger.debug(f'Item input: \n{sample.input}')

            task_state = self.benchmark.run_inference(model=self.model, sample=sample)
            task_state_list.append(task_state)

            #  model result are only used for saving the predictions
            model_result = ModelResult(index=task_state.sample_id, model_output=task_state.output.completion)
            logger.debug(f'Model result: {model_result.to_json_str()}')
            model_result_list.append(model_result)

        return task_state_list

    def get_reviews(self, subset: str, task_states: List[TaskState]) -> List[SampleScore]:
        """Get the review results."""
        sample_score_list = []
        review_result_list = []

        # Collect scores for each task state
        for task_state in task_states:
            sample_score = self.benchmark.calculate_metrics(task_state=task_state)
            sample_score_list.append(sample_score)

            review_result = ReviewResult(
                index=task_state.sample_id,
                input=task_state.input,
                prediction=sample_score.score.prediction,
                extracted_prediction=sample_score.score.extracted_prediction,
                gold=task_state.target,
                score=sample_score.score.value)
            logger.debug(f'Review result: {review_result.to_json_str()}')
            review_result_list.append(review_result)

        return sample_score_list

    def get_report(self, agg_score_dict: Dict[str, List[AggScore]]) -> Report:
        report = self.benchmark.generate_report(scores=agg_score_dict, model_name=self.model.model_id)
        return report
