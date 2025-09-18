# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Default evaluator implementation for running benchmark evaluations.

This module provides the DefaultEvaluator class which orchestrates the entire
evaluation process including data loading, model inference, metric calculation,
and report generation.
"""

import os
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from tqdm import tqdm
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from evalscope.api.dataset import Dataset, DatasetDict, Sample
from evalscope.api.evaluator import CacheManager, Evaluator, TaskState
from evalscope.api.metric import AggScore, SampleScore
from evalscope.report import Report, gen_table
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.model import Model
    from evalscope.config import TaskConfig
    from evalscope.utils.io_utils import OutputsStructure

logger = get_logger()


class DefaultEvaluator(Evaluator):
    """
    Default Evaluator for running evaluations on benchmarks.

    This evaluator handles the complete evaluation pipeline:
    1. Loading datasets from benchmarks
    2. Running model inference on samples
    3. Calculating evaluation metrics
    4. Generating and saving reports
    5. Managing caching for predictions and reviews

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
        outputs: 'OutputsStructure',
        task_config: 'TaskConfig',
    ):
        # Store core components needed for evaluation
        self.benchmark = benchmark
        self.model = model
        self.outputs = outputs
        self.task_config = task_config

        # Extract frequently used identifiers
        self.benchmark_name = benchmark.name
        """Name of the benchmark being evaluated."""

        self.model_name = task_config.model_id
        """ID of the model being evaluated."""

        self.use_cache = task_config.use_cache
        """Whether to use cache for predictions."""

        # Initialize cache manager for storing and retrieving cached results
        self.cache_manager = CacheManager(
            outputs=outputs,
            model_name=self.model_name,
            benchmark_name=self.benchmark_name,
        )

    def eval(self) -> Report:
        """
        Run the complete evaluation process.

        This is the main entry point that orchestrates the entire evaluation:
        1. Load dataset from benchmark
        2. Evaluate each subset independently
        3. Aggregate scores across subsets
        4. Generate final evaluation report

        Returns:
            Report: The complete evaluation report containing all metrics and results.
        """
        # Load the dataset and evaluate each subset
        dataset_dict = self.benchmark.load_dataset()
        agg_score_dict = defaultdict(list)

        # Process each subset (e.g., test, validation) independently
        for subset, dataset in dataset_dict.items():
            if len(dataset) == 0:
                logger.info(f'No samples found in subset: {subset}, skipping.')
                continue
            subset_score = self.evaluate_subset(subset, dataset)
            agg_score_dict[subset] = subset_score

        # Generate the report based on aggregated scores
        report = self.get_report(agg_score_dict)

        # Finalize the evaluation process
        self.finalize()
        return report

    def evaluate_subset(self, subset: str, dataset: Dataset) -> List[AggScore]:
        """
        Evaluate a single subset of the dataset.

        This method processes one subset through the complete evaluation pipeline:
        1. Get model predictions for all samples
        2. Calculate evaluation metrics for predictions
        3. Aggregate individual sample scores

        Args:
            subset: Name of the subset being evaluated (e.g., 'test', 'validation').
            dataset: The dataset subset containing samples to evaluate.

        Returns:
            List[AggScore]: Aggregated scores for this subset.
        """
        # Get model predictions for all samples in the subset
        task_states = self.get_answers(subset, dataset)

        # Calculate evaluation metrics for each prediction
        sample_scores = self.get_reviews(subset, task_states)

        # Aggregate individual sample scores into subset-level metrics
        agg_scores = self.benchmark.aggregate_scores(sample_scores=sample_scores)
        return agg_scores

    def get_answers(self, subset: str, dataset: Dataset) -> List[TaskState]:
        """
        Get model predictions for all samples in the dataset subset.

        This method handles:
        1. Loading cached predictions if available and caching is enabled
        2. Running model inference on remaining samples in parallel
        3. Saving new predictions to cache

        Args:
            subset: Name of the subset being processed.
            dataset: The dataset subset containing samples for prediction.

        Returns:
            List[TaskState]: Task states containing model predictions for each sample.
        """
        # Initialize task state list and filter cached predictions if caching is enabled
        if self.use_cache:
            task_state_list, dataset = self.cache_manager.filter_prediction_cache(subset, dataset)
        else:
            task_state_list = []

        # Get output directory for storing model predictions
        model_prediction_dir = os.path.dirname(self.cache_manager.get_prediction_cache_path(subset))

        # Convert dataset to list for parallel processing
        dataset_list = list(dataset)

        if not dataset_list:
            return task_state_list

        # Process samples in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(dataset_list), self.task_config.eval_batch_size)) as executor:
            # Submit all prediction tasks
            future_to_sample = {
                executor.submit(self._predict_sample, sample, model_prediction_dir): sample
                for sample in dataset_list
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(dataset_list), desc=f'Predicting[{self.benchmark_name}@{subset}]: ') as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        task_state = future.result()
                        task_state_list.append(task_state)

                        # Save the prediction result to cache for future use
                        model_result = self.cache_manager.save_prediction_cache(
                            subset, task_state, self.benchmark.save_metadata
                        )
                        logger.debug(f'Model result: \n{model_result.pretty_print()}')

                    except Exception as exc:
                        tb_str = traceback.format_exc()
                        logger.error(
                            f'{sample.model_dump_json(indent=2)} prediction failed: due to {exc}\nTraceback:\n{tb_str}'
                        )
                        if self.task_config.ignore_errors:
                            logger.warning('Error ignored, continuing with next sample.')
                        else:
                            raise exc
                    finally:
                        pbar.update(1)

        return task_state_list

    def _predict_sample(self, sample: Sample, model_prediction_dir: str) -> TaskState:
        """
        Helper method to predict a single sample.

        Args:
            sample: The sample to predict.
            model_prediction_dir: Directory for storing model predictions.

        Returns:
            TaskState: The task state containing the prediction result.
        """
        logger.debug(f'\n{sample.pretty_print()}')

        # Run model inference on the current sample
        task_state = self.benchmark.run_inference(model=self.model, sample=sample, output_dir=model_prediction_dir)
        return task_state

    def get_reviews(self, subset: str, task_states: List[TaskState]) -> List[SampleScore]:
        """
        Calculate evaluation metrics for model predictions.

        This method handles:
        1. Loading cached review results if available and caching is enabled
        2. Computing metrics for remaining task states in parallel
        3. Saving new review results to cache

        Args:
            subset: Name of the subset being reviewed.
            task_states: List of task states containing model predictions.

        Returns:
            List[SampleScore]: Evaluation scores for each sample.
        """
        # Initialize sample score list and filter cached reviews if caching is enabled
        if self.use_cache and not self.task_config.rerun_review:
            sample_score_list, task_states = self.cache_manager.filter_review_cache(subset, task_states)
        else:
            # Init a clean sample score list
            sample_score_list = []
            self.cache_manager.delete_review_cache(subset)

        if not task_states:
            return sample_score_list

        # Process task states in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(task_states), self.task_config.judge_worker_num)) as executor:
            # Submit all review tasks
            future_to_task_state = {
                executor.submit(self._review_task_state, task_state): task_state
                for task_state in task_states
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(task_states), desc=f'Reviewing[{self.benchmark_name}@{subset}]: ') as pbar:
                for future in as_completed(future_to_task_state):
                    task_state = future_to_task_state[future]
                    try:
                        try:
                            sample_score = future.result()
                        except TimeoutError:
                            logger.warning(
                                f'Timeout when reviewing sample {task_state.sample_id}, setting score to zero.'
                            )
                            sample_score = SampleScore(sample_id=task_state.sample_id, scores={})
                        sample_score_list.append(sample_score)

                        # Save the review result to cache for future use
                        review_result = self.cache_manager.save_review_cache(
                            subset=subset,
                            task_state=task_state,
                            sample_score=sample_score,
                            save_metadata=self.benchmark.save_metadata
                        )
                        logger.debug(f'Review result: \n{review_result.pretty_print()}')

                    except Exception as exc:
                        tb_str = traceback.format_exc()
                        logger.error(
                            f'Error when review sample {task_state.sample_id}: due to {exc}\nTraceback:\n{tb_str}'
                        )
                        if self.task_config.ignore_errors:
                            logger.warning('Error ignored, continuing with next sample.')
                        else:
                            raise exc
                    finally:
                        pbar.update(1)

        return sample_score_list

    def _review_task_state(self, task_state: TaskState) -> SampleScore:
        """
        Helper method to review a single task state.

        Args:
            task_state: The task state to review.

        Returns:
            SampleScore: The evaluation score for the task state.
        """
        # Compute evaluation metrics using the benchmark's metric calculation
        sample_score = self.benchmark.calculate_metrics(task_state=task_state)
        return sample_score

    def get_report(self, agg_score_dict: Dict[str, List[AggScore]]) -> Report:
        """
        Generate a comprehensive evaluation report from aggregated scores.

        This method handles:
        1. Creating the evaluation report from scores
        2. Generating and displaying a summary table
        3. Optionally generating detailed analysis
        4. Saving the report to file

        Args:
            agg_score_dict: Dictionary mapping subset names to their aggregated scores.

        Returns:
            Report: The complete evaluation report.
        """
        assert agg_score_dict, 'No scores to generate report from.'

        # Get paths for saving the report
        report_path = self.cache_manager.get_report_path()
        report_file = self.cache_manager.get_report_file()

        # Generate the main evaluation report using benchmark-specific logic
        report = self.benchmark.generate_report(
            scores=agg_score_dict, model_name=self.model_name, output_dir=report_path
        )

        # Generate and display a summary table of results
        try:
            report_table = gen_table(report_list=[report], add_overall_metric=self.benchmark.add_overall_metric)
            logger.info(f'\n{self.benchmark_name} report table:'
                        f'\n{report_table} \n')
        except Exception:
            logger.error('Failed to generate report table.')

        # Generate detailed analysis if requested in configuration
        if self.task_config.analysis_report:
            logger.info('Generating report analysis, please wait ...')
            analysis = report.generate_analysis(self.task_config.judge_model_args)
            logger.info(f'Report analysis:\n{analysis}')
        else:
            logger.info('Skipping report analysis (`analysis_report=False`).')

        # Save the complete report to file
        report.to_json(report_file)
        logger.info(f'Dump report to: {report_file} \n')
        return report

    def finalize(self, *args, **kwargs):
        self.benchmark.finalize(*args, **kwargs)
