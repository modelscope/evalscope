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
from typing import TYPE_CHECKING, Callable, Dict, List

from evalscope.api.dataset import Dataset, Sample
from evalscope.api.evaluator import CacheManager, Evaluator, TaskState
from evalscope.api.metric import AggScore, SampleScore
from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.report import Report, gen_table
from evalscope.utils.function_utils import run_in_threads_with_progress
from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

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
        logger.info(f'Start loading benchmark dataset: {self.benchmark_name}')
        dataset_dict = self.benchmark.load_dataset()
        agg_score_dict = defaultdict(list)

        # Process each subset (e.g., test, validation) independently
        subset_list = list(dataset_dict.keys())
        logger.info(f'Start evaluating {len(dataset_dict)} subsets of the {self.benchmark_name}: {subset_list}')
        for subset, dataset in tqdm(
            dataset_dict.items(), desc=f'Evaluating [{self.benchmark_name}]', unit='subset', logger=logger
        ):
            if len(dataset) == 0:
                logger.info(f'No samples found in subset: {subset}, skipping.')
                continue
            logger.info(f'Evaluating subset: {subset}')
            subset_score = self.evaluate_subset(subset, dataset)
            agg_score_dict[subset] = subset_score

        # Generate the report based on aggregated scores
        logger.info('Generating report...')
        report = self.get_report(agg_score_dict)

        # Finalize the evaluation process
        self.finalize()
        logger.info(f'Benchmark {self.benchmark_name} evaluation finished.')
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
        logger.info(f'Getting predictions for subset: {subset}')
        task_states = self.get_answers(subset, dataset)

        # Calculate evaluation metrics for each prediction
        logger.info(f'Getting reviews for subset: {subset}')
        sample_scores = self.get_reviews(subset, task_states)

        # Aggregate individual sample scores into subset-level metrics
        logger.info(f'Aggregating scores for subset: {subset}')
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
            cached_task_state_list, dataset = self.cache_manager.filter_prediction_cache(subset, dataset)
        else:
            cached_task_state_list = []

        # Get output directory for storing model predictions
        model_prediction_dir = os.path.dirname(self.cache_manager.get_prediction_cache_path(subset))

        # Convert dataset to list for parallel processing
        dataset_list = list(dataset)
        if not dataset_list:
            return cached_task_state_list

        logger.info(f'Processing {len(dataset_list)} samples, if data is large, it may take a while.')

        def worker(sample: Sample) -> TaskState:
            return self._predict_sample(sample, model_prediction_dir)

        def on_result(sample: Sample, task_state: TaskState) -> None:
            model_result = self.cache_manager.save_prediction_cache(subset, task_state, self.benchmark.save_metadata)
            logger.debug(f'Model result: \n{model_result.pretty_print()}')

        def on_error(sample: Sample, exc: Exception) -> None:
            tb_str = traceback.format_exc()
            logger.error(f'{sample.model_dump_json(indent=2)} prediction failed: due to {exc}\nTraceback:\n{tb_str}')
            if self.task_config.ignore_errors:
                logger.warning('Error ignored, continuing with next sample.')
                return
            raise exc

        finished_task_states = run_in_threads_with_progress(
            dataset_list,
            worker,
            desc=f'Predicting[{self.benchmark_name}@{subset}]: ',
            max_workers=self.task_config.eval_batch_size,
            log_interval=HEARTBEAT_INTERVAL_SEC,
            on_result=on_result,
            on_error=on_error,
            filter_none_results=True,
        )

        logger.info(f'Finished getting predictions for subset: {subset}.')
        return cached_task_state_list + finished_task_states

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
            cached_score_list, task_states = self.cache_manager.filter_review_cache(subset, task_states)
        else:
            # Init a clean sample score list
            cached_score_list = []
            self.cache_manager.delete_review_cache(subset)

        if not task_states:
            return cached_score_list

        logger.info(f'Reviewing {len(task_states)} samples, if data is large, it may take a while.')

        def worker(task_state: TaskState) -> SampleScore:
            return self._review_task_state(task_state)

        def on_result(task_state: TaskState, sample_score: SampleScore) -> None:
            review_result = self.cache_manager.save_review_cache(
                subset=subset,
                task_state=task_state,
                sample_score=sample_score,
                save_metadata=self.benchmark.save_metadata
            )
            logger.debug(f'Review result: \n{review_result.pretty_print()}')

        def on_error(task_state: TaskState, exc: Exception) -> None:
            tb_str = traceback.format_exc()
            logger.error(f'Error when review sample {task_state.sample_id}: due to {exc}\nTraceback:\n{tb_str}')
            if self.task_config.ignore_errors:
                logger.warning('Error ignored, continuing with next sample.')
                return
            raise exc

        # Run reviews in parallel
        reviewed_scores = run_in_threads_with_progress(
            task_states,
            worker,
            desc=f'Reviewing[{self.benchmark_name}@{subset}]: ',
            max_workers=self.task_config.judge_worker_num,
            log_interval=HEARTBEAT_INTERVAL_SEC,
            on_error=on_error,
            # Do not persist interim results when batch scoring is enabled
            on_result=None if self.benchmark.use_batch_scoring else on_result,
            filter_none_results=False,
        )

        # Batch calculate metrics if supported by the benchmark
        if self.benchmark.use_batch_scoring:
            reviewed_scores = self._batch_review_task_states(
                task_states=task_states, reviewed_scores=reviewed_scores, on_result=on_result
            )

        logger.info(f'Finished reviewing subset: {subset}. Total reviewed: {len(reviewed_scores)}')
        return cached_score_list + reviewed_scores

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

    def _batch_review_task_states(
        self, task_states: List[TaskState], reviewed_scores: List[SampleScore],
        on_result: Callable[[TaskState, SampleScore], None]
    ) -> List[SampleScore]:
        valid_indices = [i for i, score in enumerate(reviewed_scores) if score is not None]
        if not valid_indices:
            return reviewed_scores

        task_states = [task_states[i] for i in valid_indices]
        reviewed_scores = [reviewed_scores[i] for i in valid_indices]

        # Iterate in batches with progress bar
        all_reviewed_scores = []
        total = len(task_states)
        batch_size = self.task_config.judge_worker_num
        with tqdm(total=total, desc='Scoring (batch)', unit='sample', logger=logger) as pbar:
            for start in range(0, total, batch_size):
                # Process batch
                end = min(start + batch_size, total)
                batch_task_states = task_states[start:end]
                batch_scores = reviewed_scores[start:end]
                # Batch calculate metrics
                updated_reviewed_scores = self.benchmark.batch_calculate_metrics(
                    task_states=batch_task_states, sample_scores=batch_scores
                )
                # Append results
                all_reviewed_scores.extend(updated_reviewed_scores)
                # Save each result to cache
                for task_state, sample_score in zip(batch_task_states, updated_reviewed_scores):
                    on_result(task_state, sample_score)

                pbar.update(len(batch_task_states))
        return all_reviewed_scores

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
