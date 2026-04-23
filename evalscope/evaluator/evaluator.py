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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from evalscope.api.dataset import Dataset, Sample
from evalscope.api.evaluator import CacheManager, Evaluator, TaskState
from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.registry import register_evaluator
from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.evaluator.batch_reviewer import BatchReviewer
from evalscope.evaluator.perf_collector import PerfCollector
from evalscope.report import Report, gen_perf_table, gen_table
from evalscope.utils.function_utils import run_in_threads_with_progress
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.model import Model
    from evalscope.config import TaskConfig
    from evalscope.utils.io_utils import OutputsStructure

logger = get_logger()


@dataclass
class _WorkItem:
    """
    A work item for the unified evaluation pool.

    Represents a single unit of work in the evaluation pipeline:
    - If ``task_state`` is None: run full predict + review for ``sample``.
    - If ``task_state`` is provided: prediction is already cached; run review only.
    """

    subset: str
    """Subset this item belongs to."""

    sample: Optional[Sample] = None
    """Sample to predict. Set when prediction is required."""

    task_state: Optional[TaskState] = None
    """Cached task state. Set when only review is required."""

    @property
    def needs_predict(self) -> bool:
        """True when prediction has not yet been computed."""
        return self.task_state is None


@dataclass
class _PoolContext:
    """
    Carries state from :meth:`~DefaultEvaluator._collect_work_items` through
    the evaluation phases, acting as the shared data contract between phases.

    Attributes:
        work_items: Work items to process in the unified pool.
        cached_scores_by_subset: Fully cached (predict + review) scores per subset.
        review_pending_by_subset: TaskStates whose prediction is cached but review
            is still needed; populated only for ``use_batch_scoring`` benchmarks.
        model_prediction_dir: Shared parent directory for prediction JSONL files.
        total_cached: Number of samples already fully cached (skipped in pool).
    """

    work_items: List[_WorkItem]
    cached_scores_by_subset: Dict[str, List[SampleScore]]
    review_pending_by_subset: Dict[str, List[TaskState]]
    model_prediction_dir: str
    total_cached: int

    @property
    def grand_total(self) -> int:
        """Total sample count across all subsets: cached + work-pool items."""
        return self.total_cached + len(self.work_items)


@register_evaluator('default')
class DefaultEvaluator(Evaluator):
    """
    Default Evaluator for running evaluations on benchmarks.

    This evaluator handles the complete evaluation pipeline:
    1. Loading datasets from all subsets
    2. Flattening all subsets into a single unified work pool
    3. Running model inference and metric calculation atomically per sample
    4. Writing results to per-subset JSONL cache as each sample completes
    5. Aggregating scores and generating the final report

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

        # Initialize batch reviewer for benchmarks that use batch scoring
        self.batch_reviewer = BatchReviewer(
            benchmark=benchmark,
            cache_manager=self.cache_manager,
            task_config=task_config,
        )

        # Initialize PerfCollector for collecting per-request performance metrics
        self.perf_collector = PerfCollector()

    def eval(self) -> Report:
        """
        Run the complete evaluation process.

        All subsets are merged into a **single unified work pool** so slow
        samples in one subset never block samples in another.  Each sample is
        predicted and reviewed atomically; results are persisted to per-subset
        JSONL caches as they complete.

        Returns:
            Report: The complete evaluation report.
        """
        logger.info(f'Start loading benchmark dataset: {self.benchmark_name}')
        dataset_dict = {k: v for k, v in self.benchmark.load_dataset().items() if len(v) > 0}

        if not dataset_dict:
            logger.warning(f'No samples found in any subset of {self.benchmark_name}. Skipping.')
            self.finalize()
            return {}

        subset_list = list(dataset_dict.keys())
        logger.info(f'Start evaluating {len(dataset_dict)} subsets of {self.benchmark_name}: {subset_list}')

        # Phase 1 – build unified work pool from all subsets
        context = self._collect_work_items(dataset_dict)
        logger.info(
            f'Unified pool: {len(context.work_items)} items to process, '
            f'{context.total_cached} already fully cached '
            f'({context.grand_total} total across all subsets).'
        )

        # Phase 2 – execute unified thread pool (single progress bar)
        results_by_subset = self._run_pool(context)
        logger.info(f'Unified pool finished for {self.benchmark_name}.')

        # Phase 3 – aggregate scores per subset (batch review happens here too)
        agg_score_dict = self._aggregate_scores(dataset_dict, context, results_by_subset)

        # Phase 4 – generate report
        if not agg_score_dict:
            logger.warning(
                f'No valid scores generated for {self.benchmark_name} '
                '(all samples filtered or empty subsets). Skipping report generation.'
            )
            report = {}
        else:
            logger.info('Generating report...')
            report = self.get_report(agg_score_dict)

        self.finalize()
        logger.info(f'Benchmark {self.benchmark_name} evaluation finished.')
        return report

    # ------------------------------------------------------------------ #
    # Phase helpers                                                        #
    # ------------------------------------------------------------------ #

    def _collect_work_items(self, dataset_dict: Dict[str, Dataset]) -> _PoolContext:
        """
        Phase 1 – classify every sample across all subsets into a work tier.

        Each sample falls into exactly one of three tiers:

        1. **Fully cached** (predict + review on disk) → skipped; its score is
           stored in ``cached_scores_by_subset``.
        2. **Predict cached, review pending** → review-only :class:`_WorkItem`
           (or placed in ``review_pending_by_subset`` for batch-scoring).
        3. **Uncached** → full predict + review :class:`_WorkItem`.

        Args:
            dataset_dict: Mapping of subset name → dataset.

        Returns:
            A :class:`_PoolContext` ready for :meth:`_run_pool` and
            :meth:`_aggregate_scores`.
        """
        work_items: List[_WorkItem] = []
        cached_scores_by_subset: Dict[str, List[SampleScore]] = defaultdict(list)
        review_pending_by_subset: Dict[str, List[TaskState]] = defaultdict(list)

        for subset, dataset in dataset_dict.items():
            cached_pred_states, remaining_dataset = (
                self.cache_manager.filter_prediction_cache(subset, dataset) if self.use_cache else ([], dataset)
            )

            if self.benchmark.use_batch_scoring:
                # Prediction runs in the pool; review is deferred until all
                # task_states for this subset are available (after pool).
                for sample in remaining_dataset:
                    work_items.append(_WorkItem(subset=subset, sample=sample))

                if self.use_cache and not self.task_config.rerun_review:
                    cached_scores, need_review = self.cache_manager.filter_review_cache(subset, cached_pred_states)
                    cached_scores_by_subset[subset].extend(cached_scores)
                    review_pending_by_subset[subset].extend(need_review)
                else:
                    self.cache_manager.delete_review_cache(subset)
                    review_pending_by_subset[subset].extend(cached_pred_states)

            else:
                # Predict + review happen atomically per sample inside the pool.
                if self.use_cache and not self.task_config.rerun_review:
                    cached_scores, need_review = self.cache_manager.filter_review_cache(subset, cached_pred_states)
                    cached_scores_by_subset[subset].extend(cached_scores)
                    for ts in need_review:  # Tier 2: review-only items
                        work_items.append(_WorkItem(subset=subset, task_state=ts))
                else:
                    self.cache_manager.delete_review_cache(subset)
                    for ts in cached_pred_states:  # Prediction cached, review cleared
                        work_items.append(_WorkItem(subset=subset, task_state=ts))

                for sample in remaining_dataset:  # Tier 3: full predict+review
                    work_items.append(_WorkItem(subset=subset, sample=sample))

        model_prediction_dir = os.path.dirname(self.cache_manager.get_prediction_cache_path(next(iter(dataset_dict))))
        total_cached = sum(len(v) for v in cached_scores_by_subset.values())

        return _PoolContext(
            work_items=work_items,
            cached_scores_by_subset=cached_scores_by_subset,
            review_pending_by_subset=review_pending_by_subset,
            model_prediction_dir=model_prediction_dir,
            total_cached=total_cached,
        )

    def _run_pool(self, context: _PoolContext) -> Dict[str, List[Tuple[TaskState, Optional[SampleScore]]]]:
        """
        Phase 2 – execute the unified work pool under a single progress bar.

        Each item is processed by :meth:`_process_work_item`; results are
        immediately persisted by :meth:`_persist_result` and accumulated
        into a per-subset bucket for downstream aggregation.

        Args:
            context: Pool context produced by :meth:`_collect_work_items`.

        Returns:
            Mapping of subset name → ``(task_state, sample_score)`` pairs in
            completion order.  ``sample_score`` is ``None`` for batch-scoring
            benchmarks (review is deferred to :meth:`_aggregate_scores`).
        """
        results_by_subset: Dict[str, List[Tuple[TaskState, Optional[SampleScore]]]] = \
            defaultdict(list)

        def worker(item: _WorkItem) -> Tuple[TaskState, Optional[SampleScore]]:
            return self._process_work_item(item, context.model_prediction_dir)

        def on_result(item: _WorkItem, result: Tuple[TaskState, Optional[SampleScore]]) -> None:
            self._persist_result(item, *result)
            results_by_subset[item.subset].append(result)

        def on_error(item: _WorkItem, exc: Exception) -> None:
            tb_str = traceback.format_exc()
            logger.error(f'Processing item in subset={item.subset!r} failed: {exc}\nTraceback:\n{tb_str}')
            if self.task_config.ignore_errors:
                logger.warning('Error ignored, continuing with next sample.')
                return
            raise exc

        run_in_threads_with_progress(
            context.work_items,
            worker,
            desc=f'Evaluating[{self.benchmark_name}]',
            max_workers=self.task_config.eval_batch_size,
            log_interval=HEARTBEAT_INTERVAL_SEC,
            on_result=on_result,
            on_error=on_error,
            skip_failed=True,
            initial=context.total_cached,
            total=context.grand_total,
        )
        return results_by_subset

    def _process_work_item(self, item: _WorkItem, model_prediction_dir: str) -> Tuple[TaskState, Optional[SampleScore]]:
        """
        Process a single work item: predict (if needed) then review.

        Called concurrently inside the thread pool by :meth:`_run_pool`.
        Override this method to inject custom logic around inference or scoring.

        Args:
            item: The work item to process.
            model_prediction_dir: Directory for storing prediction output files.

        Returns:
            ``(task_state, sample_score)`` where ``sample_score`` is ``None``
            for batch-scoring benchmarks (review deferred).
        """
        task_state = (
            self._predict_sample(item.sample, model_prediction_dir) if item.needs_predict else item.task_state
        )
        sample_score = (None if self.benchmark.use_batch_scoring else self._review_task_state(task_state))
        return task_state, sample_score

    def _persist_result(
        self,
        item: _WorkItem,
        task_state: TaskState,
        sample_score: Optional[SampleScore],
    ) -> None:
        """
        Persist a completed item’s results to the on-disk cache.

        Called in the **main thread** by :meth:`_run_pool` immediately after
        each item completes (no concurrent writes).  Override to add custom
        persistence logic.

        Args:
            item: The originating work item.
            task_state: The completed task state (prediction output).
            sample_score: The review score, or ``None`` for batch-scoring.
        """
        if item.needs_predict:
            model_result = self.cache_manager.save_prediction_cache(
                item.subset, task_state, self.benchmark.save_metadata
            )
            logger.debug(f'Model result: \n{model_result.pretty_print()}')

        if sample_score is not None:
            review_result = self.cache_manager.save_review_cache(
                subset=item.subset,
                task_state=task_state,
                sample_score=sample_score,
                save_metadata=self.benchmark.save_metadata,
            )
            logger.debug(f'Review result: \n{review_result.pretty_print()}')

        # Collect per-request performance metrics for live inference items only
        # when perf collection is enabled.
        if self.task_config.collect_perf and item.needs_predict:
            perf = task_state.output.perf_metrics if task_state.output is not None else None
            if perf is not None:
                self.perf_collector.record(perf)

    def _aggregate_scores(
        self,
        dataset_dict: Dict[str, Dataset],
        context: _PoolContext,
        results_by_subset: Dict[str, List[Tuple[TaskState, Optional[SampleScore]]]],
    ) -> Dict[str, List[AggScore]]:
        """
        Phase 3 – aggregate per-sample scores into subset-level metrics.

        For standard benchmarks the pool scores are combined with cached scores
        and passed directly to ``benchmark.aggregate_scores``.

        For batch-scoring benchmarks :meth:`BatchReviewer.review_subset` is invoked
        first to produce final per-sample scores from all collected task states.

        Args:
            dataset_dict: Subset iteration order.
            context: Pool context from :meth:`_collect_work_items`.
            results_by_subset: Per-subset pool results from :meth:`_run_pool`.

        Returns:
            Mapping of subset name → aggregated scores.  Empty subsets omitted.
        """
        agg_score_dict: Dict[str, List[AggScore]] = {}

        for subset in dataset_dict:
            cached_scores = context.cached_scores_by_subset.get(subset, [])
            pool_results = results_by_subset.get(subset, [])

            if self.benchmark.use_batch_scoring:
                pending = context.review_pending_by_subset.get(subset, [])
                new_task_states = [ts for ts, _ in pool_results]
                batch_scores = self.batch_reviewer.review_subset(
                    subset, pending + new_task_states, review_fn=self._review_task_state
                )
                all_scores = cached_scores + batch_scores
            else:
                new_scores = [sc for _, sc in pool_results if sc is not None]
                all_scores = cached_scores + new_scores

            if not all_scores:
                logger.info(f'No valid scores generated for subset: {subset}, skipping.')
                continue

            logger.info(f'Aggregating scores for subset: {subset}')
            agg_score_dict[subset] = self.benchmark.aggregate_scores(sample_scores=all_scores)

        return agg_score_dict

    def _predict_sample(self, sample: Sample, model_prediction_dir: str) -> TaskState:
        """
        Run model inference on a single sample.

        Args:
            sample: The sample to predict.
            model_prediction_dir: Directory for storing model predictions.

        Returns:
            TaskState: The task state containing the prediction result.
        """
        logger.debug(f'\n{sample.pretty_print()}')
        task_state = self.benchmark.run_inference(model=self.model, sample=sample, output_dir=model_prediction_dir)
        return task_state

    def _review_task_state(self, task_state: TaskState) -> SampleScore:
        """
        Compute evaluation metrics for a single task state.

        Args:
            task_state: The task state to review.

        Returns:
            SampleScore: The evaluation score for the task state.
        """
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
            analysis = report.generate_analysis(self.task_config)
            logger.info(f'Report analysis:\n{analysis}')
        else:
            logger.info('Skipping report analysis (`analysis_report=False`).')

        # Inject perf metrics into the report when collect_perf is enabled
        if self.task_config.collect_perf:
            report.perf_metrics = self.perf_collector.get_perf_dict() or None

        # Save the complete report to file
        report.to_json(report_file)
        logger.info(f'Dump report to: {report_file} \n')

        # Print per-benchmark perf table when perf data is available
        if self.task_config.collect_perf and report.perf_metrics:
            try:
                perf_table = gen_perf_table(report_list=[report])
                if perf_table:
                    logger.info(f'\n{self.benchmark_name} perf table:\n{perf_table}\n')
            except Exception:
                logger.error('Failed to generate perf table.')

        return report

    def finalize(self, *args, **kwargs):
        self.benchmark.finalize(*args, **kwargs)
