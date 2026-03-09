# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Batch review executor for benchmarks that use batch scoring.

When a benchmark sets ``use_batch_scoring = True``, per-sample metrics are
first computed individually (via an injected ``review_fn``), then refined
through the benchmark's ``batch_calculate_metrics`` in fixed-size windows.
``BatchReviewer`` encapsulates this two-pass loop so that ``DefaultEvaluator``
does not need to embed the batch-execution details inline.
"""

from typing import TYPE_CHECKING, Callable, List, Optional

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.utils.function_utils import run_in_threads_with_progress
from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.evaluator import CacheManager
    from evalscope.config import TaskConfig

logger = get_logger()


class BatchReviewer:
    """
    Executes the two-pass batch review loop for a single subset.

    Pass 1 – run ``review_fn`` (per-sample metric calculation) in parallel
    using a thread pool, producing a preliminary :class:`SampleScore` per
    task state.

    Pass 2 – iterate over the preliminary scores in windows of
    ``task_config.judge_worker_num`` and call
    ``benchmark.batch_calculate_metrics`` to refine them.  Each refined score
    is immediately persisted to the review cache.

    Args:
        benchmark: The data adapter that provides ``batch_calculate_metrics``
            and ``save_metadata``.
        cache_manager: Used to persist each refined score to disk.
        task_config: Supplies ``judge_worker_num`` for parallelism / batch
            window size.
    """

    def __init__(
        self,
        benchmark: 'DataAdapter',
        cache_manager: 'CacheManager',
        task_config: 'TaskConfig',
    ):
        self.benchmark = benchmark
        self.cache_manager = cache_manager
        self.task_config = task_config

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def review_subset(
        self,
        subset: str,
        task_states: List[TaskState],
        review_fn: Callable[[TaskState], Optional[SampleScore]],
    ) -> List[SampleScore]:
        """
        Run the full batch review pipeline for one subset.

        Args:
            subset: Subset name, used for cache routing.
            task_states: All task states that need batch review.
            review_fn: Callable that computes a preliminary
                :class:`SampleScore` for a single :class:`TaskState`.
                Typically ``DefaultEvaluator._review_task_state``.

        Returns:
            Final :class:`SampleScore` list in input order.
        """
        if not task_states:
            return []

        logger.info(
            f'Batch reviewing {len(task_states)} task states for subset: {subset} '
            f'(judge_worker_num={self.task_config.judge_worker_num})'
        )

        # Pass 1 – parallel per-sample preliminary scoring
        pre_scores: List[Optional[SampleScore]] = run_in_threads_with_progress(
            task_states,
            review_fn,
            desc=f'Reviewing[{subset}]',
            max_workers=self.task_config.judge_worker_num,
            log_interval=HEARTBEAT_INTERVAL_SEC,
        )

        def on_result(ts: TaskState, score: SampleScore) -> None:
            self.cache_manager.save_review_cache(
                subset=subset,
                task_state=ts,
                sample_score=score,
                save_metadata=self.benchmark.save_metadata,
            )

        # Pass 2 – batch refinement + cache persistence
        return self._review_task_states(
            task_states=task_states,
            reviewed_scores=pre_scores,
            on_result=on_result,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _review_task_states(
        self,
        task_states: List[TaskState],
        reviewed_scores: List[Optional[SampleScore]],
        on_result: Callable[[TaskState, SampleScore], None],
    ) -> List[SampleScore]:
        """
        Pass 2 of the batch review pipeline.

        Filters out ``None`` preliminary scores, then iterates in windows of
        ``judge_worker_num`` calling ``benchmark.batch_calculate_metrics``.
        Each refined score is handed to ``on_result`` for persistence.

        Args:
            task_states: Task states corresponding to ``reviewed_scores``.
            reviewed_scores: Preliminary per-sample scores from Pass 1.
            on_result: Callback invoked with ``(task_state, sample_score)``
                after each batch window completes.

        Returns:
            Refined :class:`SampleScore` list (``None`` entries preserved at
            their original positions for invalid samples).
        """
        valid_indices = [i for i, score in enumerate(reviewed_scores) if score is not None]
        if not valid_indices:
            return reviewed_scores

        valid_states = [task_states[i] for i in valid_indices]
        valid_scores = [reviewed_scores[i] for i in valid_indices]

        all_reviewed_scores: List[SampleScore] = []
        total = len(valid_states)
        batch_size = self.task_config.judge_worker_num

        with tqdm(total=total, desc='Scoring[batch]', unit='sample', logger=logger) as pbar:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch_states = valid_states[start:end]
                batch_scores = valid_scores[start:end]

                updated_scores = self.benchmark.batch_calculate_metrics(
                    task_states=batch_states,
                    sample_scores=batch_scores,
                )

                all_reviewed_scores.extend(updated_scores)

                for ts, score in zip(batch_states, updated_scores):
                    on_result(ts, score)

                pbar.update(len(batch_states))

        return all_reviewed_scores
