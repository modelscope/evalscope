"""Execution-completeness accounting for native evaluations."""

from collections import defaultdict
from typing import Dict, List

from evalscope.api.dataset import Dataset
from evalscope.api.metric import SampleScore
from evalscope.report import ExecutionSubset, ExecutionSummary


class ExecutionTracker:
    """Track failed work items independently from benchmark aggregation semantics."""

    def __init__(self) -> None:
        self._errors_by_subset: Dict[str, int] = defaultdict(int)

    def record_error(self, subset: str) -> None:
        """Record one work item that could not produce a usable sample score."""
        self._errors_by_subset[subset] += 1

    def summarize(
        self, dataset_dict: Dict[str, Dataset], sample_scores_by_subset: Dict[str, List[SampleScore]]
    ) -> ExecutionSummary:
        """Return report-ready completion counts after scoring has finished."""
        subsets = {
            subset: ExecutionSubset(
                requested=len(dataset),
                succeeded=len(sample_scores_by_subset.get(subset, [])),
                errored=self._errors_by_subset.get(subset, 0),
            )
            for subset, dataset in dataset_dict.items()
        }
        requested = sum(summary.requested for summary in subsets.values())
        succeeded = sum(summary.succeeded for summary in subsets.values())
        errored = sum(summary.errored for summary in subsets.values())
        return ExecutionSummary(
            requested=requested,
            succeeded=succeeded,
            errored=errored,
            incomplete=succeeded < requested,
            subsets=subsets,
        )
