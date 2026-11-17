# Copyright (c) Alibaba, Inc. and its affiliates.
"""Corpus-level aggregation mixin for text-rendering benchmarks.

The official evaluators aggregate corpus-level (total numerator over total denominator
across all samples), not as the mean of per-image ratios. Metrics stash their per-image
``(numerator, denominator)`` on the metric singleton keyed by image path
(see ``_CountsStashMixin``); this mixin carries them through ``Score.metadata`` and
re-aggregates the report rows per the official protocol, without touching unrelated T2I
metrics.
"""

from typing import Any, Dict, List

from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.registry import get_metric
from evalscope.constants import FileConstants
from evalscope.utils.function_utils import thread_safe


class CorpusAggregationMixin:
    """Attach per-image counts in ``match_score`` and re-aggregate corpus-level.

    Subclasses declare ``corpus_metrics``: registered metric name -> canonical report
    name. Metric scores whose denominator is 0 (or whose counts are missing, e.g. when
    the metric errored) keep the inherited mean-of-ratios aggregation.
    """

    corpus_metrics: Dict[str, str] = {}

    @thread_safe
    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state) -> Any:
        score = super().match_score(original_prediction, filtered_prediction, reference, task_state)
        image_path = (task_state.metadata or {}).get(FileConstants.IMAGE_PATH, original_prediction)
        corpus_counts = score.metadata.setdefault('corpus_counts', {})
        for metric_name, canonical_name in self.corpus_metrics.items():
            # The metric is a SingletonMetric: read its initialized instance (if any) for the
            # counts of the image just scored; skip when the metric never ran.
            instance = getattr(get_metric(metric_name), '_instance', None)
            if instance is None:
                continue
            counts = instance.counts_for(image_path)
            if counts is not None:
                corpus_counts[canonical_name] = counts
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        agg_scores = super().aggregate_scores(sample_scores)
        corpus_totals: Dict[str, List[float]] = {}
        corpus_totals_by_category: Dict[str, Dict[str, List[float]]] = {}
        for sample_score in sample_scores:
            counts = (sample_score.score.metadata or {}).get('corpus_counts') or {}
            category = self._dimension_value((sample_score.sample_metadata or {}).get('category', ''))
            for metric_name, (numerator, denominator) in counts.items():
                totals = corpus_totals.setdefault(metric_name, [0.0, 0.0])
                totals[0] += numerator
                totals[1] += denominator
                if category:
                    by_cat = corpus_totals_by_category.setdefault(metric_name, {})
                    cat_totals = by_cat.setdefault(category, [0.0, 0.0])
                    cat_totals[0] += numerator
                    cat_totals[1] += denominator
        for agg in agg_scores:
            if agg.metric_name not in self.corpus_metrics.values():
                continue
            if 'category' in (agg.dimensions or {}):
                totals = corpus_totals_by_category.get(agg.metric_name, {}).get(agg.dimensions['category'])
            else:
                totals = corpus_totals.get(agg.metric_name)
            if totals and totals[1] > 0:
                agg.score = totals[0] / totals[1]
        return agg_scores
