# Copyright (c) Alibaba, Inc. and its affiliates.
"""Preserve text-rendering counts from individual scores through the final report."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from evalscope.api.metric import AggScore, SampleScore
from evalscope.metrics import macro_mean
from evalscope.report import Report
from evalscope.report.report import normalize_score


class CorpusAggregationMixin:
    """Aggregate successful text-rendering scores without process-global side channels."""

    corpus_metrics: Tuple[str, ...] = ()
    doc_eval_type = 'text2image'

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        aggregates = super().aggregate_scores(sample_scores)
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for sample in sample_scores:
            if not sample.score.status.is_usable:
                continue
            metadata = sample.score.metadata or {}
            metric_metadata = metadata.get('metric_metadata', {})
            category = self._dimension_value((sample.sample_metadata or {}).get('category', ''))
            for metric_name in self.corpus_metrics:
                if metric_name not in sample.score.value:
                    continue
                counts = metric_metadata.get(metric_name, {}).get('corpus_counts')
                if counts is None:
                    raise ValueError(
                        f'Missing corpus counts for {metric_name}; rerun evaluation with rerun_review=True.'
                    )
                if counts['denominator'] <= 0:
                    raise ValueError(f'Invalid corpus denominator for {metric_name}.')
                grouped[(metric_name, '')].append(counts)
                if category:
                    grouped[(metric_name, category)].append(counts)

        for aggregate in aggregates:
            if aggregate.metric_name not in self.corpus_metrics:
                continue
            counts = grouped[(aggregate.metric_name, aggregate.dimensions.get('category', ''))]
            numerator = sum(item['numerator'] for item in counts)
            denominator = sum(item['denominator'] for item in counts)
            aggregate.score = numerator / denominator
            aggregate.metadata = {'corpus_counts': {'numerator': numerator, 'denominator': denominator}}
        return aggregates

    def _corpus_report_score(self, counts: List[Dict[str, Any]]) -> float:
        """Combine CVTG subsets using the original word counts, before display rounding."""
        return sum(item['numerator'] for item in counts) / sum(item['denominator'] for item in counts)

    def _on_generate_report(self, scores: Dict[str, List[AggScore]], model_name: str) -> Report:
        report = super()._on_generate_report(scores, model_name)
        for metric in report.metrics:
            if metric.identity.name not in self.corpus_metrics:
                continue
            all_counts = []
            for category in metric.categories:
                counts = [
                    aggregate.metadata['corpus_counts']
                    for subset in category.subsets
                    for aggregate in scores[subset.name]
                    if aggregate.identity == metric.identity
                ]
                category.score = normalize_score(self._corpus_report_score(counts))
                all_counts.extend(counts)
            metric.score = normalize_score(self._corpus_report_score(all_counts))
            metric.macro_score = normalize_score(
                macro_mean([category for category in metric.categories if category.num])
            )
        return report
