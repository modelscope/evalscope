# Copyright (c) Alibaba, Inc. and its affiliates.
"""Preserve text-rendering counts from individual scores through the final report."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from evalscope.api.metric import AggScore, SampleScore
from evalscope.metrics import macro_mean
from evalscope.report import Report
from evalscope.report.report import normalize_score

CorpusMetricSpec = Union[Tuple[str, ...], Dict[str, str]]
"""Either a tuple of metric names (aggregate every row corpus-level), or a mapping of
metric name -> ``'components'`` (only the report rows carrying a ``component`` dimension
are aggregated corpus-level; the overall row keeps the inherited mean)."""


class CorpusAggregationMixin:
    """Aggregate successful text-rendering scores without process-global side channels."""

    corpus_metrics: CorpusMetricSpec = ()
    doc_eval_type = 'text2image'

    def _corpus_metric_mode(self, metric_name: str) -> str:
        if isinstance(self.corpus_metrics, dict):
            return self.corpus_metrics.get(metric_name, '')
        return 'all' if metric_name in self.corpus_metrics else ''

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        aggregates = super().aggregate_scores(sample_scores)
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        component_grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for sample in sample_scores:
            if not sample.score.status.is_usable:
                continue
            metadata = sample.score.metadata or {}
            metric_metadata = metadata.get('metric_metadata', {})
            category = self._dimension_value((sample.sample_metadata or {}).get('category', ''))
            for metric_name in list(self.corpus_metrics):
                if metric_name not in sample.score.value:
                    continue
                counts = metric_metadata.get(metric_name, {}).get('corpus_counts')
                if counts is None:
                    raise ValueError(
                        f'Missing corpus counts for {metric_name}; rerun evaluation with rerun_review=True.'
                    )
                if counts.get('denominator', 0) <= 0 and not counts.get('components'):
                    raise ValueError(f'Invalid corpus denominator for {metric_name}.')
                mode = self._corpus_metric_mode(metric_name)
                if mode == 'components':
                    # Component mode: aggregate each component corpus-level; rows without
                    # a matching component (e.g. the overall row) keep the inherited mean.
                    for component, component_counts in (counts.get('components') or {}).items():
                        if component_counts['denominator'] <= 0:
                            raise ValueError(f'Invalid corpus denominator for {metric_name}:{component}.')
                        component_grouped[(metric_name, component, '')].append(component_counts)
                        if category:
                            component_grouped[(metric_name, component, category)].append(component_counts)
                    continue
                grouped[(metric_name, '')].append(counts)
                if category:
                    grouped[(metric_name, category)].append(counts)

        for aggregate in aggregates:
            mode = self._corpus_metric_mode(aggregate.metric_name)
            if not mode:
                continue
            dimensions = aggregate.dimensions or {}
            if mode == 'components':
                if 'component' not in dimensions:
                    continue
                counts = component_grouped[
                    (aggregate.metric_name, dimensions['component'], dimensions.get('category', ''))
                ]
            else:
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
            mode = self._corpus_metric_mode(metric.identity.name)
            if not mode:
                continue
            if mode == 'components':
                # Component identities re-aggregate through the standard flow below (their
                # aggregate rows already carry the final corpus counts); the overall
                # identity keeps the inherited mean of per-image ratios.
                if 'component' not in metric.identity.dimensions:
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
