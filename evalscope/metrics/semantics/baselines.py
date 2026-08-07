"""Common metric semantics baselines.

``SEMANTIC_BASELINES`` is the shared vocabulary the benchmark catalog and the legacy mapping
build on: a benchmark entry references a baseline by key and only overrides the fields that
actually differ, so the 200+ benchmark declarations stay one line each.

Conventions enforced here:

- Keys equal the ``semantic_id`` of the declaration, named ``{domain}.{concept}.{unit}``.
- ``comparison_group`` mirrors the ``{domain}.{concept}`` prefix of ``semantic_id``. It only
  means "may sit in the same comparison matrix" and never implies that the members can be
  averaged: nothing in the contract aggregates across benchmarks.
- Diagnostic baselines always use ``direction=none`` and ``comparison_group=None``.
- Quality baselines default to ``role=primary``. A benchmark that reports several of them
  (F1 plus Precision / Recall, WER plus CER) downgrades the extra ones to ``auxiliary`` in its
  own ``MetricEntry``.
- Bounded ratios in [0, 1] render as percent with ``display_multiplier=100``; official 0-100
  scales render as percent with ``display_multiplier=1``.
"""

from typing import Dict

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricRole, MetricSemantics, ValueRange

#: Value range of a ratio metric.
_RATIO_RANGE = ValueRange(min=0.0, max=1.0)

#: Value range of an official 0-100 point scale.
_POINTS_100_RANGE = ValueRange(min=0.0, max=100.0)

#: Decimals used by every percent-rendered baseline.
_PERCENT_PRECISION = 1

SEMANTIC_BASELINES: Dict[str, MetricSemantics] = {
    # --- quality: bounded ratios, higher is better ---------------------------------------
    'quality.accuracy.ratio': MetricSemantics(
        semantic_id='quality.accuracy.ratio',
        metric_name='Accuracy',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.accuracy',
    ),
    'quality.f1.ratio': MetricSemantics(
        semantic_id='quality.f1.ratio',
        metric_name='F1',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.f1',
    ),
    'quality.precision.ratio': MetricSemantics(
        semantic_id='quality.precision.ratio',
        metric_name='Precision',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.precision',
    ),
    'quality.recall.ratio': MetricSemantics(
        semantic_id='quality.recall.ratio',
        metric_name='Recall',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.recall',
    ),
    'quality.exact_match.ratio': MetricSemantics(
        semantic_id='quality.exact_match.ratio',
        metric_name='Exact Match',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.exact_match',
    ),
    'quality.pass_at_k.ratio': MetricSemantics(
        semantic_id='quality.pass_at_k.ratio',
        metric_name='Pass@k',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.pass_at_k',
    ),
    'quality.score.ratio': MetricSemantics(
        semantic_id='quality.score.ratio',
        metric_name='Score',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.score',
    ),
    'quality.coverage.ratio': MetricSemantics(
        semantic_id='quality.coverage.ratio',
        metric_name='Coverage',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.coverage',
    ),
    'quality.win_rate.ratio': MetricSemantics(
        semantic_id='quality.win_rate.ratio',
        metric_name='Win Rate',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.win_rate',
    ),
    'quality.iou.ratio': MetricSemantics(
        semantic_id='quality.iou.ratio',
        metric_name='IoU',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.iou',
    ),
    # --- quality: text generation overlap and similarity ----------------------------------
    'quality.bleu.ratio': MetricSemantics(
        semantic_id='quality.bleu.ratio',
        metric_name='BLEU',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.bleu',
    ),
    'quality.rouge.ratio': MetricSemantics(
        semantic_id='quality.rouge.ratio',
        metric_name='ROUGE',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.rouge',
    ),
    'quality.meteor.ratio': MetricSemantics(
        semantic_id='quality.meteor.ratio',
        metric_name='METEOR',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.meteor',
    ),
    'quality.similarity.ratio': MetricSemantics(
        semantic_id='quality.similarity.ratio',
        metric_name='Similarity',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.similarity',
    ),
    #: CIDEr is a consensus score that is not bounded by 1, so it renders as a plain number.
    'quality.cider.unbounded': MetricSemantics(
        semantic_id='quality.cider.unbounded',
        metric_name='CIDEr',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=None,
        display_kind=MetricDisplayKind.NUMBER,
        display_unit=None,
        display_precision=3,
        comparison_group='quality.cider',
    ),
    # --- quality: bounded error rates, lower is better -----------------------------------
    'quality.wer.ratio': MetricSemantics(
        semantic_id='quality.wer.ratio',
        metric_name='WER',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.LOWER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.wer',
    ),
    'quality.cer.ratio': MetricSemantics(
        semantic_id='quality.cer.ratio',
        metric_name='CER',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.LOWER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.cer',
    ),
    'quality.mer.ratio': MetricSemantics(
        semantic_id='quality.mer.ratio',
        metric_name='MER',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.LOWER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.mer',
    ),
    #: Share of failed or hallucinated outcomes: a graded result, unlike the diagnostic
    #: parse-status shares, so it keeps a direction.
    'quality.error_rate.ratio': MetricSemantics(
        semantic_id='quality.error_rate.ratio',
        metric_name='Error Rate',
        role=MetricRole.AUXILIARY,
        direction=MetricDirection.LOWER_IS_BETTER,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.error_rate',
    ),
    # --- quality: official scales and unbounded judge scores ------------------------------
    'quality.score.points_100': MetricSemantics(
        semantic_id='quality.score.points_100',
        metric_name='Score',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=_POINTS_100_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=1.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group='quality.score',
    ),
    'quality.judge_score.unbounded': MetricSemantics(
        semantic_id='quality.judge_score.unbounded',
        metric_name='Judge Score',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=None,
        display_kind=MetricDisplayKind.NUMBER,
        display_unit=None,
        display_precision=2,
        comparison_group='quality.judge_score',
    ),
    #: Score assigned by a scoring model (aesthetic / preference / alignment scorers) whose
    #: scale is defined by the model rather than by the benchmark.
    'quality.model_score.unbounded': MetricSemantics(
        semantic_id='quality.model_score.unbounded',
        metric_name='Model Score',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        value_range=None,
        display_kind=MetricDisplayKind.NUMBER,
        display_unit=None,
        display_precision=4,
        comparison_group='quality.model_score',
    ),
    # --- perf: latency, lower is better ---------------------------------------------------
    'perf.latency.seconds': MetricSemantics(
        semantic_id='perf.latency.seconds',
        metric_name='Latency',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.LOWER_IS_BETTER,
        raw_unit='s',
        display_kind=MetricDisplayKind.NUMBER,
        display_unit='s',
        display_precision=3,
        comparison_group='perf.latency',
    ),
    'perf.latency.milliseconds': MetricSemantics(
        semantic_id='perf.latency.milliseconds',
        metric_name='Latency',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.LOWER_IS_BETTER,
        raw_unit='ms',
        display_kind=MetricDisplayKind.NUMBER,
        display_unit='ms',
        display_precision=2,
        comparison_group='perf.latency',
    ),
    # --- perf: throughput, higher is better -----------------------------------------------
    'perf.throughput.tokens_per_second': MetricSemantics(
        semantic_id='perf.throughput.tokens_per_second',
        metric_name='Token Throughput',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_unit='tok/s',
        display_kind=MetricDisplayKind.NUMBER,
        display_unit='tok/s',
        display_precision=2,
        comparison_group='perf.throughput',
    ),
    'perf.throughput.requests_per_second': MetricSemantics(
        semantic_id='perf.throughput.requests_per_second',
        metric_name='Request Throughput',
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_unit='req/s',
        display_kind=MetricDisplayKind.NUMBER,
        display_unit='req/s',
        display_precision=2,
        comparison_group='perf.throughput',
    ),
    # --- diagnostic: never carries a direction nor a comparison group ----------------------
    'diagnostic.count.items': MetricSemantics(
        semantic_id='diagnostic.count.items',
        metric_name='Count',
        role=MetricRole.DIAGNOSTIC,
        direction=MetricDirection.NONE,
        display_kind=MetricDisplayKind.NUMBER,
        display_unit=None,
        display_precision=0,
        comparison_group=None,
    ),
    'diagnostic.parse_status.ratio': MetricSemantics(
        semantic_id='diagnostic.parse_status.ratio',
        metric_name='Parse Status',
        role=MetricRole.DIAGNOSTIC,
        direction=MetricDirection.NONE,
        value_range=_RATIO_RANGE,
        display_kind=MetricDisplayKind.PERCENT,
        display_multiplier=100.0,
        display_unit='%',
        display_precision=_PERCENT_PRECISION,
        comparison_group=None,
    ),
    'diagnostic.unspecified': MetricSemantics(
        semantic_id='diagnostic.unspecified',
        metric_name='Unspecified',
        role=MetricRole.DIAGNOSTIC,
        direction=MetricDirection.NONE,
        display_kind=MetricDisplayKind.NUMBER,
        display_unit=None,
        display_precision=4,
        comparison_group=None,
    ),
}
"""Baseline identifier -> semantics. Keys equal the declared ``semantic_id``."""

__all__ = ['SEMANTIC_BASELINES']
