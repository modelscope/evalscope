# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from math import isfinite, sqrt
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.metric import Aggregator, AggScore, SampleScore
from evalscope.api.registry import register_aggregation
from evalscope.metrics.utils.functions import calculate_pass_at_k, calculate_pass_hat_k, mean


def collect_metric_names(scores: List[SampleScore]) -> List[str]:
    """Collect metric names across all samples, in first-seen order.

    A sample may carry no value for a metric, so the name set cannot be read off ``scores[0]``.
    """
    metric_names: Dict[str, None] = {}
    for sample_score in scores:
        for metric_name in sample_score.score.value:
            metric_names.setdefault(metric_name, None)
    return list(metric_names)


def collect_planned_attempts(
    scores: List[SampleScore], metric_name: str
) -> Dict[Any, Dict[int, Optional[SampleScore]]]:
    """Keep every planned position so unavailable attempts cannot shift later trials forward."""
    grouped: Dict[Any, Dict[int, Optional[SampleScore]]] = defaultdict(dict)
    for score in scores:
        group_id = score.group_id if score.group_id is not None else score.sample_id
        position = score.generation_index
        if position is None:
            # Legacy caches lack an explicit position. Their existing row order is the only safe
            # information available; new scores always carry generation_index.
            position = len(grouped[group_id])
        grouped[group_id][position] = score if metric_name in score.score.value else None
    return grouped


def eligible_prefixes(
    grouped: Dict[Any, Dict[int, Optional[SampleScore]]],
) -> Dict[int, List[Tuple[Any, Dict[int, Optional[SampleScore]]]]]:
    """Return eligible groups for each k without extending beyond the shortest planned group."""
    max_k = min((max(attempts, default=-1) + 1 for attempts in grouped.values()), default=0)
    result: Dict[int, List[Tuple[Any, Dict[int, Optional[SampleScore]]]]] = defaultdict(list)
    for group_id, attempts in grouped.items():
        for k in range(1, max_k + 1):
            if all(attempts.get(index) is not None for index in range(k)):
                result[k].append((group_id, attempts))
            else:
                break
    return result


@register_aggregation(name='mean')
class Mean(Aggregator):
    name = 'mean'

    def agg_func(self, values: List[float]) -> float:
        return mean(values)

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate scores by computing the mean for each metric.

        Args:
            scores: List of sample scores to aggregate

        Returns:
            List of aggregated scores with mean values
        """
        if not scores:
            return []

        # Group score values by metric name
        metric_values = defaultdict(list)
        metric_sample_ids = defaultdict(list)

        for score in scores:
            for metric_name, value in score.score.value.items():
                metric_values[metric_name].append(value)
                metric_sample_ids[metric_name].append(score.sample_id)

        # Calculate mean for each metric
        aggregated_scores = []
        for metric_name, values in metric_values.items():
            if values:  # Only process non-empty value lists
                aggregated_scores.append(
                    AggScore(
                        score=self.agg_func(values),
                        metric_name=metric_name,
                        aggregation=self.name,
                        num=len(values),
                        num_correct=int(sum(values)) if all(v in (0, 1) for v in values) else None,
                        ids=metric_sample_ids[metric_name],
                    )
                )

        return aggregated_scores


METRIC_WEIGHTS_KEY = 'metric_weights'
"""``Score.metadata`` key holding ``{metric_name: weight}`` for :class:`WeightedMean`.

A weight is the number of underlying units the metric value was already averaged over inside one
sample -- instructions for IFEval-style ``inst_level_*``, for instance. Only metrics that need it
carry one, so a benchmark can mix weighted and unweighted metrics in the same ``Score``.
"""


def collect_metric_weights(score: SampleScore) -> Dict[str, int]:
    """Read valid non-negative unit counts from one sample's metadata."""
    metadata = score.score.metadata or {}
    declared = metadata.get(METRIC_WEIGHTS_KEY) or {}
    if not isinstance(declared, dict):
        return {}

    weights: Dict[str, int] = {}
    for metric_name, weight in declared.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            continue
        numeric_weight = float(weight)
        if not isfinite(numeric_weight) or numeric_weight < 0 or not numeric_weight.is_integer():
            continue
        weights[metric_name] = int(numeric_weight)
    return weights


@register_aggregation(name='weighted_mean')
class WeightedMean(Aggregator):
    """Mean using per-sample unit counts declared in ``Score.metadata``."""

    name = 'weighted_mean'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        if not scores:
            return []

        metric_values: Dict[str, List[float]] = defaultdict(list)
        metric_weights: Dict[str, List[Optional[int]]] = defaultdict(list)
        metric_sample_ids: Dict[str, List[Any]] = defaultdict(list)

        for score in scores:
            weights = collect_metric_weights(score)
            for metric_name, value in score.score.value.items():
                metric_values[metric_name].append(value)
                metric_weights[metric_name].append(weights.get(metric_name))
                metric_sample_ids[metric_name].append(score.sample_id)

        aggregated_scores = []
        for metric_name, values in metric_values.items():
            weights = metric_weights[metric_name]
            weighted = all(weight is not None for weight in weights) and any(weight is not None for weight in weights)
            if weighted:
                declared_weights = [weight for weight in weights if weight is not None]
                total_weight = sum(declared_weights)
                if total_weight == 0:
                    continue
                aggregated = sum(value * weight for value, weight in zip(values, declared_weights)) / total_weight
                aggregation = self.name
                num = total_weight
            else:
                aggregated = mean(values)
                aggregation = 'mean'
                num = len(values)
                total_weight = float(num)

            aggregated_scores.append(
                AggScore(
                    score=aggregated,
                    metric_name=metric_name,
                    aggregation=aggregation,
                    num=num,
                    ids=metric_sample_ids[metric_name],
                    metadata={'weighted': weighted, 'samples': len(values), 'total_weight': total_weight},
                )
            )

        return aggregated_scores


@register_aggregation(name='clipped_mean')
class ClippedMean(Mean):
    name = 'clipped_mean'

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def agg_func(self, values: List[float]) -> float:
        clipped_values = min(max(mean(values), self.clip_min), self.clip_max)
        return clipped_values


@register_aggregation(name='mean_and_pass_at_k')
class MeanPassAtK(Aggregator):
    def __init__(self):
        self.name = 'mean_and_pass_at_k'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Return the mean plus structured pass@n aggregates for all n <= k.

        For each metric:
        - Group scores by group_id
        - Collect binary correctness values
        - Limit k to the number of attempts available in every group
        - Compute per-group pass@n for all n from 1 to k via calculate_pass_at_k
        - Emit ``aggregation=pass_at_k`` and ``dimensions.k=n`` directly
        """
        if not scores:
            return []

        aggregated_scores = Mean()(scores)
        metrics = collect_metric_names(scores)

        for metric_name in metrics:
            group_attempts = collect_planned_attempts(scores, metric_name)
            prefixes = eligible_prefixes(group_attempts)
            if not prefixes:
                continue
            for n, eligible in prefixes.items():
                group_order = [group_id for group_id, _ in eligible]
                values_by_group = [
                    [float(attempt.score.value[metric_name]) for attempt in attempts.values() if attempt is not None]
                    for _, attempts in eligible
                ]
                values = calculate_pass_at_k(
                    [len(items) for items in values_by_group], [int(sum(items)) for items in values_by_group], n
                )
                aggregated_scores.append(
                    AggScore(
                        score=mean(values.tolist()),
                        metric_name=metric_name,
                        aggregation='pass_at_k',
                        dimensions={'k': n},
                        num=len(values),
                        ids=group_order,
                        metadata={
                            'eligible': len(values),
                            'total': len(group_attempts),
                            'coverage': len(values) / len(group_attempts),
                            'excluded': len(group_attempts) - len(values),
                        },
                    )
                )

        return aggregated_scores


@register_aggregation(name='mean_and_vote_at_k')
class MeanVoteAtK(Aggregator):
    def __init__(self):
        self.name = 'mean_and_vote_at_k'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate scores by computing vote@n for all n <= k for each metric using group_id.

        Vote@n selects the most frequent prediction among first n samples, then checks if
        that prediction is correct. This ensures vote@n has proper monotonicity properties.

        Note: vote@n computes accuracy per unique problem (one score per group_id), while
        mean_acc averages over all samples (including repeats). Therefore, vote@n can be
        higher or lower than mean_acc depending on sample ordering and repeat distribution.

        For each metric:
        - Group scores by group_id, preserving order
        - For each n from 1 to k, find most frequent prediction among first n samples
        - Check if most frequent prediction was ever marked correct (score=1.0) in those samples
        - Assign 1.0 if correct, 0.0 otherwise

        Args:
            scores: List of sample scores to aggregate

        Returns:
            List of aggregated scores with vote@n values for all n <= k
        """
        if not scores:
            return []

        aggregated_scores = Mean()(scores)
        metrics = collect_metric_names(scores)

        for metric_name in metrics:
            group_attempts = collect_planned_attempts(scores, metric_name)
            prefixes = eligible_prefixes(group_attempts)
            if not prefixes:
                continue
            for n, eligible in prefixes.items():
                vote_at_n_map: Dict[Any, float] = {}
                for group_id, attempts in eligible:
                    n_samples = [
                        (attempts[index].score.extracted_prediction, attempts[index].score.value[metric_name])
                        for index in range(n)
                    ]

                    # Count prediction frequencies, ignoring failed extractions (None or blank)
                    prediction_counts = defaultdict(int)
                    for prediction, _ in n_samples:
                        if prediction is None or (isinstance(prediction, str) and not prediction.strip()):
                            continue
                        prediction_counts[prediction] += 1

                    if not prediction_counts:
                        vote_at_n_map[group_id] = 0.0
                        continue

                    # Select most frequent prediction (ties broken by first occurrence)
                    most_frequent_pred = max(prediction_counts, key=prediction_counts.get)

                    # Check if this prediction was ever correct in the first n samples
                    is_correct = any(
                        pred == most_frequent_pred and correctness == 1.0 for pred, correctness in n_samples
                    )

                    vote_at_n_map[group_id] = 1.0 if is_correct else 0.0

                values = list(vote_at_n_map.values())
                aggregated_scores.append(
                    AggScore(
                        score=mean(values),
                        metric_name=metric_name,
                        aggregation='vote_at_k',
                        dimensions={'k': n},
                        num=len(values),
                        ids=list(vote_at_n_map),
                        metadata={
                            'eligible': len(values),
                            'total': len(group_attempts),
                            'coverage': len(values) / len(group_attempts),
                            'excluded': len(group_attempts) - len(values),
                        },
                    )
                )

        return aggregated_scores


@register_aggregation(name='mean_and_pass_hat_k')
class MeanPassHatK(Aggregator):
    def __init__(self):
        self.name = 'mean_and_pass_hat_k'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Return the mean plus structured pass^n aggregates for all n <= k.

        For each metric:
        - Group scores by group_id
        - Collect binary correctness values
        - Limit k to the number of attempts available in every group
        - Compute per-group pass^n for all n from 1 to k via calculate_pass_hat_k
        - Emit ``aggregation=pass_hat_k`` and ``dimensions.k=n`` directly
        """
        if not scores:
            return []

        aggregated_scores = Mean()(scores)
        metrics = collect_metric_names(scores)

        for metric_name in metrics:
            group_attempts = collect_planned_attempts(scores, metric_name)
            prefixes = eligible_prefixes(group_attempts)
            if not prefixes:
                continue
            for n, eligible in prefixes.items():
                values = []
                ids = []
                for group_id, attempts in eligible:
                    attempt_values = [
                        float(attempt.score.value[metric_name]) for attempt in attempts.values() if attempt is not None
                    ]
                    values.append(float(calculate_pass_hat_k(len(attempt_values), int(sum(attempt_values)), n)))
                    ids.append(group_id)
                aggregated_scores.append(
                    AggScore(
                        score=mean(values),
                        metric_name=metric_name,
                        aggregation='pass_hat_k',
                        dimensions={'k': n},
                        num=len(values),
                        ids=ids,
                        metadata={
                            'eligible': len(values),
                            'total': len(group_attempts),
                            'coverage': len(values) / len(group_attempts),
                            'excluded': len(group_attempts) - len(values),
                        },
                    )
                )

        return aggregated_scores


# --- Confidence interval support (issue: report score confidence intervals) ---

_WILSON_Z = 1.959963984540054
"""Two-sided 95% normal quantile for the Wilson score interval."""

_BOOTSTRAP_B = 2000
"""Resample count for the percentile bootstrap; the 2.5% tail is noisy at B=1000."""

_BOOTSTRAP_SEED = 0
"""Fixed seed so repeated aggregations of the same scores are bit-identical."""

_LOW_CONFIDENCE_N = 30
"""Below this sample count no interval is meaningful: flagged, not trusted."""

_WIDE_INTERVAL_PTS = 15.0
"""Interval width (in percentage points) above which a metric is flagged high-variance.

Distinct from ``_LOW_CONFIDENCE_N``: the first flag means "too little data", this one means
"enough data, but the metric itself is high-variance" — at p=0.5, n=30 already gives a
~34pt width, so this flag only has independent meaning at n >= 30."""


def _wilson_interval(values: List[float]) -> Tuple[float, float]:
    """Wilson score interval for 0/1 values, on the same 0-1 scale as the mean.

    Analytic (no resampling) and valid at p->0 and p->1, where the normal approximation
    collapses to a zero-width interval. The interval is asymmetric; render as ``[lo, hi]``.
    """
    n = len(values)
    p = sum(values) / n
    denom = 1 + _WILSON_Z * _WILSON_Z / n
    center = (p + _WILSON_Z * _WILSON_Z / (2 * n)) / denom
    margin = _WILSON_Z * sqrt(p * (1 - p) / n + _WILSON_Z * _WILSON_Z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_interval(values: List[float]) -> Tuple[float, float]:
    """Percentile bootstrap interval for a continuous metric, same scale as the mean.

    Fixed seed and batched resampling keep memory bounded and results reproducible.
    """
    import numpy as np

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    n = len(values)
    array = np.asarray(values, dtype=float)
    means = np.empty(_BOOTSTRAP_B, dtype=float)
    batch_size = 250
    start = 0
    while start < _BOOTSTRAP_B:
        count = min(batch_size, _BOOTSTRAP_B - start)
        index = rng.integers(0, n, size=(count, n))
        means[start : start + count] = array[index].mean(axis=1)
        start += count
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@register_aggregation(name='mean_and_ci')
class MeanAndCI(Mean):
    """Mean plus a 95% confidence interval and representativeness flags.

    Ratio metrics (every sample value is 0 or 1) get the analytic Wilson interval;
    continuous metrics get a percentile bootstrap. The resampling unit is the unit the
    mean is taken over — the samples passed to this aggregator, i.e. one reported subset;
    a reported mean-of-subset-means must be bootstrapped at that layer instead.

    One ``AggScore`` per metric: the interval lives in its ``metadata`` under the ``ci``
    key (numeric endpoints in the metric's own scale, level, method, flags) — a single
    metric identity, losslessly saved with the report and formatted at the display
    boundary. ``n`` counts scored samples (post fail-closed); exclusions are assumed
    ignorable and their count is a report-layer concern.

    Repeated generations of the same item are **not** independent samples; this v1
    aggregator rejects them instead of treating repeat count as sample count.
    """

    name = 'mean_and_ci'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        if not scores:
            return []

        seen_ids: Dict[Any, None] = {}
        for score in scores:
            if score.sample_id in seen_ids:
                raise ValueError(
                    f'Sample {score.sample_id!r} appears more than once; repeated generations are '
                    'not independent samples and mean_and_ci does not support repeats. '
                    'Aggregate repeats with mean_and_pass_at_k-style aggregators instead.'
                )
            seen_ids[score.sample_id] = None

        metric_values: Dict[str, List[float]] = defaultdict(list)
        metric_sample_ids: Dict[str, List[Any]] = defaultdict(list)
        for score in scores:
            for metric_name, value in score.score.value.items():
                metric_values[metric_name].append(value)
                metric_sample_ids[metric_name].append(score.sample_id)

        aggregated_scores: List[AggScore] = []
        for metric_name, values in metric_values.items():
            if not values:
                continue
            n = len(values)
            is_ratio = all(v in (0, 1) for v in values)
            if is_ratio:
                ci_low, ci_high = _wilson_interval(values)
                ci_method = 'wilson'
            else:
                ci_low, ci_high = _bootstrap_interval(values)
                ci_method = 'bootstrap'

            flags: List[str] = []
            if n < _LOW_CONFIDENCE_N:
                flags.append('low_confidence')
            if (ci_high - ci_low) * 100 > _WIDE_INTERVAL_PTS:
                flags.append('wide_interval')

            aggregated_scores.append(
                AggScore(
                    score=self.agg_func(values),
                    metric_name=metric_name,
                    aggregation='mean',
                    num=n,
                    num_correct=int(sum(values)) if is_ratio else None,
                    ids=metric_sample_ids[metric_name],
                    metadata={
                        'ci': {
                            'low': ci_low,
                            'high': ci_high,
                            'level': 0.95,
                            'method': ci_method,
                            'flags': flags,
                        },
                    },
                )
            )
        return aggregated_scores
