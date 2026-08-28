# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
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


def collect_metric_weights(score: SampleScore) -> Dict[str, float]:
    """Read one sample's per-metric weights, tolerating scores that declare none or declare junk.

    A malformed weight must not abort a finished evaluation, so anything non-numeric is dropped and
    the metric falls back to its unweighted mean.
    """
    metadata = score.score.metadata or {}
    declared = metadata.get(METRIC_WEIGHTS_KEY) or {}
    if not isinstance(declared, dict):
        return {}
    weights: Dict[str, float] = {}
    for metric_name, weight in declared.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            continue
        weights[metric_name] = float(weight)
    return weights


@register_aggregation(name='weighted_mean')
class WeightedMean(Aggregator):
    """Mean that honours per-sample weights declared in ``Score.metadata``.

    A per-sample value is often itself a ratio over several units, so averaging the ratios gives
    every sample the same say regardless of how many units it covered -- a macro-average. Weighting
    each value by its unit count restores the micro-average the underlying benchmark defines
    (IFEval's instruction-level accuracy pools every instruction, see
    ``instruction_following_eval/evaluation_lib.py``).

    Metrics without a declared weight fall back to an unweighted mean, so prompt-level and
    instruction-level metrics coexist under one aggregator.
    """

    name = 'weighted_mean'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        if not scores:
            return []

        metric_values: Dict[str, List[float]] = defaultdict(list)
        metric_weights: Dict[str, List[float]] = defaultdict(list)
        metric_sample_ids: Dict[str, List[Any]] = defaultdict(list)
        metric_declared: Dict[str, bool] = defaultdict(bool)

        for score in scores:
            weights = collect_metric_weights(score)
            for metric_name, value in score.score.value.items():
                metric_values[metric_name].append(value)
                # A neutral 1.0 keeps a sample that declared no weight from vanishing out of a
                # weighted metric's denominator.
                metric_weights[metric_name].append(weights.get(metric_name, 1.0))
                metric_declared[metric_name] |= metric_name in weights
                metric_sample_ids[metric_name].append(score.sample_id)

        aggregated_scores = []
        for metric_name, values in metric_values.items():
            weights = metric_weights[metric_name]
            total_weight = sum(weights)
            # A fully zero-weight metric carries no units to average over; fall back rather than
            # divide by zero and report a spurious 0.0.
            if total_weight > 0:
                aggregated = sum(value * weight for value, weight in zip(values, weights)) / total_weight
            else:
                aggregated = mean(values)
            # Declared-ness, not the numeric value, decides what ``num`` means: a dataset whose
            # prompts all carry exactly one instruction yields all-1.0 weights yet is still a
            # weighted metric whose unit total happens to equal the sample count.
            weighted = metric_declared[metric_name] and total_weight > 0
            aggregated_scores.append(
                AggScore(
                    score=aggregated,
                    metric_name=metric_name,
                    aggregation=self.name,
                    # ``num`` is the weight total for a weighted metric, so the report layer's
                    # ``micro_mean`` rollup across subsets stays a true micro-average.
                    num=int(total_weight) if weighted else len(values),
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

                    # Count prediction frequencies
                    prediction_counts = defaultdict(int)
                    for prediction, _ in n_samples:
                        prediction_counts[prediction] += 1

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
