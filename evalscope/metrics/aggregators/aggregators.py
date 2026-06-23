# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from typing import Any, Dict, List

from evalscope.api.metric import Aggregator, AggScore, SampleScore
from evalscope.api.registry import register_aggregation
from evalscope.metrics.utils.functions import calculate_pass_at_k, calculate_pass_hat_k, mean


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
                        aggregation_name=self.name,
                        num=len(values),
                        ids=metric_sample_ids[metric_name]
                    )
                )

        return aggregated_scores


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
        """Add per-metric pass@n for all n <= k to each sample, then mean-aggregate.

        For each metric:
        - Group scores by group_id
        - Collect binary correctness values
        - Infer k as (total samples / number of groups) assuming uniform repetitions
        - Compute per-group pass@n for all n from 1 to k via calculate_pass_at_k
        - Annotate each sample with metric_pass@n for its group (for all n)
        Finally run Mean() over the augmented metric set.
        """
        if not scores:
            return []

        # Extract metric names present in score values
        metrics = list(scores[0].score.value.keys())

        for metric_name in metrics:
            # group_id -> list[float] (0/1 correctness values)
            group_values: Dict[str, List[float]] = defaultdict(list)
            for s in scores:
                group_id = getattr(s, 'group_id', s.sample_id)
                value = float(s.score.value[metric_name])
                group_values[group_id].append(value)

            if not group_values:
                continue

            # Infer k (assumes roughly uniform repeats)
            k = int(len(scores) / len(group_values)) if len(group_values) > 0 else 1
            if k <= 0:
                k = 1

            # Prepare inputs for calculate_pass_at_k
            num_samples: List[int] = []
            num_correct: List[int] = []
            group_order: List[str] = []
            for gid, vals in group_values.items():
                group_order.append(gid)
                num_samples.append(len(vals))
                num_correct.append(int(sum(vals)))

            # Compute per-group pass@n for all n from 1 to k
            pass_at_n_maps = {}
            for n in range(1, k + 1):
                pass_at_n_list = calculate_pass_at_k(num_samples, num_correct, n)
                pass_at_n_maps[n] = {gid: float(v) for gid, v in zip(group_order, pass_at_n_list)}

            # Annotate each sample with its group's pass@n for all n
            for s in scores:
                group_id = getattr(s, 'group_id', s.sample_id)
                for n in range(1, k + 1):
                    s.score.value[f'{metric_name}_pass@{n}'] = pass_at_n_maps[n][group_id]

        # Delegate mean aggregation over original + injected pass@n metrics
        m = Mean()
        return m(scores)


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

        # Freeze metric names before augmenting values
        metrics = list(scores[0].score.value.keys())

        for metric_name in metrics:
            # Group samples by group_id, preserving order
            # Store: (prediction, correctness_score)
            group_samples: Dict[str, List[tuple]] = defaultdict(list)
            for score in scores:
                group_id = getattr(score, 'group_id', score.sample_id)
                prediction = getattr(score.score, 'extracted_prediction', None)
                correctness = score.score.value[metric_name]
                group_samples[group_id].append((prediction, correctness))

            if not group_samples:
                continue

            # Calculate k as the repetition count
            k = int(len(scores) / len(group_samples)) if len(group_samples) > 0 else 1
            if k <= 0:
                k = 1

            # Compute vote@n for all n from 1 to k for each group
            vote_at_n_maps: Dict[int, Dict[str, float]] = {}
            for n in range(1, k + 1):
                vote_at_n_maps[n] = {}
                for group_id, samples in group_samples.items():
                    # Consider only first n samples for this group
                    n_samples = samples[:n]

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

                    vote_at_n_maps[n][group_id] = 1.0 if is_correct else 0.0

            # Annotate each sample with its group's vote@n for all n
            for score in scores:
                group_id = getattr(score, 'group_id', score.sample_id)
                for n in range(1, k + 1):
                    score.score.value[f'{metric_name}_vote@{n}'] = vote_at_n_maps[n][group_id]

        # Calculate the mean value for all metrics and their corresponding vote@n
        m = Mean()
        return m(scores)


@register_aggregation(name='mean_and_pass_hat_k')
class MeanPassHatK(Aggregator):

    def __init__(self):
        self.name = 'mean_and_pass_hat_k'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Add per-metric pass^n for all n <= k using calculate_pass_hat_k, then mean-aggregate.

        For each metric:
        - Group scores by group_id
        - Collect binary correctness values
        - Infer k as approximate repeats and clamp to min attempts across groups
        - Compute per-group pass^n for all n from 1 to k via calculate_pass_hat_k
        - Annotate each sample with metric_pass^{n} for its group (for all n)
        Finally run Mean() over the augmented metric set.
        """
        if not scores:
            return []

        # Freeze metric names before augmenting values to avoid iterating injected keys
        metrics = list(scores[0].score.value.keys())

        for metric_name in metrics:
            # group_id -> list[float] (0/1 correctness values)
            group_values: Dict[str, List[float]] = defaultdict(list)
            for s in scores:
                group_id = getattr(s, 'group_id', s.sample_id)
                value = float(s.score.value[metric_name])
                group_values[group_id].append(value)

            if not group_values:
                continue

            # Infer repeats and clamp to the smallest group size to satisfy n <= min_n
            approx_k = int(len(scores) / len(group_values)) if len(group_values) > 0 else 1
            min_n = min(len(vals) for vals in group_values.values())
            k = max(1, min(approx_k, min_n))

            # Compute per-group pass^n for all n from 1 to k
            pass_hat_n_maps: Dict[int, Dict[str, float]] = {}
            for n in range(1, k + 1):
                pass_hat_n_maps[n] = {}
                for gid, vals in group_values.items():
                    total = len(vals)
                    correct = int(sum(vals))
                    pass_hat_n_maps[n][gid] = float(calculate_pass_hat_k(total, correct, n))

            # Annotate each sample with its group's pass^n for all n
            for s in scores:
                group_id = getattr(s, 'group_id', s.sample_id)
                for n in range(1, k + 1):
                    s.score.value[f'{metric_name}_pass^{n}'] = pass_hat_n_maps[n][group_id]

        # Mean aggregate over original + injected pass^n metrics
        m = Mean()
        return m(scores)
