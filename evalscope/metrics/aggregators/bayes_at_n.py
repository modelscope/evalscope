"""Binary Bayes@N posterior moments (https://arxiv.org/abs/2510.04265)."""

from math import fsum, isfinite, sqrt
from typing import Dict, List, Union

from evalscope.api.metric import Aggregator, AggScore, SampleScore
from evalscope.api.registry import register_aggregation
from evalscope.metrics.semantics.naming import canonicalize_producer_identity


@register_aggregation(name='bayes_at_n')
class BayesAtN(Aggregator):
    """Estimate mean correctness and posterior SD from complete, fixed-N question groups.

    Each question has an independent Beta(1, 1) prior. Only binary accuracy and exact-match
    outcomes are supported. ``repeats`` is the planned generation count, including unavailable
    attempts; an incomplete or unusable group raises ``ValueError``.
    """

    name = 'bayes_at_n'

    def __init__(self, repeats: int = 1) -> None:
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError('Bayes@N requires a positive integer repeats count.')
        self.repeats = repeats

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Return posterior mean and posterior SD for each supported correctness metric."""
        if not scores:
            raise ValueError('Bayes@N requires at least one complete question group.')

        groups: Dict[Union[str, int], Dict[int, SampleScore]] = {}
        seen_ids = set()
        metric_names = None
        for sample in scores:
            if sample.sample_id is None or sample.sample_id in seen_ids:
                raise ValueError('Bayes@N requires distinct, non-missing sample IDs.')
            seen_ids.add(sample.sample_id)
            if sample.group_id is None or sample.generation_index is None:
                raise ValueError('Bayes@N requires group_id and generation_index; rerun legacy reviews.')
            position = sample.generation_index
            if not 0 <= position < self.repeats:
                raise ValueError('Bayes@N generation_index is outside the planned repeats count.')
            attempts = groups.setdefault(sample.group_id, {})
            if position in attempts:
                raise ValueError('Bayes@N received a duplicate generation position in a question group.')
            attempts[position] = sample

            if not sample.score.status.is_usable or not sample.score.value:
                raise ValueError('Bayes@N requires a usable correctness score for every planned generation.')
            names = set(sample.score.value)
            if metric_names is None:
                metric_names = names
            if names != metric_names:
                raise ValueError('Bayes@N requires the same correctness metrics on every planned generation.')
            canonical_names = [canonicalize_producer_identity(name, self.name).name for name in names]
            if any(name not in {'accuracy', 'exact_match'} for name in canonical_names):
                raise ValueError('Bayes@N supports only binary accuracy and exact_match metrics.')
            if len(set(canonical_names)) != len(canonical_names):
                raise ValueError('Bayes@N received aliases for the same correctness metric.')
            for value in sample.score.value.values():
                if not isfinite(value) or value not in (0, 1):
                    raise ValueError('Bayes@N requires binary correctness values of 0 or 1.')

        if any(len(attempts) != self.repeats for attempts in groups.values()):
            raise ValueError('Bayes@N requires every question to have exactly repeats planned generations.')

        question_count = len(groups)
        total = self.repeats + 2
        result = []
        for metric_name in scores[0].score.value:
            successes = [
                sum(attempt.score.value[metric_name] for attempt in group.values()) for group in groups.values()
            ]
            posterior_mean = fsum((count + 1) / total for count in successes) / question_count
            variance_sum = fsum(
                (count + 1) * (self.repeats - count + 1) / (total * total * (total + 1)) for count in successes
            )
            posterior_sd = sqrt(variance_sum) / question_count
            metadata = {
                'questions': question_count,
                'generations': len(scores),
                'repeats': self.repeats,
                'prior': 'beta(1,1)',
                'posterior_mean': posterior_mean,
                'posterior_sd': posterior_sd,
            }
            for aggregation, value in ((self.name, posterior_mean), ('bayes_at_n_sd', posterior_sd)):
                result.append(
                    AggScore(
                        metric_name=metric_name,
                        aggregation=aggregation,
                        dimensions={'n': self.repeats},
                        score=value,
                        num=question_count,
                        ids=list(groups),
                        metadata=metadata,
                    )
                )
        return result
