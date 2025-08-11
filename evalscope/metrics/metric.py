from collections import defaultdict
from typing import List

from evalscope.api.metric import Aggregator, AggScore, Metric, SampleScore
from evalscope.api.registry import register_aggregation, register_metric


@register_metric(name='exact_match')
class ExactMatch(Metric):

    def apply(self, predictions, references):
        return [float(prediction == reference) for prediction, reference in zip(predictions, references)]


@register_metric(name='acc')
class Accuracy(ExactMatch):

    def __init__(self, allow_inclusion: bool = False, numeric: bool = False):
        self.allow_inclusion = allow_inclusion
        self.numeric = numeric

    def apply(self, predictions, references):
        if self.allow_inclusion:
            results = []
            for prediction, reference in zip(predictions, references):
                if prediction and prediction in reference:
                    results.append(1.0)
                else:
                    results.append(0.0)
            return results
        elif self.numeric:
            from .math_parser import extract_answer, math_equal, strip_answer_string

            results = []
            for prediction, reference in zip(predictions, references):
                pred_answer = strip_answer_string(extract_answer(prediction))
                ref_answer = strip_answer_string(reference)
                results.append(float(math_equal(pred_answer, ref_answer)))

            return results
        else:
            return super().apply(predictions, references)


@register_metric(name='numeric_match')
class NumericMatch(Metric):

    def apply(self, predictions, references):
        return [float(prediction == reference) for prediction, reference in zip(predictions, references)]


@register_metric(name='math_acc')
class MathAcc(Metric):

    def apply(self, predictions, references):
        from .math_parser import extract_answer, math_equal, strip_answer_string

        results = []
        for prediction, reference in zip(predictions, references):
            pred_answer = strip_answer_string(extract_answer(prediction))
            ref_answer = strip_answer_string(reference)
            results.append(float(math_equal(pred_answer, ref_answer)))

        return results


@register_aggregation(name='mean')
class Mean(Aggregator):

    name = 'mean'

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
                mean_value = sum(values) / len(values)
                aggregated_scores.append(
                    AggScore(
                        score=mean_value,
                        metric_name=metric_name,
                        aggregation_name=self.name,
                        num=len(values),
                        ids=metric_sample_ids[metric_name]
                    )
                )

        return aggregated_scores
