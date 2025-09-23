from collections import defaultdict
from typing import List

from evalscope.api.metric import Aggregator, AggScore, Metric, SampleScore, T2IMetric
from evalscope.api.registry import register_aggregation, register_metric
from .metrics import mean


def normalize_text(text: str) -> str:
    """Normalize text by lowering case and stripping whitespace."""
    return text.strip().lower()


@register_metric(name='exact_match')
class ExactMatch(Metric):

    def apply(self, predictions, references):
        return [
            float(normalize_text(prediction) == normalize_text(reference))
            for prediction, reference in zip(predictions, references)
        ]


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


@register_metric(name='multi_choice_acc')
class MultiChoiceAcc(Metric):

    def apply(self, predictions, references):
        """
        Calculate accuracy for multiple-choice questions.

        Args:
            predictions (List[str]): List of predicted answers.
            references (List[str]): List of correct answers.

        Returns:
            List[float]: List of accuracy scores (1.0 for correct, 0.0 for incorrect).
        """
        res = []
        for prediction, reference in zip(predictions, references):
            prediction = set(prediction.strip().upper())
            reference = set(reference.strip().upper())
            # if the prediction has answer that not in reference, it is wrong
            if not prediction.issubset(reference):
                res.append(0.0)
                continue
            common = prediction.intersection(reference)
            res.append(len(common) / len(reference) if reference else 0.0)
        return res


# ##################
# T2I Metrics ######
####################
@register_metric(name='VQAScore')
class VQAScore(T2IMetric):

    def _init_once(self, model: str = 'clip-flant5-xxl'):
        from .t2v_metrics.vqascore import VQAScore
        self.model = VQAScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='PickScore')
class PickScore(T2IMetric):

    def _init_once(self, model: str = 'pickscore-v1'):
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='CLIPScore')
class CLIPScore(T2IMetric):

    def _init_once(self, model: str = 'openai:ViT-L-14-336'):
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='BLIPv2Score')
class BLIPv2Score(T2IMetric):

    def _init_once(self, model: str = 'blip2-itm'):
        from .t2v_metrics.itmscore import ITMScore
        self.model = ITMScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='HPSv2Score')
class HPSv2Score(T2IMetric):

    def _init_once(self, model: str = 'hpsv2'):
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='HPSv2.1Score')
class HPSv2_1Score(T2IMetric):

    def _init_once(self, model: str = 'hpsv2.1'):
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='ImageRewardScore')
class ImageRewardScore(T2IMetric):

    def _init_once(self, model: str = 'image-reward-v1'):
        from .t2v_metrics.itmscore import ITMScore
        self.model = ITMScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='FGA_BLIP2Score')
class FGA_BLIP2Score(T2IMetric):

    def _init_once(self, model: str = 'fga_blip2'):
        from .t2v_metrics.itmscore import ITMScore
        self.model = ITMScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


@register_metric(name='MPS')
class MPS(T2IMetric):

    def _init_once(self, model: str = 'mps'):
        from .t2v_metrics.clipscore import CLIPScore
        self.model = CLIPScore(model=model)

    def apply(self, images: List[str], texts: List[str], **kwargs) -> List[float]:
        return self.model(images, texts, **kwargs)


# ##################
# Aggregators ######
# ##################
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


@register_aggregation(name='clipped_mean')
class ClippedMean(Mean):

    name = 'clipped_mean'

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def agg_func(self, values: List[float]) -> float:
        clipped_values = min(max(mean(values), self.clip_min), self.clip_max)
        return clipped_values


@register_aggregation(name='pass_at_k')
class PassAtK(Aggregator):

    def __init__(self, k: int = 1):
        self.k = k
        self.name = f'pass_at_{k}'

    def __call__(self, scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate scores by computing the pass@k for each metric using group_id.

        Args:
            scores: List of sample scores to aggregate

        Returns:
            List of aggregated scores with pass@k values
        """
        if not scores:
            return []

        import numpy as np

        from .metrics import calculate_pass_at_k

        # Group scores by metric name and group_id
        metric_groups = defaultdict(lambda: defaultdict(list))

        for score in scores:
            group_id = getattr(score, 'group_id', score.sample_id)  # fallback to sample_id if no group_id

            for metric_name, value in score.score.value.items():
                metric_groups[metric_name][group_id].append(float(value))

        # Calculate pass@k for each metric
        aggregated_scores = []
        for metric_name, groups in metric_groups.items():
            if not groups:
                continue

            # Calculate pass@k for each group (problem)
            num_samples = []
            num_correct = []
            all_sample_ids = []

            for group_id, group_values in groups.items():
                num_samples.append(len(group_values))
                num_correct.append(sum(group_values))  # count how many passed in this group
                all_sample_ids.extend([f'{group_id}_{i}' for i in range(len(group_values))])

            if num_samples:
                # Use the calculate_pass_at_k function from metrics
                pass_at_k_values = calculate_pass_at_k(num_samples, num_correct, self.k)
                overall_pass_at_k = float(np.mean(pass_at_k_values))

                aggregated_scores.append(
                    AggScore(
                        score=overall_pass_at_k,
                        metric_name=f'pass@{self.k}',
                        aggregation_name='',
                        num=len(scores),
                        ids=all_sample_ids
                    )
                )

        return aggregated_scores
