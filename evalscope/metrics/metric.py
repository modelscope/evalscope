import json
import os
from collections import defaultdict
from typing import Dict, List

from evalscope.api.metric import Aggregator, AggScore, Metric, SampleScore, SingletonMetric, T2IMetric
from evalscope.api.registry import register_aggregation, register_metric
from evalscope.utils.import_utils import check_import
from .metrics import calculate_pass_at_k, calculate_pass_hat_k, mean, normalize_text

# ##################
# NLP Metrics ######
# ##################


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
            from .math_parser import math_equal, strip_answer_string

            results = []
            for prediction, reference in zip(predictions, references):
                ref_answer = strip_answer_string(reference)
                results.append(float(math_equal(prediction, ref_answer)))

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


@register_metric(name='anls')
class ANLS(Metric):

    def __init__(self, thresh_hold=0.5):
        self.thresh_hold = thresh_hold

    def apply(self, predictions, references):
        """
        Calculate ANLS (Average Normalized Levenshtein Similarity) for a list of predictions and references.
        This implementation is adapted from
        https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py

        Args:
            references (List[str]): List of correct answers. Each answer can be a string of json.
            predictions (List[str]): List of predicted answers.
        """
        from .metrics import levenshtein_distance

        res = []
        # Unwrap predictions if it's a nested list
        for prediction, reference in zip(predictions, references):
            # Parse the reference which is a json string
            try:
                answer = json.loads(reference)
            except json.JSONDecodeError:
                answer = reference
            if isinstance(answer, str):
                answer = [answer]
            assert isinstance(answer, list), 'The reference answer should be a list of answers.'

            # Calculate ANLS for each reference answer
            values = []
            for ans in answer:
                # preprocess both the answers - gt and prediction
                gt_answer = ' '.join(ans.strip().lower().split())
                det_answer = ' '.join(prediction.strip().lower().split())

                dist = levenshtein_distance(gt_answer, det_answer)
                length = max(len(ans.upper()), len(prediction.upper()))
                values.append(0.0 if length == 0 else float(dist) / float(length))

            question_result = 0.0
            if values:
                question_result = 1 - min(values)
                if question_result < self.thresh_hold:
                    question_result = 0.0
            res.append(question_result)
        return res


@register_metric(name='wer')
class WER(Metric):

    def __init__(self, language: str = 'English'):
        self.language = language

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        from .text_normalizer.wer import wer

        return [wer([ref], [pred], self.language) for pred, ref in zip(predictions, references)]


@register_metric(name='bertscore')
class BertScore(SingletonMetric):

    def _init_once(self, model_id_or_path: str = 'google-bert/bert-base-chinese', **kwargs):
        """BertScore metric.

        Args:
            model_id_or_path (str, optional): The model ID on modelscope or path to the pre-trained model.
                Defaults to 'google-bert/bert-base-chinese'.
        """
        check_import('torch', 'torch', raise_error=True, feature_name='BertScore Metric')

        from .bert_score.scorer import BERTScorer
        self.scorer = BERTScorer(model_id_or_path=model_id_or_path, batch_size=1024, **kwargs)

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        _, _, F1 = self.scorer.score(predictions, references)
        return [round(f1.item(), 6) for f1 in F1]


@register_metric(name='comet')
class COMETScore(SingletonMetric):

    def _init_once(self, model_id_or_path: str = 'evalscope/wmt22-comet-da'):
        """COMETScore metric.

        Args:
            model_name (str, optional): The model name on huggingface.
                Defaults to 'evalscope/wmt22-comet-da'.
        """
        check_import('comet', 'unbabel-comet', raise_error=True, feature_name='COMETScore Metric')

        from comet import load_from_checkpoint
        from modelscope import snapshot_download

        self.model_name = model_id_or_path
        model_path = snapshot_download(model_id_or_path)
        checkpoint_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')
        self.comet_scorer = load_from_checkpoint(checkpoint_path)

    def apply(self, samples: List[Dict[str, str]]) -> List[float]:
        """Apply COMET scoring."""
        import torch

        model_output = self.comet_scorer.predict(
            samples=samples,
            batch_size=1024,
            gpus=1 if torch.cuda.is_available() else 0,
            progress_bar=False,
        )
        scores = model_output.scores if hasattr(model_output, 'scores') else [model_output.system_score] * len(samples)

        return [round(score, 6) for score in scores]


# ##################
# T2I Metrics ######
# ##################
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
