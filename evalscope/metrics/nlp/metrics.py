import json
import os
from typing import List

from evalscope.api.metric import Metric, SingletonMetric
from evalscope.api.registry import register_metric
from evalscope.metrics.utils.functions import normalize_text
from evalscope.utils.import_utils import check_import

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
            from evalscope.metrics.math.parser import math_equal, strip_answer_string

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
        from evalscope.metrics.math.parser import extract_answer, math_equal, strip_answer_string

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
        from evalscope.metrics.utils.functions import levenshtein_distance

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

    def apply(self, samples: List[dict]) -> List[float]:
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


@register_metric(name='sem_score')
class SemScore(SingletonMetric):

    def _init_once(self, **kwargs):
        """SemScore metric.
        """
        check_import('bert_score', 'bert-score', raise_error=True, feature_name='SemScore Metric')
        check_import('torch', 'torch', raise_error=True, feature_name='SemScore Metric')

        from .sem_score.scorer import SemScorer
        self.scorer = SemScorer(batch_size=1024, **kwargs)

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        scores = self.scorer.score_all(predictions, references)
        return [round(score, 6) for score in scores]
