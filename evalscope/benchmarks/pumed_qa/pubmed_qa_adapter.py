import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentText
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = ('PubMedQA reasons over biomedical research texts to answer the multiple-choice questions.')


@register_benchmark(
    BenchmarkMeta(
        name='pubmedqa',
        pretty_name='PubMedQA',
        tags=[Tags.KNOWLEDGE, Tags.YES_NO],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/pubmed-qa',
        metric_list=['accuracy', 'precision', 'recall', 'f1_score', 'yes_ratio', 'maybe_ratio'],
        aggregation='f1',
        few_shot_num=0,
        eval_split='test',
        prompt_template='{question}\nPlease answer YES or NO or MAYBE without an explanation.',
    )
)
class PubMedQAAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_overall_metric = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        abstract = record['context']
        question = record['question']
        question = f'Abstract: {abstract}\n\nQuestion: {question}'
        input_text = self.prompt_template.format(question=question)
        content_list: List[Content] = [ContentText(text=input_text)]
        answer = str(record['answer']).upper()  # 'YES' or 'NO' or 'MAYBE'
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=answer,
            metadata={
                'answer': record['answer'],
                'reasoning': record['reasoning'],
            }
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        # Check if the reference answer is in the filtered prediction
        result = 1 if re.search(r'\b' + re.escape(reference) + r'\b', filtered_prediction.strip().upper()) else 0
        score.value = {'acc': result}
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Custom aggregation to compute accuracy, precision, recall, f1_score, yes_ratio and maybe_ratio.
        Handles multi-class classification with YES, NO, and MAYBE answers.
        """

        def compute_metrics(scores: List[SampleScore]):
            # Initialize confusion matrix for multi-class classification
            confusion_matrix = {
                'YES': {
                    'YES': 0,
                    'NO': 0,
                    'MAYBE': 0
                },
                'NO': {
                    'YES': 0,
                    'NO': 0,
                    'MAYBE': 0
                },
                'MAYBE': {
                    'YES': 0,
                    'NO': 0,
                    'MAYBE': 0
                }
            }

            yes_count = 0
            maybe_count = 0
            total_count = len(scores)
            correct_count = 0

            for ss in scores:
                gt = ss.sample_metadata['answer'].strip().upper()

                if ss.score.main_value == 1:
                    correct_count += 1
                    pred = gt
                else:
                    pred_text = ss.score.extracted_prediction.strip().upper()
                    # Heuristic to determine the predicted class from text
                    if 'YES' in pred_text:
                        pred = 'YES'
                    elif 'NO' in pred_text:
                        pred = 'NO'
                    elif 'MAYBE' in pred_text:
                        pred = 'MAYBE'
                    else:
                        pred = None

                if pred:
                    if pred == 'YES':
                        yes_count += 1
                    elif pred == 'MAYBE':
                        maybe_count += 1

                    if gt in confusion_matrix and pred in confusion_matrix[gt]:
                        confusion_matrix[gt][pred] += 1

            # Calculate accuracy
            accuracy = correct_count / total_count if total_count > 0 else 0.0

            # Calculate per-class precision, recall, and F1
            classes = ['YES', 'NO', 'MAYBE']
            precision_values = []
            recall_values = []
            f1_values = []

            for cls in classes:
                # True positives for this class
                tp = confusion_matrix[cls][cls]

                # Calculate predicted positives (column sum)
                pred_pos = sum(confusion_matrix[true_cls][cls] for true_cls in classes)

                # Calculate actual positives (row sum)
                act_pos = sum(confusion_matrix[cls][pred_cls] for pred_cls in classes)

                # Calculate precision and recall for this class
                cls_precision = tp / pred_pos if pred_pos > 0 else 0.0
                cls_recall = tp / act_pos if act_pos > 0 else 0.0

                # Calculate F1 for this class
                cls_f1 = (2 * cls_precision * cls_recall) / (cls_precision
                                                             + cls_recall) if (cls_precision + cls_recall) > 0 else 0.0

                precision_values.append(cls_precision)
                recall_values.append(cls_recall)
                f1_values.append(cls_f1)

            # Macro average (simple average across all classes)
            precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
            recall = sum(recall_values) / len(recall_values) if recall_values else 0.0
            f1_score = sum(f1_values) / len(f1_values) if f1_values else 0.0

            # Calculate ratios
            yes_ratio = yes_count / total_count if total_count > 0 else 0.0
            maybe_ratio = maybe_count / total_count if total_count > 0 else 0.0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'yes_ratio': yes_ratio,
                'maybe_ratio': maybe_ratio
            }

        overall_metrics = compute_metrics(sample_scores)
        agg_scores = []
        for metric_name, value in overall_metrics.items():
            agg_scores.append(AggScore(metric_name=metric_name, score=value, num=len(sample_scores), metadata={}))

        return agg_scores
