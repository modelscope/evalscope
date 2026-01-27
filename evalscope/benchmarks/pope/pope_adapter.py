from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='pope',
        pretty_name='POPE',
        tags=[Tags.MULTI_MODAL, Tags.HALLUCINATION, Tags.YES_NO],
        description="""
## Overview

POPE (Polling-based Object Probing Evaluation) is a benchmark specifically designed to evaluate object hallucination in Large Vision-Language Models (LVLMs). It tests models' ability to accurately identify objects present in images through yes/no questions.

## Task Description

- **Task Type**: Object Hallucination Detection (Yes/No Q&A)
- **Input**: Image with question "Is there a [object] in the image?"
- **Output**: YES or NO answer
- **Focus**: Measuring accuracy vs. hallucination rate

## Key Features

- Three sampling strategies: random, popular, adversarial
- Tests for false positive object claims (hallucination)
- Based on MSCOCO images
- Simple yes/no question format for objective evaluation
- Measures alignment between model responses and visual content

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Five metrics: accuracy, precision, recall, F1 score, yes_ratio
- F1 score is the primary aggregation metric
- Three subsets: `popular`, `adversarial`, `random`
- "Popular" and "adversarial" subsets are more challenging
- yes_ratio indicates model's tendency to answer "yes"
""",
        dataset_id='lmms-lab/POPE',
        metric_list=['accuracy', 'precision', 'recall', 'f1_score', 'yes_ratio'],
        aggregation='f1',
        subset_list=['popular', 'adversarial', 'random'],
        default_subset='Full',
        prompt_template='{question}\nPlease answer YES or NO without an explanation.',
    )
)
class PopeAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_as_subset = True
        self.add_overall_metric = False

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:

        input_text = self.prompt_template.format(question=record['question'])
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        answer = record['answer'].upper()  # 'YES' or 'NO'
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=answer,
            metadata={
                'id': record.get('id'),
                'answer': answer,
                'category': record.get('category'),
                'question_id': record.get('question_id'),
            }
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        # Check if the reference answer is in the filtered prediction
        result = 1 if reference in filtered_prediction.strip().upper() else 0
        score.value = {'acc': result}
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Custom aggregation to compute accuracy, precision, recall, f1_score, and yes_ratio.
        """

        def compute_metrics(scores: List[SampleScore]):
            tp = fp = tn = fn = 0
            yes_count = 0
            total_count = len(scores)

            for ss in scores:
                gt = ss.sample_metadata['answer'].strip().upper()
                # Get prediction based on score
                pred = gt if ss.score.main_value == 1 else ('NO' if gt == 'YES' else 'YES')
                if pred == 'YES':
                    yes_count += 1
                if pred == 'YES' and gt == 'YES':
                    tp += 1
                elif pred == 'YES' and gt == 'NO':
                    fp += 1
                elif pred == 'NO' and gt == 'NO':
                    tn += 1
                elif pred == 'NO' and gt == 'YES':
                    fn += 1

            accuracy = (tp + tn) / total_count if total_count > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            yes_ratio = yes_count / total_count if total_count > 0 else 0.0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'yes_ratio': yes_ratio
            }

        overall_metrics = compute_metrics(sample_scores)
        agg_scores = []
        for metric_name, value in overall_metrics.items():
            agg_scores.append(AggScore(metric_name=metric_name, score=value, num=len(sample_scores), metadata={}))

        return agg_scores
