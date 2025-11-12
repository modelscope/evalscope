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
        description=
        'POPE (Polling-based Object Probing Evaluation) is a benchmark designed to evaluate object hallucination in large vision-language models (LVLMs). It tests models by having them answer simple yes/no questions about the presence of specific objects in an image. This method helps measure how accurately a model\'s responses align with the visual content, with a focus on identifying instances where models claim objects exist that are not actually present. The benchmark employs various sampling strategies, including random, popular, and adversarial sampling, to create a robust set of questions for assessment.',  # noqa: E501
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
