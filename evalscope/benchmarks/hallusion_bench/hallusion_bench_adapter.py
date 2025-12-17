from collections import defaultdict
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
        name='hallusion_bench',
        pretty_name='HallusionBench',
        tags=[Tags.MULTI_MODAL, Tags.HALLUCINATION, Tags.YES_NO],
        description=
        'HallusionBench is an advanced diagnostic benchmark designed to evaluate image-context reasoning, analyze models\' tendencies for language hallucination and visual illusion in large vision-language models (LVLMs).',  # noqa: E501
        dataset_id='lmms-lab/HallusionBench',
        metric_list=['aAcc', 'qAcc', 'fAcc'],
        aggregation='f1',
        eval_split='image',
        prompt_template='{question}\nPlease answer YES or NO without an explanation.',
    )
)
class HallusionBenchAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:

        input_text = self.prompt_template.format(question=record['question'])
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        answer = 'NO' if record['gt_answer'] == '0' else 'YES'
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=answer,
            metadata={
                'category': record.get('category'),
                'subcategory': record.get('subcategory'),
                'visual_input': record.get('visual_input'),
                'set_id': record.get('set_id'),
                'figure_id': record.get('figure_id'),
                'question_id': record.get('question_id'),
                'gt_answer': record.get('gt_answer'),
                'gt_answer_details': record.get('gt_answer_details'),
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

        def compute_aAcc(scores: List[SampleScore]):
            total = len(scores)
            if total == 0:
                return 0.0, 0
            correct = sum(ss.score.main_value for ss in scores)
            return (correct / total), total

        def compute_group_accuracy(scores: List[SampleScore], group_type: str):
            # group_type: 'figure' or 'question'
            groups = defaultdict(list)
            for ss in scores:
                md = ss.sample_metadata
                subcategory = md.get('subcategory')
                set_id = md.get('set_id')
                group_id = md.get('figure_id') if group_type == 'figure' else md.get('question_id')
                if subcategory is None or set_id is None or group_id is None:
                    # Skip incomplete records for this grouping
                    continue
                key = f'{subcategory}_{set_id}_{group_id}'
                groups[key].append(ss.score.main_value)
            if not groups:
                return 0.0, 0
            num_correct_groups = sum(1 for vals in groups.values() if all(vals))
            num_groups = len(groups)
            return (num_correct_groups / num_groups), num_groups

        def compute_metrics(scores: List[SampleScore]) -> Dict[str, Dict[str, float]]:
            a_acc, a_n = compute_aAcc(scores)
            f_acc, f_n = compute_group_accuracy(scores, 'figure')
            q_acc, q_n = compute_group_accuracy(scores, 'question')
            return {
                'aAcc': {
                    'score': a_acc,
                    'num': a_n
                },
                'fAcc': {
                    'score': f_acc,
                    'num': f_n
                },
                'qAcc': {
                    'score': q_acc,
                    'num': q_n
                },
            }

        outputs: List[AggScore] = []

        # By subcategory
        subcategories = sorted({ss.sample_metadata.get('subcategory') for ss in sample_scores})
        for subcategory in subcategories:
            subset = [ss for ss in sample_scores if ss.sample_metadata.get('subcategory') == subcategory]
            stats = compute_metrics(subset)
            for metric in ['aAcc', 'fAcc', 'qAcc']:
                outputs.append(
                    AggScore(
                        score=stats[metric]['score'],
                        metric_name=metric,
                        aggregation_name=str(subcategory),
                        num=stats[metric]['num'],
                    )
                )

        # By category
        categories = sorted({ss.sample_metadata.get('category') for ss in sample_scores})
        for category in categories:
            subset = [ss for ss in sample_scores if ss.sample_metadata.get('category') == category]
            stats = compute_metrics(subset)
            for metric in ['aAcc', 'fAcc', 'qAcc']:
                outputs.append(
                    AggScore(
                        score=stats[metric]['score'],
                        metric_name=metric,
                        aggregation_name=str(category),
                        num=stats[metric]['num'],
                    )
                )

        # Overall
        overall = compute_metrics(sample_scores)
        for metric in ['aAcc', 'fAcc', 'qAcc']:
            outputs.append(
                AggScore(
                    score=overall[metric]['score'],
                    metric_name=metric,
                    aggregation_name='Overall',
                    num=overall[metric]['num'],
                )
            )

        return outputs
