# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.api.benchmark import BenchmarkMeta, WOChoiceMultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.metric import Score
from evalscope.api.registry import get_metric, register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import WOChoiceMultipleChoiceTemplate

from .utils import strip_string, _extract_answer
# flake8: noqa

logger = get_logger()

SUBSET_LIST = [
    "mcq_knowledge",
    "mcq_reason",
    "nomcq_reason"
]

QA_TEMPLATE = """{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."""

@register_benchmark(
    BenchmarkMeta(
        name='internal_long_reasoning',
        pretty_name='ILReasoning',
        description='Internal dataset with multiple-choice question answering subset and free form answering subset ',
        tags=[Tags.MULTIPLE_CHOICE, Tags.CUSTOM],
        dataset_id='/app/custom_eval/internal/Reason_Knowledge_Dataset',
        subset_list=SUBSET_LIST,
        metric_list=['internal_numeric_acc', 'exact_match'],
        few_shot_num=0,
        train_split=None,
        eval_split=None,
        prompt_template=WOChoiceMultipleChoiceTemplate.MULTIPLE_ANSWER_COT
    )
)
class ILReasoningAdapter(WOChoiceMultiChoiceAdapter):
    """
    something named internal long reasoning dataset adapter
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.multiple_correct = True
        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    def load_from_disk(self, **kwargs):
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record) -> Sample:
        # Extract choices from the record (A, B, C, D, etc.)

        return Sample(
            input=record['question'] + '\n' + record['instruction'],
            target=str(record['answer']),
            metadata={
                'id': record.get('id', 'unknown'),
                'is_mcq': record.get('is_mcq', False),
                'class': record.get('class', 99)
                },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        if sample.metadata['is_mcq']:
            return self.prompt_template.format(
                question=sample.input
            )
        return QA_TEMPLATE.format(
            question=sample.input
        )

    def extract_answer(self, prediction, task_state):
        return strip_string(
            _extract_answer(
                prediction=prediction,
                task_state=task_state,
                multiple_choice=self.multiple_correct
            )
        )

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        """如果is_mcq，就exact match；否则按class来，class为1是Y/N，class为2是数值问题
        """
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction
        )

        try:
            if task_state.metadata['is_mcq'] == True:
                metric_scorer = get_metric("exact_match")
                score.explanation = f"exact match from {filtered_prediction}, {reference}"
            else:
                # if class == 1
                # yes/no exact match
                # elif class == 2
                # numeric
                if task_state.metadata['class'] == 1:
                    metric_scorer = get_metric("exact_match")
                    score.explanation = f"non_mcq exact_match from {filtered_prediction}, {reference}"
                elif task_state.metadata['class'] == 2:
                    metric_scorer = get_metric("internal_numeric_acc")
                    score.explanation = f"internal_numeric_acc from {filtered_prediction}, {reference}"
                else:
                    metric_scorer = get_metric("exact_match")
                    score.explanation = f"fallback exact_match from {filtered_prediction}, {reference}"
            metric_func = metric_scorer()
            metric_score = metric_func(
                prediction=filtered_prediction,
                reference=reference,
            )
            score.explanation += f"\nmetric is {metric_score}"
            score.value['acc'] = metric_score
        except Exception as e:
            # Handle evaluation errors
            score.value['acc'] = 0
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata.update({'error': str(e)})

        return score
