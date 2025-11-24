# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.api.benchmark import BenchmarkMeta, WOChoiceMultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import WOChoiceMultipleChoiceTemplate

from .utils import _extract_answer
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
        metric_list=['acc'],
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
            input=record['question'],
            target=record['answer'],
            metadata={
                'id': record.get('id', 'unknown'),
                'is_mcq': record.get('is_mcq', False)
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
        return _extract_answer(
            prediction=prediction,
            task_state=task_state,
            multiple_choice=self.multiple_correct
        )
