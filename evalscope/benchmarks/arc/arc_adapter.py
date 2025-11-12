# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='arc',
        pretty_name='ARC',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=
        'The ARC (AI2 Reasoning Challenge) benchmark is designed to evaluate the reasoning capabilities of AI models through multiple-choice questions derived from science exams. It includes two subsets: ARC-Easy and ARC-Challenge, which vary in difficulty.',  # noqa: E501
        dataset_id='allenai/ai2_arc',
        subset_list=['ARC-Easy', 'ARC-Challenge'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class ARCAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        # Convert choice labels to indices (A->0, B->1, etc.)
        choice_texts = record['choices']['text']
        answer_key = record['answerKey']

        return Sample(
            input=record['question'],
            choices=choice_texts,
            target=answer_key,
            metadata={
                'id': record.get('id', ''),
            },
        )
