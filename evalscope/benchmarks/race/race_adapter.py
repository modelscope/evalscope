# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# flake8: noqa

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='race',
        pretty_name='RACE',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=
        'RACE is a benchmark for testing reading comprehension and reasoning abilities of neural models. It is constructed from Chinese middle and high school examinations.',  # noqa: E501
        dataset_id='evalscope/race',
        metric_list=['acc'],
        subset_list=['high', 'middle'],
        few_shot_num=3,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class RACEAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.few_shot_num > 3:
            logger.warning(f'few_shot_num <= 3 for RACE, but got {self.few_shot_num}. Use 3-shot by default.')
            self.few_shot_num = 3

    def record_to_sample(self, record) -> Sample:
        # Format the article and question as context
        context = f"Article:\n{record['article']}\nQuestion:\n{record['question']}"

        return Sample(
            input=context,
            choices=record['options'],
            target=record['answer'],
            metadata={'example_id': record.get('example_id', 'unknown')},
        )
