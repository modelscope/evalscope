# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='gsm8k',
        pretty_name='GSM8K',
        dataset_id='AI-ModelScope/gsm8k',
        tags=['Mathematics'],
        description=
        'GSM8K (Grade School Math 8K) is a dataset of grade school math problems, designed to evaluate the mathematical reasoning abilities of AI models.',  # noqa: E501
        subset_list=['main'],
        metric_list=['AverageAccuracy'],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        prompt_template="Question: {query}\nLet's think step by step\nAnswer:",
    ))
class GSM8KAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict) -> Sample:
        DELIM = '####'
        input = record['question']
        answer = record['answer'].split(DELIM)
        target = answer.pop().strip()
        reasoning = DELIM.join(answer)
        return Sample(input=input, target=target, metadata={'reasoning': reasoning.strip()})

    def sample_to_fewshot(self, sample: Sample) -> str:
        if sample.metadata:
            return (f'{sample.input}\n\nReasoning:\n' + f"{sample.metadata['reasoning']}\n\n"
                    + f'ANSWER: {sample.target}')
        else:
            return ''

    def generate_prompts(self):
        return super().generate_prompts()
