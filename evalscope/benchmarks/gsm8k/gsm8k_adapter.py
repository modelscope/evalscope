# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".lstrip()  # noqa: E501

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}

""".lstrip() + PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='gsm8k',
        pretty_name='GSM8K',
        dataset_id='AI-ModelScope/gsm8k',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'GSM8K (Grade School Math 8K) is a dataset of grade school math problems, designed to evaluate the mathematical reasoning abilities of AI models.',  # noqa: E501
        subset_list=['main'],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        metric_list=['acc'],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class GSM8KAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        DELIM = '####'
        question = record['question']
        answer = record['answer'].split(DELIM)
        target = answer.pop().strip()
        reasoning = DELIM.join(answer)

        return Sample(input=question, target=target, metadata={'reasoning': reasoning.strip()})

    def sample_to_fewshot(self, sample: Sample) -> str:
        if sample.metadata:
            return (
                f'{sample.input}\n\nReasoning:\n' + f"{sample.metadata['reasoning']}\n\n" + f'ANSWER: {sample.target}'
            )
        else:
            return ''

    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.filters.extraction import RegexFilter

        regex = RegexFilter(regex_pattern=r'(-?[0-9.,]{2,})|(-?[0-9]+)', group_select=-1)
        res = regex(prediction)
        return res.replace(',', '').replace('+', '').strip().strip('.')
