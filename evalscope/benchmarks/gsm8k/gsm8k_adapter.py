# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should display the answer enclosed within \\boxed{{\\text{{$ANSWER}}}}.

Example:

Let's solve the problem step by step.

Problem: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?

Step 1: Calculate Eliza's earnings for the first 40 hours. Eliza's hourly rate is $10, so her earnings for the first 40 hours are $10/hour x 40 hours = $400.
Step 2: Calculate Eliza's overtime pay rate. Eliza's overtime pay rate is 1.2 times her regular hourly rate, so her overtime pay rate is $10/hour x 1.2 = $12/hour.
Step 3: Calculate Eliza's earnings for the overtime hours. Eliza worked for 45 hours, so her overtime hours are 45 hours - 40 hours = 5 hours. Her earnings for the overtime hours are $12/hour x 5 hours = $60.
Step 4: Calculate Eliza's total earnings for the week. Eliza's total earnings for the week are her earnings for the first 40 hours plus her earnings for the overtime hours, which is $400 + $60 = $460.

Answer:
\\boxed{{\\text{{460}}}}

question:
{question}

Remember to put your answer on its own line at the end in the form "\\boxed{{\\text{{$ANSWER}}}}" (without quotes), where $ANSWER is replaced by the actual answer to the problem.
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
        boxed_match = re.search(r'\\boxed\\{\\text\\{([^}]*)\\}\\}', prediction)
        if boxed_match:
            result = boxed_match.group(1).strip()
            return result.strip()

        from evalscope.filters.extraction import RegexFilter

        regex = RegexFilter(regex_pattern=r'(-?[0-9.,]{2,})|(-?[0-9]+)', group_select=-1)
        res = regex(prediction)
        return res.replace(',', '').replace('+', '').strip().strip('.')
