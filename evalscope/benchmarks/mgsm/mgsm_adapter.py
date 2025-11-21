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
        name='mgsm',
        pretty_name='MGSM',
        dataset_id='evalscope/mgsm',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems, proposed in the paper Language models are multilingual chain-of-thought reasoners.',
        subset_list=[
            'en', 'es', 'fr', 
            'de', 'ru', 'zh', 
            'ja', 'th', 'sw', 
            'bn', 'te'
            ],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class MGSMAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        target = str(record['answer_number'])

        return Sample(
            input=question,
            target=target,
            metadata={
                'reasoning': record['answer'],
                'equation_solution': record['equation_solution'],
            }
        )
    
    def sample_to_fewshot(self, sample: Sample) -> str:
        if sample.metadata:
            return (
                f'{sample.input}\n\nReasoning:\n' + f"{sample.metadata['reasoning']}\n\n" + f'ANSWER: {sample.target}'
            )
        else:
            return ''
    
    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
        
