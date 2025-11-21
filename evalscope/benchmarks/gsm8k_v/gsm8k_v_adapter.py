# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

import json


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

@register_benchmark(
    BenchmarkMeta(
        name='gsm8k_v',
        pretty_name='GSM8K-V',
        dataset_id='evalscope/GSM8K-V',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTI_MODAL],
        description=
        'GSM8K-V is a purely visual multi-image mathematical reasoning benchmark that systematically maps each GSM8K math word problem into its visual counterpart to enable a clean, within-item comparison across modalities.',
        subset_list=['default'],
        eval_split='train',
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class GSM8KVAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['modify_scene_related_question']
        content_list: List[Content] = []
        input_text = self.prompt_template.format(question=question).strip()
        target = str(record['answer'])
        content_list.append(ContentText(text=input_text))

        images = record.get('image')
        if images:
            base64_list = json.loads(images)
            for image_value in base64_list:
                content_list.append(ContentImage(image=f'data:image/jpeg;base64,{image_value}'))


        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=target,
            metadata={
                'index': record['index'],
                'category': record['category'],
                'subcategory': record['subcategory'],
                'original_question': record['original_question']
            }
        )
    
    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)