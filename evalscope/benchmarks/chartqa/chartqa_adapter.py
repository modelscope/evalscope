import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

OPEN_PROMPT = """
{question}

The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the a single word answer or number to the problem.
"""


@register_benchmark(
    BenchmarkMeta(
        name='chartqa',
        pretty_name='ChartQA',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

ChartQA is a benchmark designed to evaluate question-answering capabilities over charts and data visualizations. It tests both visual reasoning and logical understanding of various chart types including bar charts, line graphs, and pie charts.

## Task Description

- **Task Type**: Chart Question Answering
- **Input**: Chart image + natural language question
- **Output**: Single word or numerical answer
- **Domains**: Data visualization, visual reasoning, numerical reasoning

## Key Features

- Covers diverse chart types (bar, line, pie, scatter plots)
- Includes both human-written and augmented test questions
- Requires understanding of chart structure and data relationships
- Tests both visual extraction and logical reasoning abilities
- Questions range from simple data lookup to complex reasoning

## Evaluation Notes

- Default evaluation uses the **test** split with two subsets:
  - `human_test`: Human-written questions
  - `augmented_test`: Automatically generated questions
- Primary metric: **Relaxed Accuracy** (allows minor variations in answers)
- Answers should be in format "ANSWER: [ANSWER]"
- Numerical answers may have tolerance for rounding differences
""",  # noqa: E501
        dataset_id='lmms-lab/ChartQA',
        subset_list=['human_test', 'augmented_test'],
        metric_list=['relaxed_acc'],
        eval_split='test',
        prompt_template=OPEN_PROMPT,
    )
)
class ChartQAAdapter(VisionLanguageAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_aggregation_name = False
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        image_data = record['image']
        image_base64 = bytes_to_base64(image_data['bytes'], format='png', add_header=True)

        content_list: List[Content] = [
            ContentText(text=OPEN_PROMPT.format(question=question)),
            ContentImage(image=image_base64)
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['answer'],
            subset_key=record['type'],  # 'human_test' or 'augmented_split'
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        pattern = r'ANSWER:\s*(.*)'
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()
        return ''

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        from .utils import relaxed_correctness

        score = relaxed_correctness(filtered_prediction, reference)
        score = 1.0 if score else 0.0

        return Score(
            value={'relaxed_acc': score},
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
        )
