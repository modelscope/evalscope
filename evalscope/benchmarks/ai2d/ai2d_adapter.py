from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, prompt

logger = get_logger()

SUBSET_LIST = ['default']

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT


@register_benchmark(
    BenchmarkMeta(
        name='ai2d',
        pretty_name='AI2D',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

AI2D (AI2 Diagrams) is a benchmark dataset for evaluating AI systems' ability to understand and reason about scientific diagrams. It contains over 5,000 diverse diagrams from science textbooks covering topics like the water cycle, food webs, and biological processes.

## Task Description

- **Task Type**: Diagram Understanding and Visual Reasoning
- **Input**: Scientific diagram image + multiple-choice question
- **Output**: Correct answer choice
- **Domains**: Science education, visual reasoning, diagram comprehension

## Key Features

- Diagrams sourced from real science textbooks
- Requires joint understanding of visual layouts, symbols, and text labels
- Tests interpretation of relationships between diagram elements
- Multiple-choice format with challenging distractors
- Covers diverse scientific domains (biology, physics, earth science)

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Requires understanding both textual labels and visual elements
""",  # noqa: E501
        dataset_id='lmms-lab/ai2d',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class Ai2dAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        answers_list: list[str] = record['options']
        input_text = prompt(question=record['question'], choices=answers_list, template=self.prompt_template)
        content_list: list[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))

        label_answer = chr(int(record['answer']) + ord('A'))

        return Sample(input=[ChatMessageUser(content=content_list)], choices=answers_list, target=label_answer)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        answers = parse_answers(task_state)
        return ''.join(sorted(list(answers)))
