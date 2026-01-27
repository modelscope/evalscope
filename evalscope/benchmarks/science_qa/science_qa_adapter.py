from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, parse_answers, prompt

logger = get_logger()

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT


@register_benchmark(
    BenchmarkMeta(
        name='science_qa',
        pretty_name='ScienceQA',
        dataset_id='AI-ModelScope/ScienceQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description="""
## Overview

ScienceQA is a multimodal benchmark consisting of multiple-choice science questions derived from elementary and high school curricula. It covers diverse subjects including natural science, social science, and language science, with questions accompanied by both image and text contexts.

## Task Description

- **Task Type**: Multimodal Science Question Answering
- **Input**: Question with optional image context + multiple choices
- **Output**: Correct answer choice letter
- **Domains**: Natural science, social science, language science

## Key Features

- Questions sourced from real K-12 science curricula
- Most questions include both image and text contexts
- Annotated with detailed lectures and explanations
- Supports research into chain-of-thought reasoning
- Covers multiple grade levels and difficulty ranges
- Rich metadata including topic, skill, and category information

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Metadata includes solution explanations for analysis
- Questions span grades from elementary to high school
""",  # noqa: E501
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class ScienceQAAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question: str = record.get('question', '')

        answers_list: List[str] = record['choices']
        content_list: List[Content] = []
        input_text = prompt(question=question, choices=answers_list, template=self.prompt_template)
        content_list.append(ContentText(text=input_text))

        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))

        target = answer_character(record['answer'])

        metadata: Dict[str, Any] = {
            'hint': record.get('hint'),
            'task': record.get('task'),
            'grade': record.get('grade'),
            'subject': record.get('subject'),
            'topic': record.get('topic'),
            'category': record.get('category'),
            'skill': record.get('skill'),
            'lecture': record.get('lecture'),
            'solution': record.get('solution'),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=target,
            metadata=metadata,
        )
