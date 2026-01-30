from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, parse_answers, prompt

logger = get_logger()

SUBSET_LIST = ['default']

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT


@register_benchmark(
    BenchmarkMeta(
        name='a_okvqa',
        pretty_name='A-OKVQA',
        dataset_id='HuggingFaceM4/A-OKVQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description="""
## Overview

A-OKVQA (Augmented OK-VQA) is a benchmark designed to evaluate commonsense reasoning and external world knowledge in visual question answering. It extends beyond basic VQA tasks that rely solely on image content, requiring models to leverage a broad spectrum of commonsense and factual knowledge about the world.

## Task Description

- **Task Type**: Visual Question Answering with Knowledge Reasoning
- **Input**: Image + natural language question requiring external knowledge
- **Output**: Answer (multiple-choice or open-ended)
- **Domains**: Commonsense reasoning, factual knowledge, visual understanding

## Key Features

- Requires commonsense reasoning beyond direct visual observation
- Combines visual understanding with external world knowledge
- Includes both multiple-choice and open-ended question formats
- Questions annotated with rationales explaining the reasoning process
- More challenging than standard VQA benchmarks

## Evaluation Notes

- Default evaluation uses the **validation** split
- Primary metric: **Accuracy** for multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting for better reasoning
- Questions require reasoning beyond what is directly visible in images
""",  # noqa: E501
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        default_subset='default',
        eval_split='validation',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class AOkvqaAdapter(VisionLanguageAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question: str = record.get('question', '')
        answers_list: List[str] = record.get('choices', [])
        content_list: List[Content] = []
        input_text = prompt(question=question, choices=answers_list, template=self.prompt_template)
        content_list.append(ContentText(text=input_text))

        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))

        target = answer_character(record['correct_choice_idx'])

        metadata: Dict[str, Any] = {
            'question_id': record.get('question_id'),
            'direct_answers': record.get('direct_answers'),
            'difficult_direct_answer': record.get('difficult_direct_answer'),
            'rationales': record.get('rationales'),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=target,
            metadata=metadata,
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        answers = parse_answers(task_state)
        return ''.join(sorted(list(answers)))
