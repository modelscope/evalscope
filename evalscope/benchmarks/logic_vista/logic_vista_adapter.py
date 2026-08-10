# flake8: noqa: E501
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = """
## Overview

LogicVista evaluates the fundamental logical reasoning abilities of multimodal large language models in visual contexts. Every item is a multiple-choice question whose answer options are drawn inside the image (diagrams, puzzles, sequences, charts), so a model must read the visual options and reason over them rather than over textual choices.

## Task Description

- **Task Type**: Visual Logical Reasoning (Multiple Choice)
- **Input**: Image containing the labelled answer options + question text
- **Output**: The label(s) of the chosen option(s)
- **Domain**: Abstract and diagrammatic logical reasoning

## Key Features

- 448 human-annotated visual multiple-choice questions collected from aptitude and reasoning tests
- Five reasoning skills used as subsets: inductive, deductive, numerical, spatial and mechanical
- Answer options live in the image and their label range varies per question (typically A-D or A-E, up to A-I)
- A handful of questions are multi-select (e.g. "which two proposals complete the diagram"), whose ground truth is a set of labels

## Evaluation Notes

- Default evaluation uses the **test** split and reports **Accuracy** overall and per reasoning skill
- Chain-of-thought prompting is used; the label is read from the final `ANSWER:` line and multi-select answers are compared as an unordered set, matching the official scoring rule
- Allow a generous `max_tokens`: when a reply is truncated before its `ANSWER:` line, the label is recovered from the last capital letter of the reply, which is a lenient guess
- Two of the released 448 items cannot be scored as published: `v1_382` carries neither a question nor an answer and is skipped, and `v1_20` labels its options with digits, which the letter-based answer parser cannot match — the reference implementations behave the same way
"""

SUBSET_LIST = ['inductive', 'deductive', 'numerical', 'spatial', 'mechanical']

# The option labels are rendered inside the image, so the framework cannot list them as text.
# `{question}` is the only field available; the instruction must not contain a literal label
# sequence that `parse_answers` could match when a model echoes the prompt.
PROMPT_TEMPLATE = """Answer the following multiple choice question. The answer options are shown in the image.
The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the label of the option you choose. If more than one option is correct, list all of their labels on that line. Think step by step before answering.

{question}"""

# Widest label range observed in the dataset ("Select answers from A-I"). This only sizes the set
# of labels `parse_answers` accepts; the labels themselves are never shown to the model.
OPTION_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


@register_benchmark(
    BenchmarkMeta(
        name='logic_vista',
        pretty_name='LogicVista',
        dataset_id='evalscope/LogicVista',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2407.04973',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class LogicVistaAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    """Data adapter for evalscope/LogicVista.

    The answer options are part of the image, so the prompt is built here instead of through
    `prompt()`, and `choices` only carries the label alphabet used to validate parsed answers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True
        # A few questions ask for more than one option (ground truth such as 'A, C').
        self.multiple_correct = True

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        answer = record.get('answer', '').strip()
        if not answer:
            logger.warning(f'Record {record.get("id")} has no answer; skipping.')
            return []

        skill: List[str] = record.get('skill') or []
        if not skill:
            logger.warning(f'Record {record.get("id")} has no reasoning skill; skipping.')
            return []

        content_list: List[Content] = [
            ContentImage(image=self._image_bytes_to_base64(record['image']['bytes'])),
            ContentText(text=self.prompt_template.format(question=record['question'])),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=OPTION_LABELS,
            # Multi-select answers are stored as 'A, C'; normalize to the sorted form that
            # `MultiChoiceAdapter.extract_answer` produces.
            target=''.join(sorted(label.strip().upper() for label in answer.split(','))),
            subset_key=skill[0],
            metadata={'id': record.get('id')},
        )
