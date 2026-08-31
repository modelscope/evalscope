from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

SUBSET_LIST = [
    'Trajectory Reasoning',
    'Action Reasoning',
    'Pointing',
    'State Estimation',
    'Spatial Reasoning',
    'Multi-view Reasoning',
    'Task Reasoning',
    'Other',
]

PROMPT_TEMPLATE = """{question}"""


@register_benchmark(
    BenchmarkMeta(
        name='erqa',
        pretty_name='ERQA',
        tags=[Tags.MULTI_MODAL, Tags.MULTIPLE_CHOICE, Tags.REASONING],
        description="""
## Overview

ERQA (Embodied Reasoning QA) is a benchmark for evaluating spatial reasoning and embodied understanding capabilities of multimodal large language models. It tests models' ability to reason about trajectories, actions, spatial relationships, and task planning in egocentric robotic scenarios.

## Task Description

- **Task Type**: Embodied Spatial Reasoning (Multiple Choice)
- **Input**: Egocentric image(s) + multiple-choice question (A/B/C/D)
- **Output**: Single answer letter (A/B/C/D)
- **Domain**: Robotics, spatial reasoning, embodied AI

## Key Features

- 400 questions across 8 reasoning categories
- Multi-image support (some questions require reasoning across multiple views)
- Categories: Trajectory Reasoning, Action Reasoning, Pointing, State Estimation, Spatial Reasoning, Multi-view Reasoning, Task Reasoning, Other
- Egocentric perspective from robotic manipulation scenarios

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Answers are single letters (A/B/C/D)
""",
        dataset_id='evalscope/ERQA',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        evaluation_version='v1.1',
    )
)
class ERQAAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record.get('question', '')
        answer = record.get('answer', '')

        content_list: List[Content] = [ContentText(text=question)]

        for image in self._extract_media(record, 'image').values():
            content_list.append(self._content_image_from_value(image))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=['A', 'B', 'C', 'D'],
            target=answer,
            subset_key=record.get('question_type'),
            metadata={
                'question_id': record.get('question_id'),
                'question_type': record.get('question_type'),
            },
        )
