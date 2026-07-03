# flake8: noqa: E501
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

DESCRIPTION = """
## Overview

BabyVision is a visual perception benchmark that evaluates the fundamental visual abilities of
multimodal large language models through tasks inspired by infant and early childhood visual
development. It focuses on fine-grained discrimination, spatial perception, visual pattern
recognition, and visual tracking.

## Task Description

- **Task Type**: Visual Perception (Choice + Fill-in-the-blank)
- **Input**: Image + question
- **Output**: Choice letter or free-form short answer
- **Domains**: Fine-grained discrimination, spatial perception, visual pattern recognition, visual tracking

## Key Features

- 388 test samples across 4 major visual ability categories and 22 subtypes
- Two answer types: choice (135 samples) and blank (253 samples)
- Subtypes include: Find the different, Find the same, Count clusters, Maze,
  3D cube unfold, Pattern completion, Paper folding, Rotation patterns, etc.
- Tests low-level visual perception rather than high-level reasoning or knowledge
- Includes Chain-of-Thought (CoT) reference for analysis

## Evaluation Notes

- Default evaluation uses the **train** split (388 samples, single split dataset)
- Primary metric: **Accuracy** via LLM-as-judge
- Subsets organized by `type` field (4 categories)
- LLM judge evaluates both choice and blank answer types uniformly
- Requires `judge_model_args` configuration for LLM judge
"""

SUBSET_LIST = [
    'Fine-grained Discrimination',
    'Spatial Perception',
    'Visual Pattern Recognition',
    'Visual Tracking',
]

# Official suffix appended to every question (instructs model to use boxed format)
ANSWER_FORMAT_SUFFIX = '\nThink about the question and give your final answer in \\boxed{Answer} format.'


@register_benchmark(
    BenchmarkMeta(
        name='baby_vision',
        pretty_name='BabyVision',
        dataset_id='evalscope/BabyVision',
        tags=[Tags.MULTI_MODAL, Tags.REASONING, Tags.QA],
        description=DESCRIPTION,
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='train',
    )
)
class BabyVisionAdapter(VisionLanguageAdapter):
    """Data adapter for evalscope/BabyVision.

    Handles both choice and blank answer types uniformly with LLM judge scoring.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._use_llm_judge = True
        self.reformat_subset = True
        self.save_metadata = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a BabyVision record to a Sample."""
        image = record.get('image')
        if image is None:
            logger.warning('Record missing image field.')
            return Sample(input='', target='')

        image_b64 = self._image_bytes_to_base64(image['bytes'], default_format='jpeg')

        question = record.get('question', '')
        ans_type = record.get('ansType', 'blank')
        options: List[str] = record.get('options', [])

        # Build prompt following official BabyVision evaluation protocol
        if ans_type == 'choice' and options:
            # Official format: question + "\nChoices:\n(A) ...\n(B) ..."
            choices_text = '\n'.join(f'({chr(65 + i)}) {opt}' for i, opt in enumerate(options))
            prompt_text = f'{question}\nChoices:\n{choices_text}{ANSWER_FORMAT_SUFFIX}'
        else:
            prompt_text = f'{question}{ANSWER_FORMAT_SUFFIX}'

        content_list: List[Content] = [
            ContentImage(image=image_b64),
            ContentText(text=prompt_text),
        ]

        # Determine the target answer
        if ans_type == 'choice':
            # choiceAns is 0-indexed: 0->A, 1->B, 2->C, ...
            choice_ans = record.get('choiceAns')
            target = chr(65 + int(choice_ans)) if choice_ans is not None else ''
        else:
            target = record.get('blankAns', '') or ''

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=str(target),
            subset_key=record.get('type', ''),
            metadata={
                'taskId': record.get('taskId'),
                'type': record.get('type', ''),
                'subtype': record.get('subtype', ''),
                'ansType': ans_type,
                'coT': record.get('coT', ''),
            },
        )
