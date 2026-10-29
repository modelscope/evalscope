# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from collections import defaultdict
from typing import Any, Dict, List

from evalscope.api.benchmark import AudioLanguageAdapter, BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentAudio, ContentText
from evalscope.api.metric import MetricSelector
from evalscope.api.metric.scorer import AggScore, SampleScore
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import extract_answer_line

logger = get_logger()

LETTERS = ['A', 'B', 'C', 'D']

MMAU_PROMPT = r"""Answer the following multiple choice question based on the audio content. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='mmau',
        pretty_name='MMAU',
        dataset_id='lmms-lab/mmau',
        hf_dataset_id='lmms-lab/mmau',
        tags=[Tags.AUDIO, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

MMAU (Massive Multitask Audio Understanding) is a comprehensive benchmark for evaluating audio understanding capabilities of multimodal large language models across diverse audio tasks.

## Task Description

- **Task Type**: Audio Understanding (Multiple Choice)
- **Input**: Audio recordings with multiple-choice questions
- **Output**: Correct answer choice (A/B/C/D)
- **Domain**: Speech, Sound, Music

## Key Features

- Large-scale audio understanding benchmark
- Covers multiple audio domains (speech, environmental sounds, music)
- Multiple-choice format with 4 options
- Scores the `test_mini` split; the full `test` split ships without answers (`'NA'`)
- Per-category accuracy reporting

## Evaluation Notes

- Default configuration uses **test_mini** split
- Primary metric: **Accuracy** (exact match on predicted letter)
- Uses the last `ANSWER:` line containing only a single choice letter (case-insensitive, with optional brackets or Markdown bold); a bare choice letter is also accepted.
- Unparseable replies score zero. Natural-language answers are not mapped to choices.
- Evaluation version **v1.1** excludes the unlabeled `test` split and removes guessed answers.
- Reports overall accuracy and per-task-category accuracy
- Prompt includes chain-of-thought instruction
- Requires an OpenAI-compatible server that supports the `input_audio` content part for audio input.
""",
        subset_list=['test_mini'],
        eval_split='test_mini',
        metric_list=['acc'],
        primary_metric=MetricSelector(name='accuracy', dimensions={'scope': 'overall'}),
        prompt_template=MMAU_PROMPT,
        evaluation_version='v1.1',
    )
)
class MMAUAdapter(AudioLanguageAdapter, MultiChoiceAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_as_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        choices = json.loads(record['choices']) if isinstance(record['choices'], str) else record['choices']
        answer_text = record['answer']

        # Map answer text to letter
        matches = [i for i, choice in enumerate(choices) if choice.strip().lower() == answer_text.strip().lower()]
        if not matches:
            raise ValueError(f'MMAU answer {answer_text!r} is not one of the choices {choices}.')
        answer_char = LETTERS[matches[0]]

        # Format choices string
        letters_str = ', '.join(LETTERS[: len(choices)])
        choices_str = '\n'.join([f'{LETTERS[i]}. {c}' for i, c in enumerate(choices)])
        input_text = MMAU_PROMPT.format(
            letters=letters_str,
            question=question,
            choices=choices_str,
        )

        content_list: List[Content] = [ContentText(text=input_text)]

        # Add audio
        audio_base64 = bytes_to_base64(record['audio']['bytes'], format='wav', add_header=True, content_type='audio')
        content_list.append(ContentAudio(audio=audio_base64, format='wav'))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=choices,
            target=answer_char,
            metadata={
                'task': record.get('task', ''),
                'answer': answer_text,
                'audio_id': record.get('audio_id', ''),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Read the last explicit single-choice answer without guessing from prose."""
        for line in reversed(prediction.splitlines()):
            answer = extract_answer_line(line.replace('**', ''))
            if answer is not None:
                letter = answer.strip(' *().[]').upper()
                if letter in LETTERS:
                    return letter
        letter = prediction.strip().strip(' *().[]').upper()
        return letter if letter in LETTERS else ''

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate scores with overall and per-task category accuracy."""
        total = len(sample_scores)
        if total == 0:
            return [AggScore(metric_name='accuracy', dimensions={'scope': 'overall'}, score=0.0, num=0)]

        total_correct = sum(ss.score.main_value for ss in sample_scores)
        overall_acc = total_correct / total

        agg_scores: List[AggScore] = [
            AggScore(metric_name='accuracy', dimensions={'scope': 'overall'}, score=overall_acc, num=total)
        ]

        # Per-task category accuracy
        group_totals: Dict[str, int] = defaultdict(int)
        group_correct: Dict[str, float] = defaultdict(float)
        for ss in sample_scores:
            task = (ss.sample_metadata or {}).get('task', 'unknown')
            group_totals[task] += 1
            group_correct[task] += ss.score.main_value

        for task_name in sorted(group_totals.keys()):
            task_acc = group_correct[task_name] / group_totals[task_name]
            agg_scores.append(
                AggScore(
                    metric_name='accuracy',
                    dimensions={'task': task_name},
                    score=task_acc,
                    num=group_totals[task_name],
                )
            )
            logger.info(f'MMAU {task_name} accuracy: {task_acc:.2%}')

        logger.info(f'MMAU overall accuracy: {overall_acc:.2%}')
        return agg_scores
