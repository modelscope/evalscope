from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

MusicTrivia is a curated multiple-choice benchmark for evaluating AI models on music knowledge. It covers both classical and modern music topics including composers, musical periods, instruments, and popular artists.

## Task Description

- **Task Type**: Multiple-Choice Question Answering
- **Input**: Music-related trivia question with multiple choice options
- **Output**: Selected correct answer
- **Domains**: Classical music, modern music, music history

## Key Features

- Comprehensive coverage of music domains
- Questions about composers, periods, and artists
- Tests factual recall and domain knowledge
- Curated for quality and accuracy
- Balanced difficulty across topics

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- Evaluates on **test** split
- Uses standard single-answer multiple-choice template
"""


@register_benchmark(
    BenchmarkMeta(
        name='music_trivia',
        pretty_name='MusicTrivia',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/music-trivia',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class MusicTriviaAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
