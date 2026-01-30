from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

DESCRIPTION = """
## Overview

Drivelology Narrative Selection evaluates models' ability to understand the underlying narrative of "drivelology" text - linguistic utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.

## Task Description

- **Task Type**: Multiple-Choice Narrative Understanding
- **Input**: Drivelology text with multiple narrative interpretation options
- **Output**: Best option representing the underlying narrative
- **Domain**: Linguistic analysis, narrative comprehension

## Key Features

- Tests deep narrative understanding
- Requires interpretation of layered meanings
- Multiple-choice format with challenging distractors
- Easy and hard difficulty levels
- Tests cultural and contextual understanding

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Simple accuracy metric
- Subsets: multiple-choice-english-easy, multiple-choice-english-hard
"""

PROMPT_TEMPLATE = r"""
Tell me the best option in the following options which represents the underlying narrative of the text?
The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
""".strip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='drivel_selection',
        pretty_name='DrivelologyNarrativeSelection',
        tags=[Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/drivel-hub',
        subset_list=['multiple-choice-english-easy', 'multiple-choice-english-hard'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class DrivelologyNarrativeSelectionAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_overall_metric = False

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['text'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
