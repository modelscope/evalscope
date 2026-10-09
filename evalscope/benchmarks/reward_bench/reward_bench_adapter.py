import hashlib
import random
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character


@register_benchmark(
    BenchmarkMeta(
        name='reward_bench',
        pretty_name='RewardBench v1 (pairwise)',
        dataset_id='allenai/reward-bench',
        dataset_hub=HubType.HUGGINGFACE,
        default_subset='default',
        subset_list=[
            'alpacaeval-easy',
            'alpacaeval-length',
            'alpacaeval-hard',
            'mt-bench-easy',
            'mt-bench-med',
            'mt-bench-hard',
            'llmbar-natural',
            'llmbar-adver-neighbor',
            'llmbar-adver-GPTInst',
            'llmbar-adver-GPTOut',
            'llmbar-adver-manual',
            'refusals-dangerous',
            'refusals-offensive',
            'xstest-should-refuse',
            'xstest-should-respond',
            'donotanswer',
            'math-prm',
            'hep-cpp',
            'hep-go',
            'hep-java',
            'hep-js',
            'hep-python',
            'hep-rust',
        ],
        eval_split='filtered',
        train_split=None,
        evaluation_version='v1.0',
        supports_choice=True,
        choice_instructions='Which candidate response better answers the user prompt in `question`, considering correctness, helpfulness and safety?',
        tags=[Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
        description="""
## Overview

Pairwise answer preference evaluation using RewardBench v1 human and reference preferences.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: User prompt and two candidate responses
- **Output**: Preferred response label
- **Domain**: Answer quality and safety

## Key Features

- Public dataset: `allenai/reward-bench` on Hugging Face
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Uses the filtered v1 split and deterministic candidate-position shuffling. Reports per-subset and sample-weighted overall accuracy, not the official category-weighted leaderboard score. RewardBench v2, ties and best-of-N tasks are excluded.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.0
""",
    )
)
class RewardBenchAdapter(MultiChoiceAdapter):
    """Treat v1 preference pairs as deterministic two-option decisions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        candidates = [(record['chosen'], True), (record['rejected'], False)]
        content = '\0'.join([record['prompt'], record['chosen'], record['rejected']])
        seed = int.from_bytes(hashlib.sha256(content.encode('utf-8')).digest()[:8], 'big')
        random.Random(seed).shuffle(candidates)
        return Sample(
            input=f'User prompt:\n{record["prompt"]}',
            choices=[text for text, _ in candidates],
            target=answer_character(next(i for i, (_, correct) in enumerate(candidates) if correct)),
            subset_key=record['subset'],
            metadata={'source_id': record.get('id'), 'category': record['subset']},
        )
