from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character


@register_benchmark(
    BenchmarkMeta(
        name='contract_nli',
        pretty_name='ContractNLI (classification)',
        dataset_id='evalscope/contract-nli',
        default_subset='contractnli_b',
        subset_list=['contractnli_b'],
        eval_split='test',
        train_split='train',
        evaluation_version='v1.0',
        supports_choice=True,
        choice_instructions='Which relationship holds between the full contract and hypothesis in `question`?',
        tags=[Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
        description="""
## Overview

Document-level natural language inference on complete non-disclosure agreements.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: Full contract and hypothesis
- **Output**: Entailment, contradiction or not mentioned
- **Domain**: Legal document understanding

## Key Features

- Public dataset: `evalscope/contract-nli` on ModelScope
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Uses the full-document contractnli_b data-only mirror. Each contract-hypothesis pair is one sample. Reports classification accuracy only; evidence extraction and official F1 metrics are not evaluated.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.0
""",
    )
)
class ContractNLIAdapter(MultiChoiceAdapter):
    """Evaluate full-document NLI classification, without evidence extraction."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        label = int(record['label'])
        if label not in (0, 1, 2):
            raise ValueError('ContractNLI requires a three-class NLI label.')
        contract, hypothesis = record['premise'], record['hypothesis']
        return Sample(
            input=f'Contract:\n{contract}\n\nHypothesis: {hypothesis}',
            choices=[
                'Contradiction: the contract contradicts the hypothesis.',
                'Entailment: the contract supports the hypothesis.',
                'Not mentioned: the contract neither supports nor contradicts the hypothesis.',
            ],
            target=answer_character(label),
        )
