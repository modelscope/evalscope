from typing import Any, Dict, Type

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import DataLoader, Dataset, Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character


@register_benchmark(
    BenchmarkMeta(
        name='anli',
        pretty_name='ANLI',
        dataset_id='facebook/anli',
        default_subset='plain_text',
        subset_list=['r1', 'r2', 'r3'],
        eval_split='test',
        train_split='train',
        evaluation_version='v1.0',
        supports_choice=True,
        choice_instructions='Which relationship holds between the premise and hypothesis in `question`?',
        tags=[Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
        description="""
## Overview

Adversarial natural language inference over three independently collected rounds.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: Premise and hypothesis
- **Output**: Entailment, neutral or contradiction
- **Domain**: English language understanding

## Key Features

- Public dataset: `facebook/anli` on ModelScope
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Reports accuracy per round and a sample-weighted overall score. Test explanations are excluded from model inputs.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.0
""",
    )
)
class ANLIAdapter(MultiChoiceAdapter):
    """Expose the three ANLI rounds without merging their source splits."""

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        with self._temporary_attribute('eval_split', f'{self.eval_split}_{subset}'):
            return super().load_subset(self.default_subset, data_loader)

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        with self._temporary_attribute('train_split', f'{self.train_split}_{subset}'):
            return super().load_fewshot_subset(self.default_subset, data_loader)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        label = int(record['label'])
        if label not in (0, 1, 2):
            raise ValueError('ANLI requires an entailment, neutral or contradiction label.')
        return Sample(
            input=f'Premise:\n{record["premise"]}\n\nHypothesis: {record["hypothesis"]}',
            choices=[
                'Entailment: the hypothesis follows from the premise.',
                'Neutral: the premise does not determine whether the hypothesis is true.',
                'Contradiction: the hypothesis conflicts with the premise.',
            ],
            target=answer_character(label),
            metadata={'uid': record.get('uid'), 'round': self.current_subset_name},
        )
