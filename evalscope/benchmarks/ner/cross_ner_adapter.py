from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.ner.cross_ner_entities import ai, literature, music, politics, science
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = """
## Overview

CrossNER is a fully-labeled collection of named entity recognition (NER) data spanning over five diverse domains: AI, Literature, Music, Politics, and Science. It enables cross-domain NER evaluation and domain adaptation research.

## Task Description

- **Task Type**: Cross-Domain Named Entity Recognition (NER)
- **Input**: Text from five specialized domains
- **Output**: Domain-specific entity spans
- **Domain**: AI, Literature, Music, Politics, Science

## Key Features

- Five diverse domain subsets
- Domain-specific entity types per subset
- Enables cross-domain transfer evaluation
- Fully labeled with expert annotations
- Useful for domain adaptation research

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Subsets: ai, literature, music, politics, science
- Entity types vary by domain subset
"""


SUBSET_ENTITIES = {'ai': ai, 'literature': literature, 'music': music, 'politics': politics, 'science': science}


@register_benchmark(
    BenchmarkMeta(
        evaluation_version='v1.2',
        name='cross_ner',
        pretty_name='CrossNER',
        dataset_id='extraordinarylab/cross-ner',
        hf_dataset_id='extraordinarylab/cross-ner',
        subset_list=['ai', 'literature', 'music', 'politics', 'science'],
        tags=[Tags.KNOWLEDGE, Tags.NER],
        description=DESCRIPTION.strip(),
        few_shot_num=5,
        train_split='train',
        eval_split='test',
        metric_list=['precision', 'recall', 'f1_score', 'accuracy'],
        primary_metric='f1',
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CrossNERAdapter(NERAdapter):
    """
    Adapter for the CrossNER Named Entity Recognition dataset.

    Each domain subset has its own entity types. Prompts are built and predictions scored after every
    subset has been loaded, so they take the labels from the sample's own subset, not from adapter state.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define CrossNER-specific entity mappings
        self.entity_type_map = {}

        # Add descriptions for each entity type
        self.entity_descriptions = {}

    def setup_entity_mappings(self, subset: Optional[str] = None) -> None:
        """Set up the entity labels of ``subset``, by default the subset being loaded."""
        module = SUBSET_ENTITIES.get(subset or self.current_subset_name)
        if module is not None:
            self.entity_type_map, self.entity_descriptions = module.get_entity_mappings()
        super().setup_entity_mappings()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Annotate the target with the loading subset's labels and remember that subset."""
        self.setup_entity_mappings()
        sample = super().record_to_sample(record)
        sample.metadata['subset'] = self.current_subset_name
        return sample

    def format_prompt_template(self, sample: Sample) -> str:
        self.setup_entity_mappings(sample.metadata['subset'])
        return super().format_prompt_template(sample)

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        self.setup_entity_mappings(sample.metadata['subset'])
        return super().format_fewshot_template(fewshot, sample)

    def get_reverse_entity_map(self, task_state: TaskState) -> Dict[str, str]:
        entity_type_map, _ = SUBSET_ENTITIES[task_state.metadata['subset']].get_entity_mappings()
        return {v.lower(): k for k, v in entity_type_map.items()}
