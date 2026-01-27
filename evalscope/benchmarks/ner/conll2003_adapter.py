from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='conll2003',
        pretty_name='CoNLL2003',
        dataset_id='extraordinarylab/conll2003',
        tags=[Tags.KNOWLEDGE, Tags.NER],
        description="""
## Overview

CoNLL-2003 is a classic Named Entity Recognition (NER) benchmark introduced at the Conference on Computational Natural Language Learning 2003. It contains news articles annotated with four entity types.

## Task Description

- **Task Type**: Named Entity Recognition (NER)
- **Input**: Text with entities to identify
- **Output**: Entity spans with type labels
- **Entity Types**: Person (PER), Organization (ORG), Location (LOC), Miscellaneous (MISC)

## Key Features

- Standard NER benchmark with well-defined entity types
- News domain text with high annotation quality
- Four entity categories with clear definitions
- Supports few-shot evaluation
- Comprehensive metrics (precision, recall, F1, accuracy)

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: **Precision**, **Recall**, **F1 Score**, **Accuracy**
- Train split: **train**, Eval split: **test**
- Entity types mapped to human-readable names
""",
        few_shot_num=5,
        train_split='train',
        eval_split='test',
        metric_list=['precision', 'recall', 'f1_score', 'accuracy'],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CoNLL2003Adapter(NERAdapter):
    """
    Adapter for the CoNLL2003 Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the CoNLL2003 dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define CoNLL2003-specific entity mappings
        self.entity_type_map = {'PER': 'person', 'ORG': 'organization', 'LOC': 'location', 'MISC': 'miscellaneous'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'PER': 'Names of people, including first and last names',
            'ORG': 'Names of companies, institutions, organizations, etc.',
            'LOC': 'Names of locations, cities, states, countries, etc.',
            'MISC': 'Miscellaneous entities not in the above categories'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
