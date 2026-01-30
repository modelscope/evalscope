from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = """
## Overview

The CoNLL++ dataset is a corrected and cleaner version of the test set from the widely-used CoNLL2003 NER benchmark. It provides improved annotation quality for evaluating named entity recognition systems on news text.

## Task Description

- **Task Type**: Named Entity Recognition (NER)
- **Input**: News article text
- **Output**: Identified entity spans with types
- **Domain**: News articles, general domain

## Key Features

- Corrected version of CoNLL2003 test set
- Higher annotation quality than original
- Standard NER entity types (PER, ORG, LOC, MISC)
- Widely used benchmark for NER evaluation
- Comparable results with original CoNLL2003

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PER, ORG, LOC, MISC
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='conllpp',
        pretty_name='CoNLL++',
        dataset_id='extraordinarylab/conllpp',
        tags=[Tags.KNOWLEDGE, Tags.NER],
        description=DESCRIPTION,
        few_shot_num=5,
        train_split='train',
        eval_split='test',
        metric_list=['precision', 'recall', 'f1_score', 'accuracy'],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CoNLLPPAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define CoNLLPP-specific entity mappings
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
