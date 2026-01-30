from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = """
## Overview

The BC5CDR corpus is a manually annotated resource of 1,500 PubMed articles developed for the BioCreative V challenge, containing over 4,400 chemical mentions, 5,800 disease mentions, and 3,100 chemical-disease interactions.

## Task Description

- **Task Type**: Biomedical Named Entity Recognition (NER)
- **Input**: PubMed article text
- **Output**: Identified chemical and disease entity spans
- **Domain**: Pharmacology, medical informatics, toxicology

## Key Features

- 1,500 PubMed articles with expert annotations
- 4,400+ chemical mentions
- 5,800+ disease mentions
- 3,100+ chemical-disease interactions
- Benchmark from BioCreative V challenge

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: CHEMICAL, DISEASE
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='bc5cdr',
        pretty_name='BC5CDR',
        dataset_id='extraordinarylab/bc5cdr',
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
class BC5CDRAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define BC5CDR-specific entity mappings
        self.entity_type_map = {'CHEMICAL': 'chemical', 'DISEASE': 'disease'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'CHEMICAL': 'Names of chemicals and drugs',
            'DISEASE': 'Names of diseases, disorders, syndromes, and related pathological conditions'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
