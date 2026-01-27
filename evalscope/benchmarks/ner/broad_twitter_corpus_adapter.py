from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = """
## Overview

BroadTwitterCorpus is a dataset of tweets collected over stratified times, places, and social uses. The goal is to represent a broad range of activities, giving a dataset more representative of the language used in this hardest of social media formats to process.

## Task Description

- **Task Type**: Social Media Named Entity Recognition (NER)
- **Input**: Diverse Twitter text (tweets)
- **Output**: Identified entity spans with types
- **Domain**: Social media, diverse contexts

## Key Features

- Stratified sampling across times, places, and uses
- Representative of diverse Twitter language
- Addresses challenges in social media NER
- Three standard NER entity types (PER, ORG, LOC)
- Useful for robust social media NLP evaluation

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PER, ORG, LOC
"""


@register_benchmark(
    BenchmarkMeta(
        name='broad_twitter_corpus',
        pretty_name='BroadTwitterCorpus',
        dataset_id='extraordinarylab/broad-twitter-corpus',
        tags=[Tags.KNOWLEDGE, Tags.NER],
        description=DESCRIPTION.strip(),
        few_shot_num=5,
        train_split='train',
        eval_split='test',
        metric_list=['precision', 'recall', 'f1_score', 'accuracy'],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class BroadTwitterCorpusAdapter(NERAdapter):
    """
    Adapter for the BroadTwitterCorpus Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the BroadTwitterCorpus dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define BroadTwitterCorpus-specific entity mappings
        self.entity_type_map = {'PER': 'person', 'ORG': 'organization', 'LOC': 'location'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'PER': 'Names of people, including first and last names',
            'ORG': 'Names of companies, institutions, organizations, etc.',
            'LOC': 'Names of locations, cities, states, countries, etc.',
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
