from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = """
## Overview

Tweebank-NER is an English Twitter corpus created by annotating the syntactically-parsed Tweebank V2 with four types of named entities: Person, Organization, Location, and Miscellaneous. It addresses NER challenges in informal social media text.

## Task Description

- **Task Type**: Social Media Named Entity Recognition (NER)
- **Input**: Twitter text (tweets)
- **Output**: Identified entity spans with types
- **Domain**: Social media, informal text

## Key Features

- Based on Tweebank V2 syntactic annotations
- Four standard NER entity types
- Addresses informal language challenges
- Handles Twitter-specific text features (hashtags, mentions)
- Useful for social media NLP applications

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PER, ORG, LOC, MISC
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='tweebank_ner',
        pretty_name='TweeBankNER',
        dataset_id='extraordinarylab/tweebank-ner',
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
class TweeBankNERAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define Tweebank-NER-specific entity mappings
        self.entity_type_map = {'PER': 'person', 'ORG': 'organization', 'LOC': 'location', 'MISC': 'miscellaneous'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'PER': 'Names of persons.',
            'ORG': 'Names of organizations, including companies, institutions, and groups.',
            'LOC': 'Names of locations, such as countries, cities, and states.',
            'MISC': 'Miscellaneous named entities, including nationalities, events, and products.'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
