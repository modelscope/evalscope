from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = """
## Overview

HarveyNER is a dataset with fine-grained locations annotated in tweets, collected during Hurricane Harvey. It presents unique challenges with complex and long location mentions in informal crisis-related descriptions.

## Task Description

- **Task Type**: Crisis-Domain Location Named Entity Recognition (NER)
- **Input**: Hurricane Harvey-related tweets
- **Output**: Fine-grained location entity spans
- **Domain**: Crisis communication, disaster response

## Key Features

- Fine-grained location annotations in tweets
- Complex and long location mentions
- Informal crisis-related text
- Four location entity types (AREA, POINT, RIVER, ROAD)
- Useful for disaster response NLP applications

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: AREA, POINT, RIVER, ROAD
"""


@register_benchmark(
    BenchmarkMeta(
        name='harvey_ner',
        pretty_name='HarveyNER',
        dataset_id='extraordinarylab/harvey-ner',
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
class HarveyNERAdapter(NERAdapter):
    """
    Adapter for the HarveyNER Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the HarveyNER dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define HarveyNER-specific entity mappings
        self.entity_type_map = {'AREA': 'area', 'POINT': 'point', 'RIVER': 'river', 'ROAD': 'road'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'AREA':
            'Geographical entities such as city subdivisions, neighborhoods, etc.',
            'POINT': (
                'An exact location that a geocoordinate can be assigned. E.g., a uniquely named '
                'building, intersections of roads or rivers.'
            ),
            'RIVER':
            'A river or a section of a river.',
            'ROAD':
            'A road or a section of a road.'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
