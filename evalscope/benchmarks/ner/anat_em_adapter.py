from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = """
## Overview

The AnatEM corpus is an extensive resource for anatomical entity recognition, created by extending and combining previous corpora. It includes over 13,000 annotations across 1,212 biomedical documents, focusing on identifying anatomical structures from subcellular components to organ systems.

## Task Description

- **Task Type**: Biomedical Named Entity Recognition (NER)
- **Input**: Biomedical text from PubMed abstracts
- **Output**: Identified anatomical entity spans with types
- **Domain**: Anatomy, biomedical literature

## Key Features

- Over 13,000 anatomical entity annotations
- 1,212 biomedical documents from PubMed
- Comprehensive anatomical coverage (cells to organs)
- Manual expert annotation
- Useful for biomedical text mining

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: ANATOMY (subcellular structures, cells, tissues, organs)
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='anat_em',
        pretty_name='AnatEM',
        dataset_id='extraordinarylab/anat-em',
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
class AnatEMAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define AnatEM-specific entity mappings
        self.entity_type_map = {'ANATOMY': 'anatomy'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'ANATOMY':
            'Anatomical entities, ranging from subcellular structures and cells to tissues, organs, and organ systems'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
