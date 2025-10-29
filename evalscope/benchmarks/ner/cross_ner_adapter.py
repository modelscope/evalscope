from typing import Any, Dict, List, Set, Tuple

from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.ner.cross_ner_entities import ai, literature, music, politics, science
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE, create_target_text

DESCRIPTION = (
    'CrossNER is a fully-labelled collected of named entity recognition (NER) data '
    'spanning over five diverse domains (AI, Literature, Music, Politics, Science).'
)


@register_benchmark(
    BenchmarkMeta(
        name='cross_ner',
        pretty_name='CrossNER',
        dataset_id='extraordinarylab/cross-ner',
        subset_list=['ai', 'literature', 'music', 'politics', 'science'],
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
class CrossNERAdapter(NERAdapter):
    """
    Adapter for the CrossNER Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the CrossNER dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define CrossNER-specific entity mappings
        self.entity_type_map = {}

        # Add descriptions for each entity type
        self.entity_descriptions = {}

    def setup_entity_mappings(self):
        """
        Setup entity mappings and descriptions for prompt formatting.
        This should be called after entity_type_map and entity_descriptions are defined.
        """
        if self.current_subset_name == 'ai':
            self.entity_type_map, self.entity_descriptions = ai.get_entity_mappings()
        elif self.current_subset_name == 'literature':
            self.entity_type_map, self.entity_descriptions = literature.get_entity_mappings()
        elif self.current_subset_name == 'music':
            self.entity_type_map, self.entity_descriptions = music.get_entity_mappings()
        elif self.current_subset_name == 'politics':
            self.entity_type_map, self.entity_descriptions = politics.get_entity_mappings()
        elif self.current_subset_name == 'science':
            self.entity_type_map, self.entity_descriptions = science.get_entity_mappings()

        # Reverse mapping for converting back from prediction to evaluation
        self.reverse_entity_map = {v.lower(): k for k, v in self.entity_type_map.items()}

        # Create list of tags for prompt formatting
        self.entity_list = [f'<{ent.lower()}>' for ent in self.entity_type_map.values()]

        # Create description of entities for prompt
        self.entities_description = ', '.join([
            f'{self.entity_type_map[tag]} ({self.entity_descriptions[tag]})' for tag in self.entity_type_map
        ])

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a record with tokens and NER tags into a Sample.
        Creates both the raw text input and annotated text target.
        """
        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()

        tokens: List[str] = record['tokens']
        ner_tags: List[str] = record['ner_tags']

        # Create the input text by joining tokens
        input_text = ' '.join(tokens)

        # Process tokens and tags to create annotated target text
        target_text = create_target_text(tokens, ner_tags, self.entity_type_map)

        # Store tokens and tags in metadata for evaluation
        metadata = {'tokens': tokens, 'ner_tags': ner_tags}

        return Sample(input=input_text, target=target_text, metadata=metadata)

    def format_prompt_template(self, sample):
        """
        Format the prompt with entity types, available tags, and text to annotate.
        """
        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
        return self.prompt_template.format(
            entities=self.entities_description, entity_list=', '.join(self.entity_list), text=sample.input
        )

    def format_fewshot_template(self, fewshot, sample):
        """
        Format the few-shot prompt with all required parameters.
        """
        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
        return self.few_shot_prompt_template.format(
            fewshot=fewshot,
            entities=self.entities_description,
            entity_list=', '.join(self.entity_list),
            text=sample.input
        )
