from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The BC2GM (BioCreative II Gene Mention) dataset is a widely used corpus '
    'for gene mention recognition, consisting of 20,000 sentences from MEDLINE '
    'abstracts where gene and protein names have been manually annotated.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='bc2gm',
        pretty_name='BC2GM',
        dataset_id='extraordinarylab/bc2gm',
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
class BC2GMAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define BC2GM-specific entity mappings
        self.entity_type_map = {'GENE': 'gene'}

        # Add descriptions for each entity type
        self.entity_descriptions = {'GENE': 'Names of genes and proteins'}

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
