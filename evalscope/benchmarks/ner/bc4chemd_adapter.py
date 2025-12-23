from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The BC4CHEMD (BioCreative IV CHEMDNER) dataset is a corpus of 10,000 '
    'PubMed abstracts with 84,355 chemical entity mentions manually annotated '
    'by experts for chemical named entity recognition.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='bc4chemd',
        pretty_name='BC4CHEMD',
        dataset_id='extraordinarylab/bc4chemd',
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
class BC4CHEMDAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define BC4CHEMD-specific entity mappings
        self.entity_type_map = {'CHEMICAL': 'chemical'}

        # Add descriptions for each entity type
        self.entity_descriptions = {'CHEMICAL': 'Names of chemicals and drugs'}

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
