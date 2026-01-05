from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The NCBI disease corpus is a manually annotated resource of '
    'PubMed abstracts designed for disease name recognition and normalization.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='ncbi',
        pretty_name='NCBI',
        dataset_id='extraordinarylab/ncbi',
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
class NCBIAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define NCBI-specific entity mappings
        self.entity_type_map = {'DISEASE': 'disease'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'DISEASE': 'Names of diseases, disorders, syndromes, and related pathological conditions'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
