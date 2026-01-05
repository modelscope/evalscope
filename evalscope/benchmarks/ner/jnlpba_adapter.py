from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The JNLPBA dataset is a widely-used resource for bio-entity recognition, '
    'consisting of 2,404 MEDLINE abstracts from the GENIA corpus annotated for '
    'five key molecular biology entity types.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='jnlpba',
        pretty_name='JNLPBA',
        dataset_id='extraordinarylab/jnlpba',
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
class JNLPBAAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define JNLPBA-specific entity mappings
        self.entity_type_map = {
            'PROTEIN': 'protein',
            'DNA': 'dna',
            'RNA': 'rna',
            'CELL_LINE': 'cell_line',
            'CELL_TYPE': 'cell_type'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'PROTEIN': 'Names of proteins, protein families, or protein complexes',
            'DNA': 'Names of DNA molecules, domains, or regions',
            'RNA': 'Names of RNA molecules',
            'CELL_LINE': 'Names of specific, cultured cell lines',
            'CELL_TYPE': 'Names of naturally occurring cell types'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
