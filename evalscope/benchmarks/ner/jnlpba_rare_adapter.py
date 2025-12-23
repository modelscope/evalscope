from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The JNLPBA-Rare dataset is a specialized subset of the JNLPBA test set '
    'created to evaluate zero-shot performance on its least frequent entity types, '
    'RNA and cell line.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='jnlpba_rare',
        pretty_name='JNLPBA-Rare',
        dataset_id='extraordinarylab/jnlpba-rare',
        tags=[Tags.KNOWLEDGE, Tags.NER],
        description=DESCRIPTION,
        few_shot_num=0,
        eval_split='test',
        metric_list=['precision', 'recall', 'f1_score', 'accuracy'],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class JNLPBARareAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define JNLPBA-Rare-specific entity mappings
        self.entity_type_map = {'RNA': 'rna', 'CELL_LINE': 'cell_line'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'RNA': 'Names of RNA molecules',
            'CELL_LINE': 'Names of specific, cultured cell lines'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
