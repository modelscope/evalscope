from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'GeniaNER consisting of 2,000 MEDLINE abstracts has been released with more than '
    '400,000 words and almost 100,000 annotations for biological terms.'
)


@register_benchmark(
    BenchmarkMeta(
        name='genia_ner',
        pretty_name='GeniaNER',
        dataset_id='extraordinarylab/genia-ner',
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
class GeniaNERAdapter(NERAdapter):
    """
    Adapter for the GeniaNER Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the GeniaNER dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define GeniaNER-specific entity mappings
        self.entity_type_map = {
            'CELL_LINE': 'cell_line',
            'CELL_TYPE': 'cell_type',
            'DNA': 'dna',
            'PROTEIN': 'protein',
            'RNA': 'rna'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'CELL_LINE':
            'A population of cells derived from a single cell and grown in a culture.',
            'CELL_TYPE':
            ('A category of cells that are part of a larger organism and share a specific '
             'structure and function.'),
            'DNA':
            'Deoxyribonucleic acid. This includes specific genes, domains, and regions of a DNA molecule.',
            'PROTEIN': (
                'Molecules composed of amino acids that perform a vast array of functions within '
                'organisms. This includes enzymes, receptors, and signaling molecules.'
            ),
            'RNA':
            'Ribonucleic acid. This refers to RNA molecules, including messenger RNA (mRNA) and other types.'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
