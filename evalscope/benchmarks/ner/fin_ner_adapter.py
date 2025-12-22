from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The FinNER dataset is a corpus of financial agreements from public '
    'U.S. Security and Exchange Commission (SEC) filings, annotated with '
    'Person, Organization, Location, and Miscellaneous entities to support '
    'information extraction for credit risk assessment.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='fin_ner',
        pretty_name='FinNER',
        dataset_id='extraordinarylab/fin-ner',
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
class FinNERAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define FinNER-specific entity mappings
        self.entity_type_map = {'PER': 'person', 'ORG': 'organization', 'LOC': 'location', 'MISC': 'miscellaneous'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'PER': 'Names of persons, including lenders and borrowers.',
            'ORG': 'Names of organizations, such as companies and financial institutions.',
            'LOC': 'Names of locations, including addresses.',
            'MISC': 'Miscellaneous named entities which do not belong to the previous three.'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
