from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The CoNLL++ dataset is a corrected and cleaner version of the '
    'test set from the widely-used CoNLL2003 NER benchmark.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='conllpp',
        pretty_name='CoNLL++',
        dataset_id='extraordinarylab/conllpp',
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
class CoNLLPPAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define CoNLLPP-specific entity mappings
        self.entity_type_map = {'PER': 'person', 'ORG': 'organization', 'LOC': 'location', 'MISC': 'miscellaneous'}

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'PER': 'Names of people, including first and last names',
            'ORG': 'Names of companies, institutions, organizations, etc.',
            'LOC': 'Names of locations, cities, states, countries, etc.',
            'MISC': 'Miscellaneous entities not in the above categories'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
