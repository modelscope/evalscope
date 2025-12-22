from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='conll2003',
        pretty_name='CoNLL2003',
        dataset_id='extraordinarylab/conll2003',
        tags=[Tags.KNOWLEDGE, Tags.NER],
        description='The ConLL-2003 dataset is for the Named Entity Recognition (NER) task. It was introduced as part '
        'of the ConLL-2003 Shared Task conference and contains texts annotated with entities such as '
        'people, organizations, places, and various names.',
        few_shot_num=5,
        train_split='train',
        eval_split='test',
        metric_list=['precision', 'recall', 'f1_score', 'accuracy'],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CoNLL2003Adapter(NERAdapter):
    """
    Adapter for the CoNLL2003 Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the CoNLL2003 dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define CoNLL2003-specific entity mappings
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
