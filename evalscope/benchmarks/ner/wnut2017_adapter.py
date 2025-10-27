from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The WNUT2017 dataset is a collection of user-generated text from various social '
    'media platforms, like Twitter and YouTube, specifically designed for a named-entity '
    'recognition task.'
)


@register_benchmark(
    BenchmarkMeta(
        name='wnut2017',
        pretty_name='WNUT2017',
        dataset_id='extraordinarylab/wnut2017',
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
class WNUT2017Adapter(NERAdapter):
    """
    Adapter for the WNUT2017 Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the WNUT2017 dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define WNUT2017-specific entity mappings
        self.entity_type_map = {
            'CORPORATION': 'corporation',
            'CREATIVE-WORK': 'creative_work',
            'GROUP': 'group',
            'LOCATION': 'location',
            'PERSON': 'person',
            'PRODUCT': 'product'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'CORPORATION': 'Named companies, businesses, agencies, and other institutions.',
            'CREATIVE-WORK': 'Named books, songs, movies, paintings, and other works of art.',
            'GROUP': 'Named groups of people, such as sports teams, bands, or political groups.',
            'LOCATION': 'Named geographical locations, such as cities, countries, and natural landmarks.',
            'PERSON': 'Named individuals, including both real and fictional people.',
            'PRODUCT': 'Named commercial products, including vehicles, software, and other goods.'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
