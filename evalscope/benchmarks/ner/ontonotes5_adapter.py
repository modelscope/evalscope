from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'OntoNotes Release 5.0 is a large, multilingual corpus containing text in English, '
    'Chinese, and Arabic across various genres like news, weblogs, and broadcast '
    'conversations. It is richly annotated with multiple layers of linguistic information, '
    'including syntax, predicate-argument structure, word sense, named entities, and '
    'coreference to support research and development in natural language processing.'
)


@register_benchmark(
    BenchmarkMeta(
        name='ontonotes5',
        pretty_name='OntoNotes5',
        dataset_id='extraordinarylab/ontonotes5',
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
class OntoNotes5Adapter(NERAdapter):
    """
    Adapter for the OntoNotes5 Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the OntoNotes5 dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define OntoNotes5-specific entity mappings
        self.entity_type_map = {
            'CARDINAL': 'cardinal',
            'DATE': 'date',
            'EVENT': 'event',
            'FAC': 'facility',
            'GPE': 'geopolitical_entity',
            'LANGUAGE': 'language',
            'LAW': 'law',
            'LOC': 'location',
            'MONEY': 'money',
            'NORP': 'nationalities_or_religious_or_political_groups',
            'ORDINAL': 'ordinal',
            'ORG': 'organization',
            'PERCENT': 'percent',
            'PERSON': 'person',
            'PRODUCT': 'product',
            'QUANTITY': 'quantity',
            'TIME': 'time',
            'WORK_OF_ART': 'work_of_art'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'PERSON': 'People, including fictional',
            'NORP': 'Nationalities or religious or political groups',
            'FAC': 'Buildings, airports, highways, bridges, etc.',
            'ORG': 'Companies, agencies, institutions, etc.',
            'GPE': 'Countries, cities, states',
            'LOC': 'Non-GPE locations, mountain ranges, bodies of water',
            'PRODUCT': 'Vehicles, weapons, foods, etc. (Not services)',
            'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
            'WORK_OF_ART': 'Titles of books, songs, etc.',
            'LAW': 'Named documents made into laws',
            'LANGUAGE': 'Any named language',
            'DATE': 'Absolute or relative dates or periods',
            'TIME': 'Times smaller than a day',
            'PERCENT': 'Percentage (including "%")',
            'MONEY': 'Monetary values, including unit',
            'QUANTITY': 'Measurements, as of weight or distance',
            'ORDINAL': '"first", "second"',
            'CARDINAL': 'Numerals that do not fall under another type'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
