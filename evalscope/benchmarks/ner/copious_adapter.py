from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'Copious corpus is a gold standard corpus that covers a wide range of biodiversity '
    'entities, consisting of 668 documents downloaded from the Biodiversity Heritage '
    'Library with over 26K sentences and more than 28K entities.'
)


@register_benchmark(
    BenchmarkMeta(
        name='copious',
        pretty_name='Copious',
        dataset_id='extraordinarylab/copious',
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
class CopiousAdapter(NERAdapter):
    """
    Adapter for the Copious Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the Copious dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define Copious-specific entity mappings
        self.entity_type_map = {
            'TAXON': 'taxon',
            'GEOGRAPHICAL_LOCATION': 'geographical_location',
            'HABITAT': 'habitat',
            'PERSON': 'person',
            'TEMPORAL_EXPRESSION': 'temporal_expression'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'TAXON': (
                'Mentions of taxonomic ranks such as species, genus, and family. '
                'This includes scientific names (e.g., "Salvelinus alpinus") and '
                'vernacular names (e.g., "flying fox"), but excludes general terms '
                'like "fish" or "birds" and microorganism names.'
            ),
            'GEOGRAPHICAL_LOCATION': (
                'Identifiable points or areas on the planet, including continents, '
                'countries, cities, landforms, and bodies of water (e.g., "East coast '
                'of Mindoro", "Balayan Bay"). This also includes geographical '
                'coordinates (e.g., "13o 36\' 11\\" N.").'
            ),
            'HABITAT': (
                'Descriptions of environments where organisms live. This includes '
                'natural environments (e.g., "Lowland forest", "subalpine calcareous '
                'pastures") and places where parasites or epiphytes reside (e.g., '
                '"parasitic on Achillea holosericea"). It excludes habitat attributes '
                'like altitude or depth.'
            ),
            'PERSON': (
                'Proper nouns referring to person names, including those in historical '
                'accounts or citations related to a species observation (e.g., "In 1905, '
                '[Tattersall] follows..."). It excludes titles, general references like '
                '"the researcher", and names that are part of a taxon\'s authority.'
            ),
            'TEMPORAL_EXPRESSION': (
                'Spans of text referring to points in time. This includes specific dates '
                '(e.g., "10 June 2013"), years, decades, seasons, and geochronological ages '
                '(e.g., "late Pleistocene"). It excludes time-of-day information and dates '
                'within a taxon name\'s authority.'
            )
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
