# flake8: noqa: E501

from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'MultiNERD is a large-scale, multilingual, and multi-genre dataset for '
    'fine-grained Named Entity Recognition, automatically generated from '
    'Wikipedia and Wikinews, covering 10 languages and 15 distinct entity categories.'
)


@register_benchmark(
    BenchmarkMeta(
        name='multi_nerd',
        pretty_name='MultiNERD',
        dataset_id='extraordinarylab/multi-nerd',
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
class MultiNERDAdapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define MultiNERD-specific entity mappings
        self.entity_type_map = {
            'PER': 'person',
            'ORG': 'organization',
            'LOC': 'location',
            'ANIM': 'animal',
            'BIO': 'biological_entity',
            'CEL': 'celestial_body',
            'DIS': 'disease',
            'EVE': 'event',
            'FOOD': 'food',
            'INST': 'instrument',
            'MEDIA': 'media',
            'MYTH': 'mythological_entity',
            'PLANT': 'plant',
            'TIME': 'time',
            'VEHI': 'vehicle'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'PER': 'People.',
            'ORG': 'Associations, companies, agencies, institutions, nationalities and religious or political groups.',
            'LOC':
            'Physical locations (e.g. mountains, bodies of water), geopolitical entities (e.g. cities, states), and facilities (e.g. bridges, buildings, airports).',
            'ANIM': 'Breeds of dogs, cats and other animals, including their scientific names.',
            'BIO': 'Genus of fungus, bacteria and protoctists, families of viruses, and other biological entities.',
            'CEL': 'Planets, stars, asteroids, comets, nebulae, galaxies and other astronomical objects.',
            'DIS':
            'Physical, mental, infectious, non-infectious, deficiency, inherited, degenerative, social and self-inflicted diseases.',
            'EVE': 'Sport events, battles, wars and other events.',
            'FOOD': 'Foods and drinks.',
            'INST': 'Technological instruments, mechanical instruments, musical instruments, and other tools.',
            'MEDIA': 'Titles of films, books, magazines, songs and albums, fictional characters and languages.',
            'MYTH': 'Mythological and religious entities.',
            'PLANT': 'Types of trees, flowers, and other plants, including their scientific names.',
            'TIME':
            'Specific and well-defined time intervals, such as eras, historical periods, centuries, years and important days. No months and days of the week.',
            'VEHI': 'Cars, motorcycles and other vehicles.'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
