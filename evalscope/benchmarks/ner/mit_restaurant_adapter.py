from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The MIT-Restaurant dataset is a collection of restaurant review text specifically '
    'curated for training and testing Natural Language Processing (NLP) models, '
    'particularly for Named Entity Recognition (NER). It contains sentences from real '
    'reviews, along with corresponding labels in the BIO format.'
)


@register_benchmark(
    BenchmarkMeta(
        name='mit_restaurant',
        pretty_name='MIT-Restaurant',
        dataset_id='extraordinarylab/mit-restaurant',
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
class MITRestaurantAdapter(NERAdapter):
    """
    Adapter for the MIT-Restaurant Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the MIT-Restaurant dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define MIT-Restaurant-specific entity mappings
        self.entity_type_map = {
            'AMENITY': 'amenity',
            'CUISINE': 'cuisine',
            'DISH': 'dish',
            'HOURS': 'hours',
            'LOCATION': 'location',
            'PRICE': 'price',
            'RATING': 'rating',
            'RESTAURANT_NAME': 'restaurant_name'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'AMENITY': 'A feature or service offered by the restaurant.',
            'CUISINE': 'The type of food a restaurant serves.',
            'DISH': 'A specific food or drink item.',
            'HOURS': 'The operating hours of a restaurant.',
            'LOCATION': 'The address or general location of a restaurant.',
            'PRICE': 'The price range of a restaurant.',
            'RATING': 'A rating or review of the restaurant.',
            'RESTAURANT_NAME': 'The name of a restaurant.',
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
