from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'TweetNER7 is a large-scale NER dataset featuring over 11,000 tweets from '
    '2019-2021, annotated with seven entity types to facilitate the study of '
    'short-term temporal shifts in social media language.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='tweet_ner_7',
        pretty_name='TweetNER7',
        dataset_id='extraordinarylab/tweet-ner-7',
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
class TweetNER7Adapter(NERAdapter):

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define TweetNER7-specific entity mappings
        self.entity_type_map = {
            'CORPORATION': 'corporation',
            'CREATIVE_WORK': 'creative_work',
            'EVENT': 'event',
            'GROUP': 'group',
            'LOCATION': 'location',
            'PERSON': 'person',
            'PRODUCT': 'product'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'CORPORATION': 'Names of corporations and companies.',
            'CREATIVE_WORK': 'Titles of creative works like books, songs, and movies.',
            'EVENT': 'Names of specific events, such as festivals or sports events.',
            'GROUP': 'Names of groups, such as musical bands or political groups.',
            'LOCATION': 'Names of locations, such as cities, countries, and landmarks.',
            'PERSON': 'Names of people.',
            'PRODUCT': 'Names of commercial products.'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
