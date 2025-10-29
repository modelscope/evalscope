from evalscope.api.benchmark import BenchmarkMeta, NERAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.ner import FEWSHOT_TEMPLATE, PROMPT_TEMPLATE

DESCRIPTION = (
    'The MIT-Movie-Trivia dataset, originally created for slot filling, is modified by '
    'ignoring some slot types (e.g. genre, rating) and merging others (e.g. director '
    'and actor in person, and song and movie title in title) in order to keep '
    'consistent named entity types across all datasets.'
)


@register_benchmark(
    BenchmarkMeta(
        name='mit_movie_trivia',
        pretty_name='MIT-Movie-Trivia',
        dataset_id='extraordinarylab/mit-movie-trivia',
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
class MITMovieTriviaAdapter(NERAdapter):
    """
    Adapter for the MIT-Movie-Trivia Named Entity Recognition dataset.

    This adapter inherits the NER functionality from NERAdapter and
    configures it specifically for the MIT-Movie-Trivia dataset's entity types.
    """

    def __init__(self, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)

        # Define MIT-Movie-Trivia-specific entity mappings
        self.entity_type_map = {
            'ACTOR': 'actor',
            'AWARD': 'award',
            'CHARACTER_NAME': 'character_name',
            'DIRECTOR': 'director',
            'GENRE': 'genre',
            'OPINION': 'opinion',
            'ORIGIN': 'origin',
            'PLOT': 'plot',
            'QUOTE': 'quote',
            'RELATIONSHIP': 'relationship',
            'SOUNDTRACK': 'soundtrack',
            'YEAR': 'year'
        }

        # Add descriptions for each entity type
        self.entity_descriptions = {
            'ACTOR': 'The name of an actor or actress starring in the movie.',
            'AWARD': 'An award the movie won or was nominated for.',
            'CHARACTER_NAME': 'The name of a character in the movie.',
            'DIRECTOR': 'The name of the person who directed the movie.',
            'GENRE': 'The category or style of the movie.',
            'OPINION': 'A subjective review or personal opinion about the movie.',
            'ORIGIN': 'The source material or basis for the movie.',
            'PLOT': 'A description or summary of the movie\'s storyline.',
            'QUOTE': 'A memorable line or phrase spoken in the movie.',
            'RELATIONSHIP': 'The connection or relationship between characters.',
            'SOUNDTRACK': 'The music or a specific song from the movie.',
            'YEAR': 'The release year of the movie.'
        }

        # Setup entity mappings based on the defined entity types
        self.setup_entity_mappings()
