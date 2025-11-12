from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = (
    'MusicTrivia is a curated dataset of multiple-choice questions covering both classical and modern music topics. '
    'It includes questions about composers, musical periods, and popular artists, designed for evaluating '
    'factual recall and domain-specific music knowledge.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='music_trivia',
        pretty_name='MusicTrivia',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/music-trivia',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class MusicTriviaAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
