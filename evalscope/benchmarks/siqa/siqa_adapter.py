from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTOIN = (
    'Social Interaction QA (SIQA) is a question-answering benchmark for testing social commonsense intelligence. '
    'Contrary to many prior benchmarks that focus on physical or taxonomic knowledge, Social IQa focuses on '
    "reasoning about people's actions and their social implications."
)


@register_benchmark(
    BenchmarkMeta(
        name='sciq',
        pretty_name='SciQ',
        tags=[Tags.READING_COMPREHENSION, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTOIN.strip(),
        dataset_id='extraordinarylab/sciq',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class SciQAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
