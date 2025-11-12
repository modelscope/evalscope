from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='winogrande',
        pretty_name='Winogrande',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=
        'Winogrande is a benchmark for evaluating AI models on commonsense reasoning tasks, specifically designed to test the ability to resolve ambiguous pronouns in sentences.',  # noqa: E501
        dataset_id='AI-ModelScope/winogrande_val',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class WinograndeAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['sentence'],
            choices=[record['option1'], record['option2']],
            target=chr(ord('A') + int(record['answer']) - 1),  # Convert 1,2 to A,B
            metadata={'id': record.get('id', 'unknown')},
        )
