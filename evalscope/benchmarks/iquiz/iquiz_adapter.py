from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='iquiz',
        pretty_name='IQuiz',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.CHINESE],
        description=
        'IQuiz is a benchmark for evaluating AI models on IQ and EQ questions. It consists of multiple-choice questions where the model must select the correct answer and provide an explanation.',  # noqa: E501
        dataset_id='AI-ModelScope/IQuiz',
        metric_list=['acc'],
        subset_list=['IQ', 'EQ'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT,
    )
)
class IQuizAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={'level': record.get('level', 'unknown')},
        )
