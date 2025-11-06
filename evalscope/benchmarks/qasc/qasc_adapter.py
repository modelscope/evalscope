from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = (
    'QASC is a question-answering dataset with a focus on sentence composition. '
    'It consists of 9,980 8-way multiple-choice questions about grade school science.'
)


@register_benchmark(
    BenchmarkMeta(
        name='qasc',
        pretty_name='QASC',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/qasc',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class QASCAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
