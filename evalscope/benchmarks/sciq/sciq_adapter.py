from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = (
    'The SciQ dataset contains crowdsourced science exam questions about Physics, '
    'Chemistry and Biology, among others. For the majority of the questions, '
    'an additional paragraph with supporting evidence for the correct answer is provided.'
)  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='sciq',
        pretty_name='SciQ',
        tags=[Tags.READING_COMPREHENSION, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/sciq',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class SciQAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
