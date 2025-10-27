from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = 'PIQA addresses the challenging task of reasoning about physical commonsense in natural language.'


@register_benchmark(
    BenchmarkMeta(
        name='piqa',
        pretty_name='PIQA',
        tags=[Tags.REASONING, Tags.COMMONSENSE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/piqa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class PIQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
