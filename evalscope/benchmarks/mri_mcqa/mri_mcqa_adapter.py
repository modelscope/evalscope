from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = (
    'MRI-MCQA is a benchmark composed by multiple-choice questions related to Magnetic Resonance Imaging (MRI).'
)


@register_benchmark(
    BenchmarkMeta(
        name='mri_mcqa',
        pretty_name='MRI-MCQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MEDICAL],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/mri-mcqa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class MRIMCQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
