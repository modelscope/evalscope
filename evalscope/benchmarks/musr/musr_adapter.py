import ast
from typing import Any

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='musr',
        pretty_name='MuSR',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=
        'MuSR is a benchmark for evaluating AI models on multiple-choice questions related to murder mysteries, object placements, and team allocation.',  # noqa: E501
        dataset_id='AI-ModelScope/MuSR',
        metric_list=['acc'],
        subset_list=['murder_mysteries', 'object_placements', 'team_allocation'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MuSRAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.split_as_subset = True

    def record_to_sample(self, record) -> Sample:
        choices = ast.literal_eval(record['choices'])
        choice_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        target_letter = choice_letters[record['answer_index']]

        return Sample(
            input=f"{record['narrative']}\n\n{record['question']}",
            choices=choices,
            target=target_letter,
        )
