from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

DESCRIPTION = (
    'Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - '
    'utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, '
    'or rhetorically subversive.'
)

PROMPT_TEMPLATE = r"""
Tell me the best option in the following options which represents the underlying narrative of the text?
The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
""".strip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='drivel_selection',
        pretty_name='DrivelologyNarrativeSelection',
        tags=[Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/drivel-hub',
        subset_list=['multiple-choice-english-easy', 'multiple-choice-english-hard'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class DrivelologyNarrativeSelectionAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_overall_metric = False

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['text'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
