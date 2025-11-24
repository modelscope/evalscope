# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.

""".lstrip()  # noqa: E501

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}

""".lstrip() + PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='mgsm',
        pretty_name='MGSM',
        dataset_id='evalscope/mgsm',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTI_LINGUAL],
        description=
        'Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems, proposed in the paper Language models are multilingual chain-of-thought reasoners.',  # noqa: E501
        subset_list=['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te'],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class MGSMAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        target = str(record['answer_number'])

        return Sample(
            input=question,
            target=target,
            metadata={
                'reasoning': record['answer'],
                'equation_solution': record['equation_solution'],
            }
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        if sample.metadata:
            return (
                f'{sample.input}\n\nReasoning:\n' + f"{sample.metadata['reasoning']}\n\n" + f'ANSWER: {sample.target}'
            )
        else:
            return ''

    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
