# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.math_parser import extract_answer_with_code_block, math_equal, strip_answer_string
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='math_500',
        pretty_name='MATH-500',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        "MATH-500 is a benchmark for evaluating mathematical reasoning capabilities of AI models. It consists of 500 diverse math problems across five levels of difficulty, designed to test a model's ability to solve complex mathematical problems by generating step-by-step solutions and providing the correct final answer.",  # noqa: E501
        dataset_id='AI-ModelScope/MATH-500',
        subset_list=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
        metric_list=[{
            'acc': {}
        }],  # Not using numeric since we extract the answer in extract_answer method
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class Math500Adapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            subset_key=f"Level {record['level']}",
            metadata={
                'question_id': record['unique_id'],
                'solution': record['solution'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """
        Extract answer from prediction, supporting code blocks and math formulas.
        Priority: code block > boxed formula > other formats

        Supported code block formats:
        - ```python ... ```
        - ```math ... ```
        - ```latex ... ```
        - ``` ... ``` (without language tag)
        """
        logger.debug(f'[Math500Adapter.extract_answer] Prediction length: {len(prediction)}')
        result = extract_answer_with_code_block(prediction)
        logger.debug(f'[Math500Adapter.extract_answer] Result length: {len(result)}')
        return result

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Compare extracted answers using math_equal.
        No need to call extract_answer again since filtered_prediction is already extracted.
        """
        # filtered_prediction is already extracted, use it directly
        pred_answer = strip_answer_string(filtered_prediction)
        ref_answer = strip_answer_string(reference)

        # Use math_equal for symbolic math comparison
        is_correct = math_equal(pred_answer, ref_answer)

        score = Score(
            extracted_prediction=filtered_prediction, prediction=original_prediction, value={'acc': float(is_correct)}
        )

        return score
