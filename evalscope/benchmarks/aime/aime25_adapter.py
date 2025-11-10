# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

JUDGE_PROMPT = """
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: {expression1}
    Expression 2: {expression2}

"""

PROMPT_TEMPLATE = """
Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{question}

Remember to put your answer inside \\boxed{{}}."""


@register_benchmark(
    BenchmarkMeta(
        name='aime25',
        pretty_name='AIME-2025',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'The AIME 2025 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model\'s ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.',
        dataset_id='opencompass/AIME2025',
        subset_list=['AIME2025-I', 'AIME2025-II'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class AIME25Adapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            target=record['answer'],
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """
        Args:
            prediction (str): The model prediction to extract from
            task_state (TaskState): The task state for additional context

        Returns:
            str: The extracted answer
        """
        from evalscope.metrics.math_parser import extract_answer
        from .math_normalize import normalize_answer

        extracted_pred = extract_answer(prediction)
        filtered_pred = normalize_answer(extracted_pred)
        return filtered_pred

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from evalscope.metrics.math_parser import extract_answer
        from .grader import grade_answer

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Use the custom grade_answer function for evaluation
        try:
            is_correct = grade_answer(extract_answer(original_prediction), reference)
            accuracy_score = 1.0 if is_correct else 0.0
            score.value['acc'] = accuracy_score
        except Exception as e:
            logger.error(f'Error in custom grading: {e}')
            score.value['acc'] = 0.0
            score.metadata['acc'] = f'grading_error: {str(e)}'
        return score

    def llm_match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        judge_prompt = JUDGE_PROMPT.format(expression1=original_prediction, expression2=reference)

        # Request judge and obtain score
        judge_response = self.llm_judge.judge(prompt=judge_prompt)

        # Parse judge response to get accuracy score
        is_correct = bool(re.search(r'\bYes\b', judge_response, re.IGNORECASE))
        score.value = {
            'acc': 1.0 if is_correct else 0.0,
        }
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
        }
        score.main_score_name = 'acc'
        return score
