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
        }],  # 不使用 numeric，因为我们已经在 extract_answer 中提取了
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
        提取答案，支持代码块和数学公式的提取。
        优先级：代码块 > boxed公式 > 其他格式

        支持的代码块格式：
        - ```python ... ```
        - ```math ... ```
        - ```latex ... ```
        - ``` ... ``` (无语言标记)
        """
        logger.debug(f'[Math500Adapter.extract_answer] 预测长度: {len(prediction)}')
        result = extract_answer_with_code_block(prediction)
        logger.debug(f'[Math500Adapter.extract_answer] 返回结果长度: {len(result)}')
        return result

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        使用 math_equal 比较已提取的答案。
        不再次调用 extract_answer，因为 filtered_prediction 已经是提取后的答案。
        """
        # filtered_prediction 已经是提取后的答案，直接使用
        pred_answer = strip_answer_string(filtered_prediction)
        ref_answer = strip_answer_string(reference)

        # 使用 math_equal 进行数学表达式的符号比较
        is_correct = math_equal(pred_answer, ref_answer)

        score = Score(
            extracted_prediction=filtered_prediction, prediction=original_prediction, value={'acc': float(is_correct)}
        )

        return score
