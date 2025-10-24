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
        from evalscope.metrics.math_parser import extract_answer as math_extract_answer

        # DEBUG: 添加日志
        print(f"\n{'='*80}")
        print('🔍🔍🔍 [DEBUG] Math500Adapter.extract_answer 被调用！')
        print(f'预测长度: {len(prediction)}')
        print(f'预测前100字符: {prediction[:100]}')
        print(f"{'='*80}\n")

        # 1. 尝试提取代码块中的内容（支持多种语言标记，包括无标记的情况）
        code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)```', prediction, re.DOTALL)
        if code_blocks:
            # 如果有多个代码块，取最后一个（通常是最终答案）
            code_content = code_blocks[-1].strip()
            # 从代码块中提取答案
            extracted = math_extract_answer(code_content)
            if extracted:
                return extracted

        # 2. 尝试提取单行代码格式（如：`answer`）
        inline_code = re.findall(r'`([^`]+)`', prediction)
        if inline_code:
            # 取最后一个内联代码
            inline_content = inline_code[-1].strip()
            extracted = math_extract_answer(inline_content)
            if extracted:
                return extracted

        # 3. 如果没有代码块，使用原有的提取逻辑（支持 \boxed{}, "答案是" 等格式）
        result = math_extract_answer(prediction)

        print(f"\n{'='*80}")
        print('✅✅✅ [DEBUG] Math500Adapter.extract_answer 返回！')
        print(f"返回结果: '{result}'")
        print(f'返回结果长度: {len(result)}')
        print(f"{'='*80}\n")

        return result

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        使用 math_equal 比较已提取的答案。
        不再次调用 extract_answer，因为 filtered_prediction 已经是提取后的答案。
        """
        from evalscope.metrics.math_parser import math_equal, strip_answer_string

        # filtered_prediction 已经是提取后的答案，直接使用
        pred_answer = strip_answer_string(filtered_prediction)
        ref_answer = strip_answer_string(reference)

        # 使用 math_equal 进行数学表达式的符号比较
        is_correct = math_equal(pred_answer, ref_answer)

        score = Score(
            extracted_prediction=filtered_prediction, prediction=original_prediction, value={'acc': float(is_correct)}
        )

        return score
