# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI, Inc. and its affiliates.

import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

PROMPT_TEMPLATE = """
Problem:
{question}

Please reason step by step, and put your final answer within \\boxed{{}}.
""".lstrip()

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}
""".lstrip() + PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='competition_math',
        pretty_name='MATH',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'The MATH (Mathematics) benchmark is designed to evaluate the mathematical reasoning abilities of AI models through a variety of problem types, including arithmetic, algebra, geometry, and more.',
        dataset_id='evalscope/competition_math',
        subset_list=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CompetitionMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        from evalscope.metrics.math_parser import extract_answer

        return Sample(
            input=record['problem'],
            target=extract_answer(record['solution']),
            subset_key=record['level'],
            metadata={
                'reasoning': record.get('solution', ''),
                'type': record.get('type', ''),
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

        # DEBUG: 添加日志（使用 print 确保一定能看到）
        print(f"\n{'='*80}")
        print(f'🔍🔍🔍 [DEBUG] CompetitionMathAdapter.extract_answer 被调用！')
        print(f'预测长度: {len(prediction)}')
        print(f'预测前100字符: {prediction[:100]}')
        print(f"{'='*80}\n")
        logger.warning(f'🔍 [CompetitionMathAdapter.extract_answer] 被调用！预测长度: {len(prediction)}')

        # 1. 尝试提取代码块中的内容（支持多种语言标记，包括无标记的情况）
        # 匹配 ``` 或 ```language 开头的代码块
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
        print(f'✅✅✅ [DEBUG] CompetitionMathAdapter.extract_answer 返回！')
        print(f"返回结果: '{result}'")
        print(f'返回结果长度: {len(result)}')
        print(f"{'='*80}\n")
        logger.warning(f'✅ [CompetitionMathAdapter.extract_answer] 返回结果长度: {len(result)}')
        return result

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'Problem:\n{sample.input}\nSolution:\n{sample.target}'
