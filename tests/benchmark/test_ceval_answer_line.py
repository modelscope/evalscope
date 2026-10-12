"""C-Eval reads the "答案：[LETTER]" line its prompt asks for, in the forms CMMLU already accepts."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize(
    'answer_line',
    ['答案：B', '答案：**B**', '答案: B', '答案:B', '答案： B', '答案：(B)', '答案：（B）'],
)
def test_answer_line_variants(answer_line):
    adapter = get_benchmark('ceval', TaskConfig(model='mock', datasets=['ceval']))
    completion = f'根据题意分析，选项 B 正确。\n{answer_line}'
    state = TaskState(
        model='mock',
        sample=Sample(input='q', choices=['甲', '乙', '丙', '丁'], target='B'),
        output=ModelOutput.from_content(model='mock', content=completion),
    )
    assert adapter.extract_answer(completion, state) == 'B'
