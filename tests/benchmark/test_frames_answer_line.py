"""FRAMES reads the answer after the last "the answer is", in any case, and stops at the end of its line."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize(
    'reply',
    [
        'Therefore, the answer is Paris.',
        'Both sources agree.\nThe answer is Paris.',
        'Therefore, the answer is Paris.\n\nLet me know if you need more detail.',
        'Therefore, the answer is:\n\n**Paris**',
    ],
)
def test_answer_line_is_extracted(reply: str) -> None:
    adapter = get_benchmark('frames', TaskConfig(model='mock', datasets=['frames']))
    sample = Sample(input='Which city hosted the 1900 Summer Olympics?', target='Paris', metadata={'context': ''})
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content(model='mock', content=reply))

    assert adapter.extract_answer(reply, state) == 'Paris'
