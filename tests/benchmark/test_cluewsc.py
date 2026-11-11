"""CLUEWSC maps FewCLUE true/false labels to A/B and reads the Chinese answer line."""
import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

RECORD = {
    'target': {'span2_index': 40, 'span1_index': 6, 'span1_text': '舌头', 'span2_text': '它'},
    'idx': 0,
    'text': '刀光一闪，那舌头象一只受惊的老鼠从受刑人的嘴巴和行刑人的手之间跳出来，看那样子，它是想往天上去的。',
}


def _adapter():
    return get_benchmark('cluewsc', TaskConfig(model='mock', datasets=['cluewsc']))


@pytest.mark.parametrize(('label', 'target'), [('true', 'A'), ('false', 'B')])
def test_record_to_sample(label, target):
    sample = _adapter().record_to_sample({**RECORD, 'label': label})
    assert sample.target == target
    assert sample.choices == ['是', '否']
    assert sample.input.endswith('此处，“它”是否指代“舌头”？')


def test_extracts_answer_line():
    adapter = _adapter()
    sample = adapter.record_to_sample({**RECORD, 'label': 'false'})
    completion = '答案：B'
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content(model='mock', content=completion))
    assert adapter.extract_answer(completion, state) == 'B'
