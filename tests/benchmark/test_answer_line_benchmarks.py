"""Open-answer benchmarks read the 'ANSWER:' line the way the multiple-choice parser does, so a bold or
lower-case answer line still scores."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize('answer_line', ['ANSWER: 45', '**ANSWER: 45**', 'ANSWER: **45**', 'Answer: 45'])
def test_chartqa_answer_line(answer_line):
    adapter = get_benchmark('chartqa', TaskConfig(model='mock', datasets=['chartqa']))
    completion = f'The tallest bar reads 45.\\n{answer_line}'
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='45'),
        output=ModelOutput.from_content('mock', completion),
    )
    extracted = adapter.extract_answer(completion, state)
    assert adapter.match_score(completion, extracted, '45', state).main_value == 1.0
