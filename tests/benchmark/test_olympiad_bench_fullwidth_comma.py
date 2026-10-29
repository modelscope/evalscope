"""Chinese OlympiadBench references separate answers with a full-width comma; the official judge
reads it as ','. An answer in the format the prompt asks for (ASCII commas) must match."""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize(
    ('final_answer', 'is_multiple_answer', 'answer_type', 'boxed'),
    [
        ('$4$，$30$', True, 'Numerical', '4,30'),
        ('$0.1275，0.0475，0.09$', True, 'Numerical', '0.1275,0.0475,0.09'),
        ('$(1，+\\infty )$', False, 'Interval', '(1,+\\infty)'),
    ],
)
def test_fullwidth_comma_reference(final_answer, is_multiple_answer, answer_type, boxed):
    adapter = get_benchmark('olympiad_bench', TaskConfig(model='mock', datasets=['olympiad_bench']))
    metadata = {
        'final_answer': [final_answer],
        'is_multiple_answer': is_multiple_answer,
        'unit': '',
        'answer_type': answer_type,
        'error': None,
        'language': 'Chinese',
    }
    completion = f'所以最终答案是\\boxed{{{boxed}}}。'
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target=final_answer, metadata=metadata, id=0),
        output=ModelOutput.from_content('mock', completion),
        completed=True,
    )
    score = adapter.calculate_metrics(state).score
    assert score.value.get('acc') == 1.0
