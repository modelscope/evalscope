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


def test_chartqa_falls_back_to_prediction_when_no_answer_marker():
    """Without an 'ANSWER:' marker, chartqa must fall back to the stripped prediction so a model that
    gives the bare answer still scores — mirroring docvqa/infovqa/mmmu. Previously it returned '' and
    scored 0 even when the prediction was exactly the answer."""
    adapter = get_benchmark('chartqa', TaskConfig(model='mock', datasets=['chartqa']))
    completion = '45'  # bare answer, no ANSWER: marker
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='45'),
        output=ModelOutput.from_content('mock', completion),
    )
    extracted = adapter.extract_answer(completion, state)
    assert extracted == '45'
    assert adapter.match_score(completion, extracted, '45', state).main_value == 1.0


def test_chartqa_fallback_still_zero_when_prediction_is_not_the_answer():
    """The fallback only helps when the bare prediction equals the answer; a verbose prediction with no
    marker still scores 0 (relaxed correctness requires an exact match for non-numeric answers)."""
    adapter = get_benchmark('chartqa', TaskConfig(model='mock', datasets=['chartqa']))
    completion = 'The tallest bar reads 45.'  # verbose, no marker
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='45'),
        output=ModelOutput.from_content('mock', completion),
    )
    extracted = adapter.extract_answer(completion, state)
    assert extracted == 'The tallest bar reads 45.'
    assert adapter.match_score(completion, extracted, '45', state).main_value == 0.0


def test_real_world_qa_falls_back_to_prediction_when_no_answer_marker():
    """real_world_qa must fall back to the stripped+lowered prediction without an 'ANSWER:' marker,
    matching chartqa/docvqa. Previously it returned '' and scored 0."""
    adapter = get_benchmark('real_world_qa', TaskConfig(model='mock', datasets=['real_world_qa']))
    completion = 'Paris'  # bare answer, no marker
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='paris'),
        output=ModelOutput.from_content('mock', completion),
    )
    extracted = adapter.extract_answer(completion, state)
    assert extracted == 'paris'


def test_real_world_qa_still_parses_marker_when_present():
    """An 'ANSWER:' marker still wins over the fallback for real_world_qa."""
    adapter = get_benchmark('real_world_qa', TaskConfig(model='mock', datasets=['real_world_qa']))
    completion = 'Some reasoning.\\nANSWER: Paris'
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='paris'),
        output=ModelOutput.from_content('mock', completion),
    )
    extracted = adapter.extract_answer(completion, state)
    assert extracted == 'paris'
