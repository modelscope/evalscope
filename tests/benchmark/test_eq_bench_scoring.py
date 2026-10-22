"""EQ-Bench keeps a question's negative v2 score, as the official benchmark averages it."""
import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

RECORD = {
    'prompt': 'Rate the emotions of Claudia at the end of the dialogue.',
    'reference_answer': "{'emotion1': 'Remorseful', 'emotion2': 'Indifferent', 'emotion3': 'Affectionate', "
    "'emotion4': 'Annoyed', 'emotion1_score': 2, 'emotion2_score': 3, 'emotion3_score': 0, 'emotion4_score': 5}",
    'reference_answer_fullscale': "{'emotion1': 'Remorseful', 'emotion2': 'Indifferent', 'emotion3': 'Affectionate', "
    "'emotion4': 'Annoyed', 'emotion1_score': 0, 'emotion2_score': '6', 'emotion3_score': 0, 'emotion4_score': '7'}",
}


def score(completion: str) -> float:
    adapter = get_benchmark('eq_bench', TaskConfig(model='mock', datasets=['eq_bench']))
    sample = adapter.record_to_sample(RECORD)
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content(model='mock', content=completion))
    return adapter.match_score(completion, completion, sample.target, state).value['eq_bench_score']


def test_answer_far_from_reference_scores_below_zero() -> None:
    # Remorseful and Annoyed are each 7 off: 10 - (7 + 7) * 0.7477 = -0.4678 on the 0-10 scale.
    completion = 'Remorseful: 7\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 0'

    assert score(completion) == pytest.approx((10 - 14 * 0.7477) / 10)


def test_exact_answer_scores_one() -> None:
    assert score('Remorseful: 0\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7') == pytest.approx(1.0)
