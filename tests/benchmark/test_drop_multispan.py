import pytest

from evalscope import TaskConfig
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark


@pytest.mark.parametrize(('prediction', 'gold', 'em', 'f1'), [
    ('["Brady", "Palmer"]', [('Brady', 'Palmer')], 1, 1),
    ('["Palmer", "Brady"]', [('Brady', 'Palmer')], 1, 1),
    ('["Brady", "Other"]', [('Brady', 'Palmer')], 0, 0.5),
    ('["Brady", "Brady"]', [('Brady', 'Palmer')], 0, 0.5),
    ('["Paris, Texas", "Rome"]', [('Paris, Texas', 'Rome')], 1, 1),
    ('["β", "雪"]', [('another',), ('雪', 'β')], 1, 1),
    ('Paris, Texas', [('Paris, Texas',)], 1, 1),
    ('23', [('23',)], 1, 1),
    ('January 2 2020', [('January 2 2020',)], 1, 1),
    ('Brady, Palmer', [('Brady', 'Palmer')], 0, 0.33),
    ('["Brady", "Palmer"', [('Brady', 'Palmer')], 0, 0.33),
    ('["Brady", 7]', [('Brady', '7')], 0, 0.33),
    ('[' + '9' * 5000 + ']', [('Brady', 'Palmer')], 0, 0),
    ('[]', [('Brady', 'Palmer')], 0, 0),
])
def test_drop_scores_final_answer(prediction: str, gold: list[tuple], em: float, f1: float) -> None:
    adapter = get_benchmark('drop', TaskConfig(datasets=['drop']))
    completion = f'Answer: earlier guess\nAnswer: {prediction}'
    state = TaskState(model='mock', sample=Sample(input='question', target=str(gold)), completed=True,
                      output=ModelOutput.from_content('mock', completion))
    score = adapter.calculate_metrics(state).score
    assert score.value == {'em': em, 'f1': f1}
    assert score.extracted_prediction == prediction
