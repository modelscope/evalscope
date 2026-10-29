"""ZebraLogicBench: a puzzle whose answer cannot be parsed still counts toward the cell totals (and
no_answer), and an answer whose JSON has raw newlines in a string value is still read, as in the
official ZeroEval scorer."""
import json

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

REFERENCE = json.dumps({'header': ['House', 'Name', 'Color'], 'rows': [['1', 'Alice', 'red'], ['2', 'Bob', 'blue']]})
SOLVED = (
    '{"reasoning": "Alice is in house 1.\nSo Bob is in house 2.", "solution": '
    '{"House 1": {"Name": "Alice", "Color": "red"}, "House 2": {"Name": "Bob", "Color": "blue"}}}'
)
TRUNCATED = '{"reasoning": "Alice is in house 1, so'


def _score(adapter, completion):
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target=REFERENCE),
        output=ModelOutput.from_content('mock', completion),
    )
    extracted = adapter.extract_answer(completion, state)
    return extracted, SampleScore(score=adapter.match_score(completion, extracted, REFERENCE, state), sample_id=0)


def test_unparsed_answer_counts_and_multiline_json_is_read():
    adapter = get_benchmark('zebralogicbench', TaskConfig(model='mock', datasets=['zebralogicbench']))
    solved_text, solved = _score(adapter, SOLVED)
    truncated_text, truncated = _score(adapter, TRUNCATED)

    assert solved_text  # the raw newline inside "reasoning" no longer makes the JSON unreadable
    assert truncated_text == ''
    agg = {a.metric_name: a.score for a in adapter.aggregate_scores([solved, truncated])}
    assert agg['puzzle_acc'] == 0.5
    assert agg['cell_acc'] == 0.5  # 4 of 8 cells: the unparsed puzzle's cells count as wrong
