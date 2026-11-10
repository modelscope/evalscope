"""ZebraLogicBench: braces inside a JSON string value are data, not structure.

The extractor scans for the last balanced ``{...}`` run before handing it to ``json.loads``. A brace
that appears inside a string value used to be counted as an opening or closing object, which either
emptied the extraction or truncated it to an inner object, so a valid answer was scored as wrong.
"""
import json

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

SOLUTION = {'House 1': {'Name': 'Alice', 'Color': 'red'}}
REFERENCE = json.dumps({'header': ['House', 'Name', 'Color'], 'rows': [['1', 'Alice', 'red']]})


def _extract(completion: str) -> str:
    adapter = get_benchmark('zebralogicbench', TaskConfig(model='mock', datasets=['zebralogicbench']))
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='{}'),
        output=ModelOutput.from_content('mock', completion),
    )
    return adapter.extract_answer(completion, state)


def test_opening_brace_inside_string_value_does_not_reset_the_object():
    completion = json.dumps({'reasoning': 'the set { is open', 'solution': SOLUTION})

    extracted = _extract(completion)

    assert extracted
    assert json.loads(extracted)['solution'] == SOLUTION


def test_closing_brace_inside_string_value_does_not_truncate():
    completion = json.dumps({'reasoning': 'I used } as a marker', 'solution': SOLUTION})

    extracted = _extract(completion)

    assert extracted
    assert json.loads(extracted)['solution'] == SOLUTION


def test_escaped_quote_before_a_brace_does_not_leak_string_state():
    completion = '{"reasoning": "he said \\"done {\\" and stopped", "solution": {"House 1": {"Name": "Alice", "Color": "red"}}}'

    extracted = _extract(completion)

    assert extracted
    assert json.loads(extracted)['solution'] == SOLUTION


def test_opening_brace_inside_string_value_scores_as_solved():
    completion = json.dumps({'reasoning': 'the set { is open', 'solution': SOLUTION})
    adapter = get_benchmark('zebralogicbench', TaskConfig(model='mock', datasets=['zebralogicbench']))
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target=REFERENCE),
        output=ModelOutput.from_content('mock', completion),
    )

    extracted = adapter.extract_answer(completion, state)
    score = adapter.match_score(completion, extracted, REFERENCE, state)

    assert score.value['Solved Puzzle'] == 1
    assert score.value['Solved Cell'] == 2
    assert score.value['No answer'] == 0


def test_evaluation_version_tracks_string_aware_extraction():
    adapter = get_benchmark('zebralogicbench', TaskConfig(model='mock', datasets=['zebralogicbench']))

    assert adapter.benchmark_meta.evaluation_version == 'v1.2'
