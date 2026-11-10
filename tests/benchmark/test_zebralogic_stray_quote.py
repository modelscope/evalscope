"""ZebraLogicBench: a stray quote in the prose before the answer must not swallow it."""

import json

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

SOLUTION = {'House 1': {'Name': 'Arnold'}}


def _extract(completion: str) -> str:
    adapter = get_benchmark('zebralogicbench', TaskConfig(model='mock', datasets=['zebralogicbench']))
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='{}'),
        output=ModelOutput.from_content('mock', completion),
    )
    return adapter.extract_answer(completion, state)


def test_stray_quote_before_the_object_does_not_swallow_the_answer():
    completion = 'The box is 5" wide.\n' + json.dumps({'reasoning': 'R', 'solution': SOLUTION})

    extracted = _extract(completion)

    assert extracted
    assert json.loads(extracted)['solution'] == SOLUTION
