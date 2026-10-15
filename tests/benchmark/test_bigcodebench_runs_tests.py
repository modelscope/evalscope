"""BigCodeBench's `test` field only defines `TestCases`; scoring must run it, as the official
harness does, so a solution that imports but fails its tests does not pass."""
import subprocess
import sys

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

CODE_PROMPT = 'def task_func(x):\n'
TEST = '''import unittest


class TestCases(unittest.TestCase):

    def test_double(self):
        self.assertEqual(task_func(2), 4)
'''


def _python_executor(code, timeout=60, language='python'):
    # The sandbox's python executor runs the file and reports success iff the exit code is 0.
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=timeout)
    return {'status': 'success' if proc.returncode == 0 else 'error', 'output': proc.stdout}


@pytest.mark.parametrize(('solution', 'passes'), [
    ('def task_func(x):\n    return 2 * x\n', True),
    ('def task_func(x):\n    return None\n', False),
])
def test_match_score_runs_test_cases(monkeypatch, solution, passes):
    adapter = get_benchmark('bigcodebench', TaskConfig(model='mock', datasets=['bigcodebench']))
    monkeypatch.setattr(type(adapter), 'use_sandbox', property(lambda self: True))
    monkeypatch.setattr(adapter, 'execute_code_in_sandbox', _python_executor)
    metadata = {'task_id': 'BigCodeBench/test', 'code_prompt': CODE_PROMPT, 'test': TEST}
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='', metadata=metadata),
        output=ModelOutput.from_content('mock', solution),
    )
    score = adapter.match_score(solution, solution, '', state)
    assert score.value['acc'] is passes
