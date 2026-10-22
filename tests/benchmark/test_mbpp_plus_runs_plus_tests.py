"""MBPP+ scores a solution with its `test` suite, as EvalPlus does, and with the imports in `test_imports`
that the base asserts use."""
import subprocess
import sys

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

# Shaped like evalplus/mbppplus task 3: the base asserts never call is_not_prime(1).
IS_NOT_PRIME = {
    'task_id': 3,
    'test_imports': [],
    'test_list': ['assert is_not_prime(10) == True', 'assert is_not_prime(37) == False'],
    'test': 'inputs = [[10], [37], [1]]\nresults = [True, False, True]\n'
    'for inp, exp in zip(inputs, results):\n    assert is_not_prime(*inp) == exp\n',
}
# Shaped like task 98: the base assert calls math.isclose, imported only by `test_imports`.
MULTIPLY_NUM = {
    'task_id': 98,
    'test_imports': ['import math'],
    'test_list': ['assert math.isclose(multiply_num((8, 2, 3, -1, 7)), -67.2, rel_tol=0.001)'],
    'test': 'assert multiply_num((-10, -20, -30)) == -2000.0\n',
}


def _python_executor(code: str, timeout: int = 60, language: str = 'python') -> dict:
    # The sandbox's python executor runs the file and reports success iff the exit code is 0.
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=timeout)
    return {'status': 'success' if proc.returncode == 0 else 'error', 'output': proc.stdout}


def score(monkeypatch: pytest.MonkeyPatch, metadata: dict, solution: str) -> bool:
    adapter = get_benchmark('mbpp_plus', TaskConfig(model='mock', datasets=['mbpp_plus']))
    monkeypatch.setattr(type(adapter), 'use_sandbox', property(lambda self: True))
    monkeypatch.setattr(adapter, 'execute_code_in_sandbox', _python_executor)
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='', metadata=metadata),
        output=ModelOutput.from_content('mock', solution),
    )
    return adapter.match_score(solution, solution, '', state).value['acc']


def test_solution_failing_the_plus_suite_does_not_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    solution = 'def is_not_prime(n):\n    return any(n % i == 0 for i in range(2, n))\n'

    assert score(monkeypatch, IS_NOT_PRIME, solution) is False


@pytest.mark.parametrize('future_import', ['', 'from __future__ import annotations\n'])
def test_correct_solution_passes_with_test_imports(monkeypatch: pytest.MonkeyPatch, future_import: str) -> None:
    solution = future_import + 'def multiply_num(numbers):\n    total = 1\n    for x in numbers:\n        total *= x\n    return total / len(numbers)\n'

    assert score(monkeypatch, MULTIPLY_NUM, solution) is True
