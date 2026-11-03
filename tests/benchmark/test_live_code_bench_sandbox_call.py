"""Call-based sandbox grading must preserve candidate source when building its wrapper."""
import json
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import pytest

from evalscope.benchmarks.live_code_bench.sandbox_evaluate_utils import _evaluate_call_based_in_sandbox


class _ExecutingSandbox:

    def __init__(self) -> None:
        self.calls: List[Tuple[str, int, str]] = []

    def execute_code_in_sandbox(self, code: str, timeout: int, language: str) -> Dict[str, Any]:
        """Execute only the fixed, trusted snippets supplied by these tests."""
        self.calls.append((code, timeout, language))
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=timeout)
        return {
            'status': 'success' if result.returncode == 0 else 'error',
            'output': result.stdout,
            'error': result.stderr,
        }


@pytest.mark.parametrize(
    ('code', 'value', 'expected'),
    [
        pytest.param(
            "class Solution:\n    def solve(self, value: str) -> str:\n        return 'ok'",
            'input',
            'ok',
            id='solution-ends-in-single-quoted-literal',
        ),
        pytest.param(
            "class Solution:\n    def solve(self, value: str) -> str:\n"
            "        '''Return the supplied value.'''\n        return value",
            'input',
            'input',
            id='embedded-triple-single-quoted-docstring',
        ),
        pytest.param(
            "def solve(value: str) -> str:\n    return 'ok'",
            'input',
            'ok',
            id='standalone-function-ends-in-single-quoted-literal',
        ),
        pytest.param(
            r'''class Solution:
    def solve(self, value: str) -> str:
        return value + r"\path" + "\n" + "'single' \"double\" 雪"''',
            'prefix',
            'prefix\\path\n\'single\' "double" 雪',
            id='backslashes-newline-quotes-and-unicode',
        ),
        pytest.param(
            'class Solution:\n    def solve(self, value: str) -> str:\n        return "ok"',
            'input',
            'ok',
            id='solution-ends-in-double-quoted-literal',
        ),
    ],
)
def test_call_based_source_quoting(code: str, value: str, expected: str) -> None:
    sandbox = _ExecutingSandbox()

    passed, details = _evaluate_call_based_in_sandbox(
        sandbox,
        code=code,
        inputs=[json.dumps(value)],
        outputs=[json.dumps(expected)],
        fn_name='solve',
        timeout=5,
        debug=False,
    )

    assert passed is True, details
    assert details == {'total_tests': 1, 'passed_tests': 1, 'failed_cases': []}
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0][1:] == (5, 'python')


def test_call_based_wrong_answer_fails() -> None:
    sandbox = _ExecutingSandbox()
    code = 'class Solution:\n    def solve(self, value: int) -> int:\n        return value + 1'

    passed, details = _evaluate_call_based_in_sandbox(
        sandbox,
        code=code,
        inputs=['1'],
        outputs=['1'],
        fn_name='solve',
        timeout=5,
        debug=False,
    )

    assert passed is False
    assert details['passed_tests'] == 0
    assert details['failed_cases'] == ['Test 0: expected 1, got 2']
    assert len(sandbox.calls) == 1


def test_call_based_candidate_runtime_error_fails() -> None:
    sandbox = _ExecutingSandbox()
    code = "def solve(value: int) -> int:\n    raise RuntimeError('candidate failed')"

    passed, details = _evaluate_call_based_in_sandbox(
        sandbox,
        code=code,
        inputs=['1'],
        outputs=['1'],
        fn_name='solve',
        timeout=5,
        debug=False,
    )

    assert passed is False
    assert details['passed_tests'] == 0
    assert details['failed_cases'] == ['Test 0: EXECUTION_ERROR: candidate failed']
    assert len(sandbox.calls) == 1
