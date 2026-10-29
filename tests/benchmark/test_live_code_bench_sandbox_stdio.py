"""With use_sandbox, stdio output is compared the way the local grader (testing_util.grade_stdio)
compares it: each line stripped, then numbers compared as Decimals."""
import pytest

from evalscope.benchmarks.live_code_bench.sandbox_evaluate_utils import _evaluate_stdio_in_sandbox


class _FakeSandbox:

    def __init__(self, output: str):
        self.output = output

    def execute_code_in_sandbox(self, code, timeout, language):
        return {'status': 'success', 'output': self.output}


@pytest.mark.parametrize(
    ('printed', 'expected', 'passes'),
    [
        ('1 2 3\n', '1 2 3\n', True),
        ('1 2 3 \n', '1 2 3\n', True),
        ('2.0\n', '2.0000000000\n', True),
        ('1 2 \n3 4\n', '1 2\n3 4\n', True),
        ('1 2 4\n', '1 2 3\n', False),
        ('1 2 3\n4\n', '1 2 3\n', False),
        ('yes\n', 'Yes\n', False),
    ],
)
def test_stdio_output_comparison(printed, expected, passes):
    passed, _ = _evaluate_stdio_in_sandbox(
        _FakeSandbox(printed), code='', inputs=[''], outputs=[expected], timeout=5, debug=False
    )
    assert passed is passes
