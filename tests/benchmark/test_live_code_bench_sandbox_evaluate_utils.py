import contextlib
import io
import json

from evalscope.benchmarks.live_code_bench.sandbox_evaluate_utils import evaluate_in_sandbox
from evalscope.benchmarks.live_code_bench.testing_util import run_test


class LocalSandboxAdapter:

    def execute_code_in_sandbox(self, code: str, timeout: int, language: str):
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(code, {})
        except Exception as e:
            return {'status': 'error', 'output': output.getvalue(), 'error': repr(e)}
        return {'status': 'success', 'output': output.getvalue()}


def test_stdio_sandbox_supports_stdin_buffer_read():
    code = """
import sys

data = sys.stdin.buffer.read().split()
print(int(data[0]) + int(data[1]))
"""
    evaluation_sample = '{"inputs": ["4 2\\n"], "outputs": ["6\\n"], "fn_name": null}'

    passed, details = evaluate_in_sandbox(LocalSandboxAdapter(), code, evaluation_sample)

    assert passed is True
    assert details['passed_tests'] == 1


def test_stdio_sandbox_keeps_stringio_text_read_behavior():
    code = """
import sys

print(sys.stdin.read().strip().upper())
"""
    evaluation_sample = '{"inputs": ["ok\\n"], "outputs": ["OK\\n"], "fn_name": null}'

    passed, details = evaluate_in_sandbox(LocalSandboxAdapter(), code, evaluation_sample)

    assert passed is True
    assert details['passed_tests'] == 1


def test_stdio_local_evaluation_supports_stdin_buffer_read():
    code = """
import sys

data = sys.stdin.buffer.read().split()
print(int(data[0]) + int(data[1]))
"""
    sample = {'input_output': json.dumps({'inputs': ['4 2\n'], 'outputs': ['6\n'], 'fn_name': None})}

    results, details = run_test(sample, test=code, debug=False, timeout=6)

    assert results == [True]
    assert 'execution time' in details
