"""With use_sandbox, call-based inputs with several arguments are parsed as JSON, so true/false/null
reach the solution as bool/None, as in the local grader."""
import json
import subprocess
import sys
import tempfile

from evalscope.benchmarks.live_code_bench.sandbox_evaluate_utils import evaluate_in_sandbox


class _LocalSandbox:

    def execute_code_in_sandbox(self, code, timeout, language):
        with tempfile.NamedTemporaryFile('w', suffix='.py') as f:
            f.write(code)
            f.flush()
            proc = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=30)
        return {'status': 'success', 'output': proc.stdout}


def test_multi_argument_json_literals():
    code = 'def pick(nums, largest):\n    return max(nums) if largest else min(nums)\n'
    sample = json.dumps({'fn_name': 'pick', 'inputs': ['[3,1,2]\ntrue', '[3,1,2]\nfalse'], 'outputs': ['3', '1']})
    passed, _ = evaluate_in_sandbox(_LocalSandbox(), code, sample)
    assert passed
