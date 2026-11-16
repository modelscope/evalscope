"""Opt-in checks against an image built from the bundled SciCode Dockerfile.

Run with SCICODE_TEST_IMAGE=<built-image> python -m pytest tests/benchmark/test_scicode_sandbox.py.
The container creates its own tiny HDF5 fixture; no benchmark download or model is needed.
"""

import os
import subprocess
from pathlib import Path

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageAssistant
from evalscope.benchmarks.scicode.scicode_adapter import SciCodeAdapter


@pytest.mark.skipif(not os.environ.get('SCICODE_TEST_IMAGE'), reason='Set SCICODE_TEST_IMAGE to a built SciCode image')
@pytest.mark.parametrize('answer,expected_exit', [(42, 0), (41, 1)])
def test_scicode_sandbox_executes_matplotlib_dependencies(answer: int, expected_exit: int) -> None:
    subproblem = {'step_number': '80.1', 'test_cases': ['assert answer() == target']}
    sample = Sample(input='synthetic', metadata={
        'required_dependencies': 'from mpl_toolkits.mplot3d import Axes3D',
        'sub_steps': [subproblem],
    })
    state = TaskState(
        model='local', sample=sample,
        messages=[ChatMessageAssistant(content=f'```python\ndef answer():\n    return {answer}\n```')],
    )
    adapter = object.__new__(SciCodeAdapter)
    code = adapter._compose_code(subproblem, state)
    bootstrap = (
        "import h5py\n"
        "with h5py.File('/workspace/test_data.h5', 'w') as fixture:\n"
        "    fixture.create_dataset('80.1/test1/value', data=42)\n"
        f'exec({code!r})\n'
    )
    docker_dir = Path(__file__).resolve().parents[2] / 'evalscope' / 'benchmarks' / 'scicode' / 'docker'
    command = ['docker', 'run', '--rm', '--network', 'none']
    for helper in ('process_data.py', 'test_util.py'):
        command.extend(['-v', f'{docker_dir / helper}:/workspace/{helper}:ro'])
    command.extend(['-i', os.environ['SCICODE_TEST_IMAGE'], 'python', '-'])

    result = subprocess.run(command, input=bootstrap, text=True, capture_output=True, timeout=120)

    assert result.returncode == expected_exit, result.stderr
    if expected_exit:
        assert 'AssertionError' in result.stderr
