# Answer extraction and test-program execution for DS-1000 (https://github.com/xlang-ai/DS-1000).
#
# Independent implementation of the evaluation protocol described in the DS-1000 repository: the
# test program is the problem's `code_context`, followed by `test_execution(code)` and, when the context
# defines it, `test_string(code)`; a problem passes when this program exits without error.

import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_FENCE_RE = re.compile(r'```[ \t]*(?:python|py)?[ \t]*\n(.*?)```', re.IGNORECASE | re.DOTALL)
_STOP_MARKERS = ('</code>', '\nEND SOLUTION', '# SOLUTION END')

# Same execution environment hints as the official harness, plus a non-interactive matplotlib backend.
_EXEC_ENV = {'TF_CPP_MIN_LOG_LEVEL': '3', 'CUDA_VISIBLE_DEVICES': '-1', 'MPLBACKEND': 'Agg'}


def extract_code(completion: str) -> str:
    """Return the solution code of a completion.

    The DS-1000 prompt ends with ``BEGIN SOLUTION\\n<code>`` and the official harness keeps the text up to
    ``</code>`` (or ``END SOLUTION`` / ``# SOLUTION END``), dropping ``<code>`` tags and markdown fences.
    Chat models often write prose around the code, so the code after an opening ``<code>`` tag, or else
    the first fenced block, is used when present.
    """
    text = _THINK_RE.sub('', completion or '')
    if '<code>' in text:
        text = text.split('<code>', 1)[1]
    for marker in _STOP_MARKERS:
        text = text.split(marker, 1)[0]
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    elif '```' in text:  # an unterminated fence
        text = re.sub(r'^\s*```[ \t]*(?:python|py)?[ \t]*\n?', '', text, flags=re.IGNORECASE).split('```', 1)[0]
    return text.replace('<code>', '')


def build_test_program(code_context: str, code: str) -> str:
    """The official test program: context, the solution as a string, and the test calls."""
    program = f'{code_context}\ncode = {code!r}\ntest_execution(code)\n'
    if 'test_string(' in code_context:
        program += 'test_string(code)\n'
    return program


def run_locally(program: str, timeout: float) -> Dict[str, Any]:
    """Run the test program in a fresh Python process inside a temporary working directory.

    Tests write files such as ``output.png`` in the working directory, so every program gets its own.
    This is NOT a security sandbox: the program runs with the permissions of the evaluating user.
    """
    with tempfile.TemporaryDirectory(prefix='ds1000_') as workdir:
        with open(os.path.join(workdir, 'test_program.py'), 'w', encoding='utf-8') as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, 'test_program.py'],
                cwd=workdir,
                env={**os.environ, **_EXEC_ENV},
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {'passed': False, 'status': 'timeout', 'stderr': ''}
    passed = proc.returncode == 0
    return {'passed': passed, 'status': 'passed' if passed else 'failed', 'stderr': proc.stderr[-2000:]}
