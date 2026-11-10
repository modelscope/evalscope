import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from evalscope.benchmarks.lawbench.cherrant import score_corrections


@pytest.fixture
def toolkit(tmp_path: Path) -> Path:
    root = tmp_path / 'toolkit with spaces'
    root.mkdir()
    (root / 'parallel_to_m2.py').write_text('''
import argparse
from pathlib import Path
p = argparse.ArgumentParser()
for name in ['-f', '-o', '-g']: p.add_argument(name)
a = p.parse_args()
assert a.g == 'char'
assert Path.cwd() != Path(__file__).parent
Path(a.o).write_text(Path(a.f).read_text())
''')
    (root / 'compare_m2_for_evaluation.py').write_text('''
import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('-hyp'); p.add_argument('-ref')
a = p.parse_args()
assert Path(a.hyp).read_text() == '0 \\t 原句 \\t 正确句子 \\n'
assert Path(a.ref).read_text() == '0 \\t 原句 \\t 正确句子 \\n'
print('TP\\tFP\\tFN\\tPrec\\tRec\\tF0.5')
print('1\\t0\\t0\\t1.0\\t1.0\\t1.0')
''')
    return root


def _rows() -> list[dict[str, str]]:
    return [{'origin_prompt': '请纠错。\n句子：\n原句\n', 'prediction': '正确句子', 'refr': '正确句子'}]


def test_correction_subprocesses_are_isolated_and_concurrent(toolkit: Path) -> None:
    before = os.getcwd(), dict(os.environ)
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(lambda _: score_corrections(_rows(), toolkit, sys.executable, 30), range(8)))
    assert scores == [1.0] * 8
    assert (os.getcwd(), dict(os.environ)) == before
    assert sorted(p.name for p in toolkit.iterdir()) == ['compare_m2_for_evaluation.py', 'parallel_to_m2.py']


def test_correction_failure_is_not_a_zero_score(toolkit: Path) -> None:
    (toolkit / 'parallel_to_m2.py').write_text('raise RuntimeError("broken dependency")')
    with pytest.raises(RuntimeError, match='(?s)ChERRANT.*broken dependency'):
        score_corrections(_rows(), toolkit, sys.executable, 30)


@pytest.mark.parametrize('code', [
    'import evalscope_cherrant_missing_dependency',
    'raise ImportError("incompatible dependency")',
])
def test_missing_dependency_explains_installation_in_selected_interpreter(toolkit: Path, code: str) -> None:
    (toolkit / 'parallel_to_m2.py').write_text(code)
    with pytest.raises(RuntimeError, match='ChERRANT dependencies are missing or incompatible') as error:
        score_corrections(_rows(), toolkit, sys.executable, 30)

    message = str(error.value)
    assert sys.executable in message
    assert '-m pip install' in message
    assert 'evalscope[lawbench-cherrant]' in message
    assert '.[lawbench-cherrant]' in message
    assert 'extra_params.cherrant_python' in message
    assert 'Original error:' in message


def test_missing_interpreter_explains_python_configuration(toolkit: Path, tmp_path: Path) -> None:
    python = str(tmp_path / 'missing-python')
    with pytest.raises(RuntimeError, match='extra_params.cherrant_python') as error:
        score_corrections(_rows(), toolkit, python, 30)
    assert python in str(error.value)


def test_correction_timeout_is_reported(toolkit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='ChERRANT.*timed out'):
        score_corrections(_rows(), toolkit, sys.executable, 0.1)


def test_correction_rejects_missing_tools_and_invalid_output(toolkit: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='ChERRANT'):
        score_corrections(_rows(), tmp_path / 'missing', sys.executable, 30)
    (toolkit / 'compare_m2_for_evaluation.py').write_text('print("unexpected output")')
    with pytest.raises(ValueError, match='F0.5'):
        score_corrections(_rows(), toolkit, sys.executable, 30)
