import json
import pytest

from evalscope.benchmarks.humaneval import utils as humaneval_utils
from evalscope.benchmarks.live_code_bench import testing_util


def test_humaneval_time_limit_without_posix_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(humaneval_utils.signal, 'setitimer', raising=False)

    with humaneval_utils.time_limit(0.01):
        pass


def test_live_code_bench_without_posix_alarm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(testing_util.signal, 'SIGALRM', raising=False)
    monkeypatch.delattr(testing_util.signal, 'alarm', raising=False)
    monkeypatch.setattr(testing_util, 'reliability_guard', lambda: None)
    sample = {
        'input_output': json.dumps({
            'fn_name': 'add',
            'inputs': ['1\n2'],
            'outputs': ['3'],
        })
    }

    results, _ = testing_util.run_test(sample, test='def add(a, b): return a + b', debug=False, timeout=1)

    assert results == [True]


def test_live_code_bench_standard_input_without_posix_alarm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(testing_util.signal, 'SIGALRM', raising=False)
    monkeypatch.delattr(testing_util.signal, 'alarm', raising=False)
    monkeypatch.setattr(testing_util, 'reliability_guard', lambda: None)
    sample = {
        'input_output': json.dumps({
            'inputs': ['1 2'],
            'outputs': ['3'],
        })
    }

    results, _ = testing_util.run_test(
        sample,
        test='a, b = map(int, input().split()); print(a + b)',
        debug=False,
        timeout=1,
    )

    assert results == [True]
