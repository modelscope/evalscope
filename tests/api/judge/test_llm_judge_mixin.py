"""Lazy ``llm_judges`` initialization must be thread-safe."""

import threading
import time
from typing import List, cast

import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.mixin import LLMJudgeMixin
from evalscope.metrics import LLMJudge


class _CountingJudgeMixin(LLMJudgeMixin):
    """Minimal judge mixin with deliberately slow initialization."""

    use_llm_judge = True

    def __init__(self) -> None:
        super().__init__(benchmark_meta=BenchmarkMeta(name='stub', dataset_id='stub'), task_config=None)
        self.init_calls = 0
        self._calls_lock = threading.Lock()
        self._judge = cast(LLMJudge, object())

    def init_llm_judges(self) -> List[LLMJudge]:
        with self._calls_lock:
            self.init_calls += 1
        time.sleep(0.05)
        return [self._judge]


class _FailOnceJudgeMixin(_CountingJudgeMixin):
    """Fails initialization once so callers can verify retry behavior."""

    def init_llm_judges(self) -> List[LLMJudge]:
        with self._calls_lock:
            self.init_calls += 1
            should_fail = self.init_calls == 1
        if should_fail:
            raise RuntimeError('judge initialization failed')
        return [self._judge]


class _DisabledJudgeMixin(_CountingJudgeMixin):
    """Disables judge construction without requiring task-config plumbing."""

    use_llm_judge = False


def test_llm_judges_initializes_exactly_once_under_concurrency() -> None:
    """All concurrent first accesses must receive one initialized judge list."""
    mixin = _CountingJudgeMixin()
    num_threads = 8
    barrier = threading.Barrier(num_threads)
    results: List[List[LLMJudge]] = []
    errors: List[BaseException] = []

    def access() -> None:
        try:
            barrier.wait()
            results.append(mixin.llm_judges)
        except BaseException as error:  # noqa: BLE001 - surfaced after every worker exits
            errors.append(error)

    threads = [threading.Thread(target=access) for _ in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert mixin.init_calls == 1
    assert len(results) == num_threads
    assert all(result is results[0] for result in results)
    assert all(result[0] is results[0][0] for result in results)


def test_llm_judges_retries_after_initialization_failure() -> None:
    """A failed initialization must not publish a partial result."""
    mixin = _FailOnceJudgeMixin()

    with pytest.raises(RuntimeError, match='judge initialization failed'):
        _ = mixin.llm_judges

    assert mixin._llm_judges is None
    assert mixin.llm_judges == [mixin._judge]
    assert mixin.init_calls == 2


def test_llm_judges_returns_empty_when_judge_is_disabled() -> None:
    """Rule-only scoring must not initialize any judge."""
    mixin = _DisabledJudgeMixin()

    assert mixin.llm_judges == []
    assert mixin.init_calls == 0
