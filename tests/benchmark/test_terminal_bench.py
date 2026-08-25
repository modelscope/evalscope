import json
import math
from types import SimpleNamespace
from typing import Any

import pytest

from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.metric import Score
from evalscope.benchmarks.terminal_bench.terminal_bench_adapter import _phase_timeout_options, _TerminalBenchBase

TRIAL_URI = 'file:///tmp/terminal-bench-trial'


def _score(result: dict[str, Any]) -> Score:
    adapter = object.__new__(_TerminalBenchBase)
    task_state = SimpleNamespace(metadata={'result': result})
    return adapter.match_score('raw', 'filtered', 'target', task_state)


@pytest.mark.parametrize('reward', [0, 1])
def test_terminal_bench_scores_valid_binary_reward(reward: int) -> None:
    result = {
        'trial_uri': TRIAL_URI,
        'verifier_result': {
            'rewards': {
                'reward': reward
            }
        },
    }

    score = _score(result)

    assert score.value == {'acc': float(reward)}
    assert score.metadata == result


@pytest.mark.parametrize(
    ('verifier_result', 'expected_context'),
    [
        (None, 'verifier_result'),
        ({}, 'rewards'),
        ({'rewards': None}, 'rewards'),
        ({'rewards': {}}, 'reward'),
    ],
)
def test_terminal_bench_rejects_missing_reward(verifier_result: Any, expected_context: str) -> None:
    result = {
        'trial_uri': TRIAL_URI,
        'verifier_result': verifier_result,
    }

    with pytest.raises(RuntimeError) as exc_info:
        _score(result)

    assert TRIAL_URI in str(exc_info.value)
    assert expected_context in str(exc_info.value)


@pytest.mark.parametrize(
    'reward',
    [
        pytest.param(None, id='null'),
        pytest.param('1', id='string'),
        pytest.param(True, id='bool'),
        pytest.param(math.nan, id='nan'),
        pytest.param(math.inf, id='infinity'),
        pytest.param(-0.1, id='below-range'),
        pytest.param(1.1, id='above-range'),
    ],
)
def test_terminal_bench_rejects_invalid_reward(reward: Any) -> None:
    result = {
        'trial_uri': TRIAL_URI,
        'verifier_result': {
            'rewards': {
                'reward': reward
            }
        },
    }

    with pytest.raises(RuntimeError) as exc_info:
        _score(result)

    assert TRIAL_URI in str(exc_info.value)
    assert 'invalid reward' in str(exc_info.value)


def test_terminal_bench_rejects_trial_exception_even_with_reward() -> None:
    result = {
        'trial_uri': TRIAL_URI,
        'exception_info': {
            'exception_type': 'RewardFileNotFoundError',
            'message': 'reward.json is missing',
        },
        'verifier_result': {
            'rewards': {
                'reward': 0
            }
        },
    }

    with pytest.raises(RuntimeError) as exc_info:
        _score(result)

    error = str(exc_info.value)
    assert TRIAL_URI in error
    assert 'RewardFileNotFoundError' in error
    assert 'reward.json is missing' in error


def test_terminal_bench_absolute_timeout_disables_global_multiplier_for_that_phase() -> None:
    assert _phase_timeout_options(10_800, None, 'agent') == (10_800.0, 1.0)


def test_terminal_bench_rejects_conflicting_phase_timeout_options() -> None:
    with pytest.raises(ValueError, match='agent_timeout_sec'):
        _phase_timeout_options(10_800, 2.0, 'agent')


def test_terminal_bench_trace_preserves_request_perf_metrics(tmp_path) -> None:
    trial_dir = tmp_path / 'trial'
    trajectory_dir = trial_dir / 'agent'
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / 'trajectory.json').write_text(
        json.dumps({
            'agent': {
                'name': 'terminus-2',
                'model_name': 'test-model',
            },
            'steps': [{
                'source': 'agent',
                'message': 'answer',
                'step_id': 0,
            }],
        })
    )
    adapter = object.__new__(_TerminalBenchBase)
    adapter.environment_type = 'docker'

    _, messages = adapter._load_harbor_trace(
        {'trial_uri': f'file://{trial_dir}'},
        [PerformanceMetrics(latency=1.0, input_tokens=3, output_tokens=2)],
    )

    assert messages[0].perf_metrics.latency == 1.0
    assert messages[0].perf_metrics.input_tokens == 3
