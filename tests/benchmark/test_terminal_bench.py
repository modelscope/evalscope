import json
import math
import sys
from fnmatch import fnmatch
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.metric import Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.terminal_bench import terminal_bench_adapter
from evalscope.benchmarks.terminal_bench.terminal_bench_adapter import _phase_timeout_options, _TerminalBenchBase
from evalscope.config import TaskConfig

TRIAL_URI = 'file:///tmp/terminal-bench-trial'


@pytest.fixture
def harbor_dataset(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls = []

    class TaskConfigDouble(BaseModel):
        name: str
        ref: str
        source: str

    class DatasetConfigDouble:

        def __init__(self, **kwargs: Any) -> None:
            self.options = kwargs
            calls.append(kwargs)

        async def get_task_configs(self) -> list[TaskConfigDouble]:
            revision = self.options.get('ref') or 'latest'
            tasks = [
                TaskConfigDouble(
                    name=f'terminal-bench/{name}',
                    ref=f'sha256:{revision}-{name}',
                    source=self.options['name'],
                ) for name in ('cpu-task', 'sidecar-task')
            ]
            patterns = self.options.get('task_names')
            if patterns:
                tasks = [task for task in tasks if any(fnmatch(task.name, pattern) for pattern in patterns)]
                if not tasks:
                    raise ValueError('No tasks matched the filter(s)')
            return tasks

    module = ModuleType('harbor.models.job.config')
    module.DatasetConfig = DatasetConfigDouble
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(terminal_bench_adapter, 'check_import', lambda *args, **kwargs: True)
    monkeypatch.setattr(terminal_bench_adapter, 'version', lambda package: '0.14.0')
    monkeypatch.setattr(terminal_bench_adapter, '_validate_environment_requirements', lambda environment: None)
    return calls


@pytest.mark.parametrize(
    ('name', 'source', 'revision', 'max_turns'),
    [
        ('terminal_bench_v4', 'terminal-bench/terminal-bench', '4.0.0', None),
        ('terminal_bench_v2', 'terminal-bench/terminal-bench-2', None, 200),
        ('terminal_bench_v2_1', 'terminal-bench/terminal-bench-2-1', None, 200),
    ],
)
def test_terminal_bench_load_preserves_revision_and_task_digests(
    harbor_dataset: list[dict[str, Any]], name: str, source: str, revision: str | None, max_turns: int | None
) -> None:
    adapter = get_benchmark(name, config=TaskConfig(datasets=[name], model='mock', eval_type='mock_llm'))

    dataset = adapter.load_dataset()['test']

    assert harbor_dataset[0]['name'] == source
    assert harbor_dataset[0]['ref'] == revision
    assert len(dataset) == 2
    assert dataset[0].metadata['ref'] == f'sha256:{revision or "latest"}-cpu-task'
    assert dataset[0].metadata['source'] == source
    assert adapter.max_turns == max_turns


def test_terminal_bench_v4_task_filter_precedes_limit_and_repeats(
    harbor_dataset: list[dict[str, Any]],
) -> None:
    config = TaskConfig(
        datasets=['terminal_bench_v4'],
        model='mock',
        eval_type='mock_llm',
        dataset_args={'terminal_bench_v4': {'extra_params': {'task_names': ['terminal-bench/sidecar-*']}}},
        limit=1,
        repeats=2,
    )
    adapter = get_benchmark('terminal_bench_v4', config=config)

    dataset = adapter.load_dataset()['test']

    assert harbor_dataset[0]['task_names'] == ['terminal-bench/sidecar-*']
    assert len(dataset) == 2
    assert {sample.metadata['name'] for sample in dataset} == {'terminal-bench/sidecar-task'}
    assert [sample.id for sample in dataset] == [0, 1]
    assert dataset[0].group_id == dataset[1].group_id
    dataset[0].metadata['result'] = {'reward': 1}
    assert 'result' not in dataset[1].metadata


def test_terminal_bench_v4_unmatched_task_filter_fails(harbor_dataset: list[dict[str, Any]]) -> None:
    config = TaskConfig(
        datasets=['terminal_bench_v4'],
        model='mock',
        eval_type='mock_llm',
        dataset_args={'terminal_bench_v4': {'extra_params': {'task_names': ['terminal-bench/misspelled']}}},
    )
    adapter = get_benchmark('terminal_bench_v4', config=config)

    with pytest.raises(ValueError, match='No tasks matched'):
        adapter.load_dataset()


@pytest.mark.parametrize('harbor_version', ['0.13.2', '1.0.0', '1.1.0'])
def test_terminal_bench_v4_rejects_unsupported_harbor_before_dataset_io(
    harbor_dataset: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, harbor_version: str
) -> None:
    monkeypatch.setattr(terminal_bench_adapter, 'version', lambda package: harbor_version)

    with pytest.raises(ImportError, match=r'Harbor>=0\.14\.0,<1\.0\.0') as error:
        get_benchmark('terminal_bench_v4')

    assert f'found {harbor_version}' in str(error.value)
    assert "pip install --upgrade 'evalscope[terminal_bench]'" in str(error.value)
    assert harbor_dataset == []


@pytest.mark.parametrize('harbor_version', ['0.14.0', '0.24.0'])
def test_terminal_bench_v4_accepts_supported_harbor(
    harbor_dataset: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, harbor_version: str
) -> None:
    monkeypatch.setattr(terminal_bench_adapter, 'version', lambda package: harbor_version)

    adapter = get_benchmark('terminal_bench_v4')

    assert adapter.name == 'terminal_bench_v4'
    assert harbor_dataset == []


@pytest.mark.parametrize('failed_trial', [False, True])
def test_terminal_bench_v4_mock_trial_runs_through_scoring_and_report(
    harbor_dataset: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed_trial: bool
) -> None:
    from evalscope.run import run_task

    class ConfigDouble(BaseModel):
        pass

    class TaskConfigDouble(BaseModel):
        name: str
        ref: str

    class TrialConfigDouble(BaseModel):
        task: BaseModel
        trials_dir: Path
        agent: BaseModel
        verifier: BaseModel
        environment: BaseModel

    class TrialDouble:

        @classmethod
        async def create(cls, config: TrialConfigDouble) -> 'TrialDouble':
            trial = cls()
            trial.agent = SimpleNamespace(_llm=None)
            return trial

        async def run(self) -> Any:
            result = {
                'trial_uri': TRIAL_URI,
                'verifier_result': {'rewards': {'reward': 1}},
            }
            if failed_trial:
                result['exception_info'] = {'exception_type': 'EnvironmentStartTimeoutError', 'message': 'timeout'}
            return SimpleNamespace(model_dump=lambda **kwargs: result)

    class HarborLLMDouble:

        def __init__(self, model: Any) -> None:
            self.perf_metrics = []

    modules = {}
    config_module = ModuleType('harbor.models.trial.config')
    for name in ('AgentConfig', 'EnvironmentConfig', 'VerifierConfig'):
        setattr(config_module, name, ConfigDouble)
    config_module.TaskConfig = TaskConfigDouble
    config_module.TrialConfig = TrialConfigDouble
    modules[config_module.__name__] = config_module
    trial_module = ModuleType('harbor.trial.trial')
    trial_module.Trial = TrialDouble
    modules[trial_module.__name__] = trial_module
    utils_module = ModuleType('evalscope.benchmarks.terminal_bench.utils')
    utils_module.HarborLLM = HarborLLMDouble
    modules[utils_module.__name__] = utils_module
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    reports = run_task(
        TaskConfig(
            model='mock',
            eval_type='mock_llm',
            datasets=['terminal_bench_v4'],
            limit=1,
            eval_batch_size=1,
            work_dir=str(tmp_path),
            no_timestamp=True,
            collect_perf=False,
            ignore_errors=failed_trial,
        )
    )
    report = reports['terminal_bench_v4']

    assert report.dataset_pretty_name == 'Terminal-Bench-4.0'
    assert report.execution_summary.requested == 1
    assert report.execution_summary.errored == int(failed_trial)
    assert report.num == (0 if failed_trial else 1)
    assert report.score == (None if failed_trial else 1.0)
    snapshots = list(tmp_path.glob('configs/*.yaml'))
    assert snapshots
    snapshot = snapshots[0].read_text()
    assert 'dataset_revision: 4.0.0' in snapshot
    assert 'evaluation_version: v1.0' in snapshot


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
