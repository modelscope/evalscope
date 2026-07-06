import copy
import json
import os
import pytest
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import pathname2url

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.benchmarks.deep_swe.deep_swe_adapter import (
    COMMON_EXTRA_PARAMS,
    DEFAULT_HUGGINGFACE_DATASET_ID,
    DEFAULT_MODELSCOPE_DATASET_ID,
    DeepSWEAdapter,
)
from evalscope.config import TaskConfig


class MockModel:

    name = 'mock-model'


def make_adapter(tmp_path: Path, **extra_params: Any) -> DeepSWEAdapter:
    base_extra_params = copy.deepcopy(COMMON_EXTRA_PARAMS)
    for key, value in extra_params.items():
        base_extra_params[key]['value'] = value
    meta = BenchmarkMeta(
        name='deep_swe',
        pretty_name='DeepSWE',
        dataset_id=DEFAULT_MODELSCOPE_DATASET_ID,
        eval_split='test',
        prompt_template='{question}',
        metric_list=['acc'],
        extra_params=base_extra_params,
    )
    cfg = TaskConfig(
        datasets=['deep_swe'],
        dataset_args={'deep_swe': {
            'extra_params': extra_params
        }},
        work_dir=str(tmp_path / 'outputs'),
    )
    adapter = DeepSWEAdapter(benchmark_meta=meta, task_config=cfg)
    return adapter


def write_snapshot(tmp_path: Path, tasks: Optional[list] = None) -> Path:
    snapshot = tmp_path / 'snapshot'
    tasks_dir = snapshot / 'tasks'
    tasks_dir.mkdir(parents=True)
    tasks = tasks or [
        {'task_id': 'task-a', 'language': 'python', 'category': 'bugfix', 'display_description': 'Fix A'},
        {'task_id': 'task-b', 'language': 'go', 'category': 'feature', 'display_description': 'Fix B'},
        {'task_id': 'task-c', 'language': 'python', 'category': 'bugfix', 'display_description': 'Fix C'},
    ]
    (tasks_dir / 'manifest.json').write_text(json.dumps({'tasks': tasks}), encoding='utf-8')
    for task in tasks:
        task_path = tasks_dir / task['task_id']
        task_path.mkdir()
        (task_path / 'task.toml').write_text('id = "task"\n', encoding='utf-8')
        (task_path / 'instruction.md').write_text(f'Instruction for {task["task_id"]}', encoding='utf-8')
    return snapshot


def test_resolve_default_dataset_ids(tmp_path: Path) -> None:
    assert make_adapter(tmp_path)._resolve_dataset_id() == DEFAULT_MODELSCOPE_DATASET_ID
    assert make_adapter(tmp_path, dataset_hub='huggingface')._resolve_dataset_id() == DEFAULT_HUGGINGFACE_DATASET_ID
    assert make_adapter(tmp_path, dataset_id='custom/deep-swe')._resolve_dataset_id() == 'custom/deep-swe'


def test_load_filters_tasks_and_applies_limit_and_seed(monkeypatch: Any, tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path)
    adapter = make_adapter(tmp_path, languages=['python'], categories=['bugfix'], sample_seed=3)
    adapter._task_config.limit = 1

    monkeypatch.setattr('evalscope.benchmarks.deep_swe.deep_swe_adapter._validate_environment_requirements', lambda _: None)
    monkeypatch.setattr(adapter, '_download_snapshot', lambda: snapshot)

    dataset, _ = adapter.load()

    assert len(dataset['test']) == 1
    assert dataset['test'][0].metadata['language'] == 'python'
    assert dataset['test'][0].metadata['category'] == 'bugfix'


def test_load_dataset_post_processes_sample_prompt(monkeypatch: Any, tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path)
    adapter = make_adapter(tmp_path, task_ids=['task-a'])

    monkeypatch.setattr('evalscope.benchmarks.deep_swe.deep_swe_adapter._validate_environment_requirements', lambda _: None)
    monkeypatch.setattr(adapter, '_download_snapshot', lambda: snapshot)

    dataset = adapter.load_dataset()

    assert len(dataset['test']) == 1
    assert dataset['test'][0].input[-1].text == 'Instruction for task-a'


def test_load_validates_snapshot_layout(monkeypatch: Any, tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    snapshot = tmp_path / 'missing-manifest'
    snapshot.mkdir()

    monkeypatch.setattr('evalscope.benchmarks.deep_swe.deep_swe_adapter._validate_environment_requirements', lambda _: None)
    monkeypatch.setattr(adapter, '_download_snapshot', lambda: snapshot)

    with pytest.raises(FileNotFoundError, match='manifest.json'):
        adapter.load()


def test_load_validates_task_toml(monkeypatch: Any, tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path, tasks=[{'task_id': 'task-a'}])
    (snapshot / 'tasks' / 'task-a' / 'task.toml').unlink()
    adapter = make_adapter(tmp_path)

    monkeypatch.setattr('evalscope.benchmarks.deep_swe.deep_swe_adapter._validate_environment_requirements', lambda _: None)
    monkeypatch.setattr(adapter, '_download_snapshot', lambda: snapshot)

    with pytest.raises(FileNotFoundError, match='task.toml'):
        adapter.load()


def test_build_score_metadata_collects_reward_and_artifacts(tmp_path: Path) -> None:
    trial_dir = tmp_path / 'trial'
    (trial_dir / 'verifier').mkdir(parents=True)
    (trial_dir / 'agent').mkdir()
    (trial_dir / 'artifacts').mkdir()
    (trial_dir / 'verifier' / 'reward.json').write_text(
        json.dumps({'reward': 1, 'partial': 0.5, 'f2p': 2, 'p2p': 3, 'apply_failed': False}),
        encoding='utf-8',
    )
    (trial_dir / 'verifier' / 'reward.txt').write_text('1\n', encoding='utf-8')
    adapter = make_adapter(tmp_path)

    metadata = adapter._build_score_metadata({
        'job_result_path': str(tmp_path / 'job'),
        'trial_results': [{
            'trial_uri': f'file://{trial_dir}',
            'verifier_result': {
                'rewards': {
                    'reward': 0
                }
            },
        }],
    })

    assert metadata['reward'] == 1
    assert metadata['partial'] == 0.5
    assert metadata['f2p'] == 2
    assert metadata['p2p'] == 3
    assert metadata['apply_failed'] is False
    assert metadata['pier_job_result_path'] == str(tmp_path / 'job')
    assert metadata['verifier_reward_json_path'] == str(trial_dir / 'verifier' / 'reward.json')
    assert metadata['trajectory_path'] == str(trial_dir / 'agent' / 'trajectory.json')


def test_on_inference_uses_mock_pier_result_and_scores_acc(monkeypatch: Any, tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    sample = Sample(input='', metadata={'task_id': 'task-a', 'task_path': str(tmp_path / 'task-a')})
    result = {
        'job_result_path': str(tmp_path / 'job'),
        'trial_results': [{
            'trial_uri': f'file://{tmp_path / "trial"}',
            'verifier_result': {
                'rewards': {
                    'reward': 1
                }
            },
        }],
    }
    monkeypatch.setattr(adapter, '_run_pier_job', lambda model, sample: result)

    inference = adapter._on_inference(MockModel(), sample)
    task_state = TaskState(model='mock-model', sample=sample, output=inference.output)
    score = adapter.match_score('', '', '', task_state)

    assert inference.output.completion == f'file://{tmp_path / "trial"}'
    assert score.value == {'acc': 1.0}
    assert score.metadata['reward'] == 1


def test_pier_exception_result_raises() -> None:
    with pytest.raises(RuntimeError, match='RewardFileNotFoundError'):
        DeepSWEAdapter._raise_for_pier_failures({
            'trial_results': [{
                'exception_info': {
                    'exception_type': 'RewardFileNotFoundError',
                    'message': 'reward.txt missing',
                }
            }]
        })


def test_missing_pier_reward_without_exception_raises() -> None:
    with pytest.raises(RuntimeError, match='did not return a reward or exception info'):
        DeepSWEAdapter._raise_for_pier_failures({
            'trial_results': [{
                'verifier_result': {
                    'rewards': {}
                }
            }]
        })


def test_pier_exception_with_verifier_reward_is_scored(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    result = {
        'trial_results': [{
            'exception_info': {
                'exception_type': 'NonZeroAgentExitCodeError',
            },
            'verifier_result': {
                'rewards': {
                    'reward': 0,
                    'partial': 0.25,
                }
            },
        }]
    }

    DeepSWEAdapter._raise_for_pier_failures(result)
    metadata = adapter._build_score_metadata(result)

    assert metadata['reward'] == 0
    assert metadata['partial'] == 0.25
    assert metadata['agent_execution_failed'] is True


def test_artifact_path_decodes_file_uri(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path)
    trial_dir = tmp_path / 'trial dir'
    result = {'trial_results': [{'trial_uri': f'file://{pathname2url(str(trial_dir))}'}]}

    assert adapter._artifact_path(result, 'verifier/reward.json') == trial_dir / 'verifier' / 'reward.json'


def test_parse_timestamp_handles_z_suffix() -> None:
    assert DeepSWEAdapter._parse_timestamp('2026-01-01T00:00:00Z') == DeepSWEAdapter._parse_timestamp(
        '2026-01-01T00:00:00+00:00'
    )


def install_fake_pier(monkeypatch: Any, captured: Dict[str, Any], result: Dict[str, Any]) -> None:

    class FakeConfig:

        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class FakeJob:

        def __init__(self, config: Any) -> None:
            self.config = config

        @classmethod
        async def create(cls, config: Any) -> 'FakeJob':
            captured['config'] = config
            return cls(config)

        async def run(self) -> Any:

            class FakeResult:

                @staticmethod
                def model_dump(mode: str = 'json') -> Dict[str, Any]:
                    return result

            return FakeResult()

    pier = types.ModuleType('pier')
    pier_job = types.ModuleType('pier.job')
    pier_job.Job = FakeJob
    pier_models = types.ModuleType('pier.models')
    pier_models_job = types.ModuleType('pier.models.job')
    pier_models_job_config = types.ModuleType('pier.models.job.config')
    pier_models_job_config.JobConfig = FakeConfig
    pier_models_trial = types.ModuleType('pier.models.trial')
    pier_models_trial_config = types.ModuleType('pier.models.trial.config')
    pier_models_trial_config.AgentConfig = FakeConfig
    pier_models_trial_config.EnvironmentConfig = FakeConfig
    pier_models_trial_config.TaskConfig = FakeConfig
    pier_models_trial_config.VerifierConfig = FakeConfig

    for name, module in {
        'pier': pier,
        'pier.job': pier_job,
        'pier.models': pier_models,
        'pier.models.job': pier_models_job,
        'pier.models.job.config': pier_models_job_config,
        'pier.models.trial': pier_models_trial,
        'pier.models.trial.config': pier_models_trial_config,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr('evalscope.benchmarks.deep_swe.deep_swe_adapter.check_import', lambda *args, **kwargs: True)


def test_run_pier_job_uses_adhoc_task_source(monkeypatch: Any, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}
    result = {
        'trial_results': [{
            'trial_uri': f'file://{tmp_path / "trial"}',
            'verifier_result': {
                'rewards': {
                    'reward': 1
                }
            },
        }]
    }
    install_fake_pier(monkeypatch, captured, result)
    task_path = tmp_path / 'tasks' / 'task-a'
    task_path.mkdir(parents=True)
    adapter = make_adapter(tmp_path, agent_kwargs={'model_class': 'litellm'}, environment_kwargs={'delete': True})
    sample = Sample(input='', metadata={'task_id': 'task-a', 'task_path': str(task_path)})

    result_dict = adapter._run_pier_job(MockModel(), sample)

    assert result_dict['job_result_path'].startswith(str(Path(adapter.output_dir) / 'deep_swe_jobs'))
    assert captured['config'].tasks[0].path == task_path
    assert getattr(captured['config'].tasks[0], 'source', None) is None
    assert captured['config'].agents[0].kwargs == {'model_class': 'litellm'}
    assert captured['config'].environment.type == 'docker'
    assert captured['config'].environment.delete is True


@pytest.mark.skipif(
    os.getenv('EVALSCOPE_DEEP_SWE_E2E') != '1' or not os.getenv('OPENAI_API_KEY'),
    reason='Set EVALSCOPE_DEEP_SWE_E2E=1 and OPENAI_API_KEY to run the real Pier DeepSWE smoke test.',
)
def test_deep_swe_real_e2e(tmp_path: Path) -> None:
    from evalscope import run_task

    agent_kwargs = {
        'cost_limit': float(os.getenv('EVALSCOPE_DEEP_SWE_COST_LIMIT', '0.05')),
        'model_class': os.getenv('EVALSCOPE_DEEP_SWE_MODEL_CLASS', 'litellm'),
    }
    step_limit = os.getenv('EVALSCOPE_DEEP_SWE_AGENT_STEP_LIMIT')
    if step_limit:
        agent_kwargs['config_yaml'] = f'agent:\n  step_limit: {int(step_limit)}\n'

    result = run_task(
        TaskConfig(
            model=os.getenv('EVALSCOPE_DEEP_SWE_MODEL', 'qwen-plus'),
            eval_type='mock_llm',
            datasets=['deep_swe'],
            limit=1,
            eval_batch_size=1,
            work_dir=str(tmp_path / 'outputs'),
            no_timestamp=True,
            dataset_args={
                'deep_swe': {
                    'extra_params': {
                        'task_ids': ['abs-module-cache-flags'],
                        'agent_kwargs': agent_kwargs,
                    }
                }
            },
        )
    )

    assert isinstance(result, dict)
    assert 'deep_swe' in result
