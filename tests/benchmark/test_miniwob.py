"""Tests for the OpenEnv MiniWoB adapter."""

import csv
import hashlib
import io
import json
import numpy as np
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from evalscope.api.agent import AgentContext, EventType, NativeAgentConfig
from evalscope.api.benchmark.adapters import AgentAdapter, AgentLoopAdapter
from evalscope.api.environment import EnvironmentStepResult
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.metric import AggScore
from evalscope.api.model import ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.registry import get_benchmark
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.benchmarks.miniwob.miniwob_adapter import OPENENV_PATCH_SHA256, MiniWobAdapter, _MiniWobStrategy
from evalscope.benchmarks.miniwob.utils import load_miniwob_records, validate_browser_action
from evalscope.config import TaskConfig
from evalscope.models.mockllm import MockLLM
from evalscope.run import run_task
from evalscope.utils.doc_utils.generate_dataset_md import extract_adapter_meta


class FakeMiniWobSession:
    """OpenEnv-shaped task session for deterministic adapter tests."""

    backend_name = 'openenv'

    def __init__(
        self,
        *,
        reset_error=None,
        step_error=None,
        screenshot=None,
        step_reward=1.0,
        step_done=True,
        last_action_error=False,
    ):
        self.reset_error = reset_error
        self.step_error = step_error
        self.screenshot = screenshot
        self.step_reward = step_reward
        self.step_done = step_done
        self.last_action_error = last_action_error
        self.operations = []
        self.closed = False

    async def state(self):
        return {'benchmark': 'miniwob'}

    async def reset(self, **kwargs):
        self.operations.append(('reset', kwargs))
        if self.reset_error:
            raise RuntimeError(self.reset_error)
        return EnvironmentStepResult(
            observation={
                'goal': 'Click OK',
                'url': 'http://miniwob/click-dialog.html',
                'axtree_txt': '[1] button "OK"',
                'screenshot': self.screenshot,
                'last_action_error': False,
            },
            reward=0.0,
            done=False,
        )

    async def step(self, action):
        self.operations.append(('step', action))
        if self.step_error:
            raise RuntimeError(self.step_error)
        return EnvironmentStepResult(
            observation={
                'goal': 'Click OK',
                'url': 'http://miniwob/click-dialog.html',
                'axtree_txt': '[1] button "OK"',
                'screenshot': self.screenshot,
                'last_action_error': self.last_action_error,
                'error': 'invalid action' if self.last_action_error else '',
            },
            reward=self.step_reward,
            done=self.step_done,
        )

    async def close(self):
        self.closed = True


class FakeEnvironmentLease:
    """Owned service runtime used alongside ``FakeMiniWobSession``."""

    name = 'ms_enclave_docker'

    def __init__(self):
        self.base_url = 'http://127.0.0.1:18123'
        self.closed = False
        self.log_paths = []

    async def capture_logs(self, path):
        self.log_paths.append(Path(path))
        return True

    async def close(self):
        self.closed = True


def _environment_pair(session=None, lease=None):
    if session is None:
        screenshot = np.zeros((2, 3, 3), dtype=np.uint8).tolist()
        session = FakeMiniWobSession(screenshot=screenshot)
    return lease or FakeEnvironmentLease(), session


def _adapter(*, observation_mode='axtree', agent_config=None, task_environment=None, repeats=1):
    if task_environment is None and agent_config is None:
        task_environment = {
            'backend': 'openenv',
            'observation_mode': observation_mode,
            'runtime': {
                'name': 'ms_enclave_docker',
            },
        }
    if task_environment is not None:
        assert agent_config is None
        agent_config = {
            'task_environment': task_environment,
        }
    return get_benchmark(
        'miniwob',
        TaskConfig(
            model='mock',
            datasets=['miniwob'],
            eval_type='mock_llm',
            agent_config=agent_config,
            repeats=repeats,
        ),
    )


def _default_adapter():
    return get_benchmark(
        'miniwob',
        TaskConfig(
            model='mock',
            datasets=['miniwob'],
            eval_type='mock_llm',
        ),
    )


def test_observation_mode_is_task_environment_config_not_dataset_arg():
    with pytest.raises(KeyError, match='observation_mode'):
        get_benchmark(
            'miniwob',
            TaskConfig(
                model='mock',
                datasets=['miniwob'],
                eval_type='mock_llm',
                dataset_args={'miniwob': {'extra_params': {'observation_mode': 'axtree'}}},
            ),
        )


def _records(repeats=1):
    return [{
        'task_name': 'miniwob.click-dialog',
        'miniwob_category': 'test',
        'comment': '',
        'webgum_subset': 'False',
        'similarity_group': '0',
        'browsergym_split': 'test',
        'task_id': 'miniwob.click-dialog',
        'openenv_task_name': 'click-dialog',
        '_episode_seeds': list(range(28, 28 + repeats)),
        'seed': 28,
        'repeat': 0,
    }]


def _model(environment):
    call = ToolCall(
        id='browser-1',
        function=ToolFunction(name='browser_action', arguments={'action': 'click("1")'}),
    )
    output = ModelOutput(
        model='mock',
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content='', tool_calls=[call]),
                stop_reason='tool_calls',
            )
        ],
    )
    model = MagicMock()
    model.name = 'mock'

    async def generate_async(*, input, tools):
        assert environment.operations[0][0] == 'reset'
        assert input[0].role == 'system'
        assert 'Click OK' in input[1].text
        assert [tool.name for tool in tools] == ['browser_action']
        return output

    model.generate_async = AsyncMock(side_effect=generate_async)
    return model


def _final_answer_model(content='done'):
    model = MagicMock()
    model.name = 'mock'
    model.generate_async = AsyncMock(return_value=ModelOutput.from_content(model='mock', content=content))
    return model


def _metadata_csv(task_count=125):
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            'task_name',
            'miniwob_category',
            'comment',
            'webgum_subset',
            'similarity_group',
            'browsergym_split',
        ],
        lineterminator='\n',
    )
    writer.writeheader()
    for index in range(task_count):
        writer.writerow({
            'task_name': f'miniwob.task-{index:03d}',
            'miniwob_category': 'test',
            'comment': '',
            'webgum_subset': 'False',
            'similarity_group': str(index),
            'browsergym_split': 'test',
        })
    return output.getvalue().encode()


def _fake_download(csv_bytes):
    """Stand-in for the shared download helper that writes csv_bytes to save_path."""

    def _download(url, save_path, **kwargs):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(csv_bytes)

    return _download


def test_schedule_download_cache_and_generation(tmp_path: Path):
    csv_bytes = _metadata_csv()
    checksum = hashlib.sha256(csv_bytes).hexdigest()
    schedule_checksum = '64c3fbbacc44bf05c6d23b916a0fd01f5c7d71612cc732cb68c79242edb2c1e7'

    with patch('evalscope.benchmarks.miniwob.utils.BROWSERGYM_METADATA_SHA256', checksum), \
            patch('evalscope.benchmarks.miniwob.utils.MINIWOB_SCHEDULE_SHA256_BY_REPEATS', {1: schedule_checksum}), \
            patch(
                'evalscope.benchmarks.miniwob.utils.download_url', side_effect=_fake_download(csv_bytes)
            ) as download:
        records, path = load_miniwob_records(tmp_path)
        cached_records, cached_path = load_miniwob_records(tmp_path)

    assert len(records) == 125
    assert path == cached_path
    assert records == cached_records
    assert download.call_count == 1
    assert records[0]['task_id'] == 'miniwob.task-000'
    assert [record['_episode_seeds'][0] for record in records[:5]] == [
        1608637542,
        3421126067,
        4083286876,
        787846414,
        3143890026,
    ]
    assert max(seed for record in records for seed in record['_episode_seeds']) < (2**32)


def test_schedule_digest_is_stable(tmp_path: Path):
    csv_bytes = _metadata_csv()
    checksum = hashlib.sha256(csv_bytes).hexdigest()
    destination = tmp_path / 'sources' / 'browsergym' / (
        '0a785fbed075224ae81ca9c1fe924f66050696fe/miniwob.csv'
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(csv_bytes)

    with patch('evalscope.benchmarks.miniwob.utils.BROWSERGYM_METADATA_SHA256', checksum), \
            patch(
                'evalscope.benchmarks.miniwob.utils.MINIWOB_SCHEDULE_SHA256_BY_REPEATS',
                {5: '50d15c6da5fb326cd20c06cd0a75f43f7badc2fa21d21ab76e993c4931de2cdb'},
            ):
        records, _ = load_miniwob_records(tmp_path, repeats=5)

    digest = hashlib.sha256(
        '\n'.join(f"{record['task_id']}:{seed}" for record in records for seed in record['_episode_seeds']).encode()
    ).hexdigest()
    assert digest == '50d15c6da5fb326cd20c06cd0a75f43f7badc2fa21d21ab76e993c4931de2cdb'


def test_corrupt_cache_is_redownloaded_atomically(tmp_path: Path):
    csv_bytes = _metadata_csv()
    checksum = hashlib.sha256(csv_bytes).hexdigest()
    schedule_checksum = '64c3fbbacc44bf05c6d23b916a0fd01f5c7d71612cc732cb68c79242edb2c1e7'
    destination = tmp_path / 'sources' / 'browsergym' / (
        '0a785fbed075224ae81ca9c1fe924f66050696fe/miniwob.csv'
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'corrupt')

    with patch('evalscope.benchmarks.miniwob.utils.BROWSERGYM_METADATA_SHA256', checksum), \
            patch('evalscope.benchmarks.miniwob.utils.MINIWOB_SCHEDULE_SHA256_BY_REPEATS', {1: schedule_checksum}), \
            patch('evalscope.benchmarks.miniwob.utils.download_url', side_effect=_fake_download(csv_bytes)):
        records, path = load_miniwob_records(tmp_path)

    assert len(records) == 125
    assert path.read_bytes() == csv_bytes
    assert not list(destination.parent.glob('tmp*'))


def test_offline_cache_hit_and_no_cache_failure(tmp_path: Path):
    csv_bytes = _metadata_csv()
    checksum = hashlib.sha256(csv_bytes).hexdigest()
    schedule_checksum = '64c3fbbacc44bf05c6d23b916a0fd01f5c7d71612cc732cb68c79242edb2c1e7'
    destination = tmp_path / 'sources' / 'browsergym' / (
        '0a785fbed075224ae81ca9c1fe924f66050696fe/miniwob.csv'
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(csv_bytes)

    with patch('evalscope.benchmarks.miniwob.utils.BROWSERGYM_METADATA_SHA256', checksum), \
            patch('evalscope.benchmarks.miniwob.utils.MINIWOB_SCHEDULE_SHA256_BY_REPEATS', {1: schedule_checksum}), \
            patch('evalscope.benchmarks.miniwob.utils.download_url', side_effect=OSError('offline')) as download:
        records, _ = load_miniwob_records(tmp_path)
    assert len(records) == 125
    download.assert_not_called()

    with patch('evalscope.benchmarks.miniwob.utils.download_url', side_effect=OSError('offline')):
        with pytest.raises(RuntimeError, match='No ModelScope or Hugging Face fallback'):
            load_miniwob_records(tmp_path / 'empty')


def test_adapter_registers_only_full_miniwob():
    adapter = _adapter()
    with patch(
        'evalscope.benchmarks.miniwob.miniwob_adapter.load_miniwob_records',
        return_value=(_records(), Path('/cache/miniwob.csv')),
    ):
        dataset = adapter.load_dataset()['default']

    assert isinstance(adapter, MiniWobAdapter)
    assert isinstance(adapter, AgentAdapter)
    assert not isinstance(adapter, AgentLoopAdapter)
    assert 'agent_config' not in extract_adapter_meta(adapter)
    assert len(dataset) == 1
    with pytest.raises(ValueError, match='not found'):
        get_benchmark('miniwob_tiny', TaskConfig(model='mock', datasets=['miniwob_tiny'], eval_type='mock_llm'))


def test_repeats_generate_distinct_seeded_episodes():
    adapter = _adapter(repeats=5)
    with patch(
        'evalscope.benchmarks.miniwob.miniwob_adapter.load_miniwob_records',
        return_value=(_records(repeats=5), Path('/cache/miniwob.csv')),
    ) as load_records:
        dataset = adapter.load_dataset()['default']

    load_records.assert_called_once_with(repeats=5)
    assert len(dataset) == 5
    assert [sample.metadata['seed'] for sample in dataset] == [28, 29, 30, 31, 32]
    assert [sample.metadata['repeat'] for sample in dataset] == [0, 1, 2, 3, 4]
    assert [sample.group_id for sample in dataset] == [0, 0, 0, 0, 0]
    assert all('_episode_seeds' not in sample.metadata for sample in dataset)


def test_reset_step_reward_and_score(tmp_path: Path):
    adapter = _adapter()
    sample = adapter.record_to_sample(_records()[0])
    sample.id = 0
    session = FakeMiniWobSession()
    lease = FakeEnvironmentLease()
    model = _model(session)

    with patch.object(adapter, 'build_task_environment', return_value=(lease, session)):
        task_state = adapter.run_inference(model, sample, str(tmp_path))
    sample_score = adapter.calculate_metrics(task_state)

    assert task_state.output.completion == '1'
    assert sample.metadata['success'] is True
    assert sample.metadata['runtime_error'] is None
    assert session.operations == [
        ('reset', {
            'seed': 28,
            'task_name': 'click-dialog'
        }),
        ('step', {
            'action_str': 'click("1")'
        }),
    ]
    assert session.closed is True
    assert lease.closed is True
    assert lease.log_paths == []
    assert sample_score.score.value == {
        'success_rate': 1.0,
        'error_rate': 0.0
    }
    assert sample_score.score.main_score_name == 'success_rate'
    assert task_state.agent_trace.events[0].type == EventType.ENV_RESET
    assert task_state.agent_trace.events[0].payload['reward'] == 0.0


def test_reset_failure_scores_zero_and_captures_logs(tmp_path: Path):
    adapter = _adapter()
    sample = adapter.record_to_sample(_records()[0])
    sample.id = 0
    session = FakeMiniWobSession(reset_error='reset failed')
    lease = FakeEnvironmentLease()
    model = MagicMock(name='model')
    model.name = 'mock'

    with patch.object(adapter, 'build_task_environment', return_value=(lease, session)):
        task_state = adapter.run_inference(model, sample, str(tmp_path))
    sample_score = adapter.calculate_metrics(task_state)

    assert task_state.output.completion == '0'
    assert 'reset failed' in sample.metadata['runtime_error']
    assert session.closed is True
    assert lease.closed is True
    assert lease.log_paths[0].name == 'openenv-container.log'
    assert sample_score.score.value == {
        'success_rate': 0.0,
        'error_rate': 1.0
    }


def test_step_failure_adds_runtime_error_trace(tmp_path: Path):
    adapter = _adapter()
    sample = adapter.record_to_sample(_records()[0])
    sample.id = 0
    session = FakeMiniWobSession(step_error='socket closed')
    lease = FakeEnvironmentLease()

    with patch.object(adapter, 'build_task_environment', return_value=(lease, session)):
        task_state = adapter.run_inference(_model(session), sample, str(tmp_path))
    sample_score = adapter.calculate_metrics(task_state)

    assert task_state.output.completion == '0'
    assert sample_score.score.value == {
        'success_rate': 0.0,
        'error_rate': 1.0
    }
    assert any(
        event.type == EventType.ERROR and event.payload.get('source') == 'openenv_runtime'
        for event in task_state.agent_trace.events
    )


def test_early_model_finish_and_non_positive_reward_are_normal_failures(tmp_path: Path):
    adapter = _adapter()
    early_sample = adapter.record_to_sample(_records()[0])
    early_sample.id = 0
    early_session = FakeMiniWobSession()

    with patch.object(adapter, 'build_task_environment', return_value=_environment_pair(early_session)):
        early_state = adapter.run_inference(_final_answer_model(), early_sample, str(tmp_path))
    assert adapter.calculate_metrics(early_state).score.value == {
        'success_rate': 0.0,
        'error_rate': 0.0
    }

    failed_sample = adapter.record_to_sample(_records()[0])
    failed_sample.id = 1
    failed_session = FakeMiniWobSession(step_reward=-1.0)
    with patch.object(adapter, 'build_task_environment', return_value=_environment_pair(failed_session)):
        failed_state = adapter.run_inference(_model(failed_session), failed_sample, str(tmp_path))
    assert adapter.calculate_metrics(failed_state).score.value == {
        'success_rate': 0.0,
        'error_rate': 0.0
    }


def test_model_exception_is_not_a_runtime_error(tmp_path: Path):
    adapter = _adapter()
    sample = adapter.record_to_sample(_records()[0])
    sample.id = 0
    session = FakeMiniWobSession()
    model = MagicMock()
    model.name = 'mock'
    model.generate_async = AsyncMock(side_effect=RuntimeError('model unavailable'))

    with patch.object(adapter, 'build_task_environment', return_value=_environment_pair(session)):
        task_state = adapter.run_inference(model, sample, str(tmp_path))

    assert adapter.calculate_metrics(task_state).score.value == {
        'success_rate': 0.0,
        'error_rate': 0.0
    }
    assert sample.metadata['runtime_error'] is None
    assert 'model unavailable' in sample.metadata['model_error']


def test_default_screenshot_is_saved_and_attached(tmp_path: Path):
    adapter = _default_adapter()
    assert adapter.observation_mode == 'axtree_screenshot'
    sample = adapter.record_to_sample(_records()[0])
    assert sample.metadata['observation_mode'] == 'axtree_screenshot'
    sample.id = 0
    screenshot = np.zeros((2, 3, 3), dtype=np.uint8).tolist()
    session = FakeMiniWobSession(screenshot=screenshot)
    model = _model(session)

    with patch.object(adapter, 'build_task_environment', return_value=_environment_pair(session)):
        task_state = adapter.run_inference(model, sample, str(tmp_path))

    images = sorted((Path(sample.metadata['artifact_dir'])).glob('step-*.png'))
    assert [image.name for image in images] == ['step-000.png', 'step-001.png']
    assert any(
        message.role == 'user' and isinstance(message.content, list)
        for message in task_state.messages
    )


def test_default_screenshot_mode_rejects_missing_screenshot(tmp_path: Path):
    adapter = _default_adapter()
    sample = adapter.record_to_sample(_records()[0])
    sample.id = 0
    session = FakeMiniWobSession(screenshot=None)

    with patch.object(adapter, 'build_task_environment', return_value=_environment_pair(session)):
        task_state = adapter.run_inference(_final_answer_model(), sample, str(tmp_path))

    assert adapter.calculate_metrics(task_state).score.value == {
        'success_rate': 0.0,
        'error_rate': 1.0
    }
    assert 'did not return a screenshot' in sample.metadata['runtime_error']


@pytest.mark.parametrize(
    'action',
    [
        'click("1"); fill("2", "x")',
        'x = click("1")',
        'page.click("1")',
        'click(noop())',
        'unknown_action("1")',
    ],
)
def test_browser_action_rejects_non_single_calls(action):
    with pytest.raises(ValueError):
        validate_browser_action(action)


def test_browser_action_accepts_miniwob_all_coordinate_action_and_rejects_default_only_action():
    assert validate_browser_action('mouse_click(420, 260)') == 'mouse_click(420, 260)'
    with pytest.raises(ValueError, match='miniwob_all'):
        validate_browser_action('goto("https://example.test")')
    with pytest.raises(ValueError, match='string BID'):
        validate_browser_action('click(420, 260)')


def test_observation_exposes_screenshot_pixel_size_and_raw_browsergym_action_error():
    adapter = _adapter()
    sample = adapter.record_to_sample(_records()[0])
    result = EnvironmentStepResult(
        observation={
            'goal': 'Click the target',
            'url': 'http://miniwob.test',
            'axtree_txt': "RootWebArea 'Task'",
            'screenshot': np.zeros((2, 3, 3), dtype=np.uint8).tolist(),
            'last_action_error': True,
            'error': '',
        },
        reward=0.0,
        done=False,
        metadata={
            'browsergym_obs': {
                'last_action_error': 'TypeError: click expects a string BID',
            }
        },
    )

    observation = adapter._merge_step_result(sample, result, action='click(420, 260)')
    text = adapter._format_observation(observation)

    assert 'Screenshot size: 3x2 pixels' in text
    assert 'not normalized 0-1000 coordinates' in text
    assert 'Environment message: TypeError: click expects a string BID' in text


def test_agent_config_allows_step_count_override_but_rejects_strategy_override():
    assert _adapter(agent_config=NativeAgentConfig(max_steps=9)).max_steps == 9
    with pytest.raises(ValueError, match='fixes its internal strategy'):
        _adapter(agent_config=NativeAgentConfig(strategy='function_calling'))


def test_report_contains_fixed_profile_metadata():
    adapter = _adapter(repeats=5)
    report = adapter._on_generate_report(
        {
            'default': [
                AggScore(metric_name='success_rate', aggregation_name='mean', score=0.5, num=2),
                AggScore(metric_name='error_rate', aggregation_name='mean', score=0.0, num=2),
            ]
        },
        model_name='mock',
    )

    assert report.metadata['profile'] == 'openenv_v0.4.1_miniwob_all_10_steps'
    assert report.metadata['max_steps'] == 10
    assert report.metadata['repeats'] == 5
    assert report.metadata['official_browsergym_action_config'] is True
    assert report.metadata['official_browsergym_evaluation_protocol'] is True
    assert report.metadata['runtime_mode'] == 'local'
    assert report.metadata['observation_mode'] == 'axtree'
    patch_path = Path(__file__).parents[2] / 'evalscope/benchmarks/miniwob/runtime/openenv-miniwob-all.patch'
    assert OPENENV_PATCH_SHA256 == hashlib.sha256(patch_path.read_bytes()).hexdigest()
    assert report.metadata['openenv_patch_sha256'] == OPENENV_PATCH_SHA256
    assert report.metadata['csv_sha256'] == (
        '37117db27909a17b1b78035528472922c98c479a54619ac398dc256a7d2fef09'
    )


def test_report_marks_custom_step_count_as_non_official():
    adapter = _adapter(agent_config=NativeAgentConfig(max_steps=9))
    report = adapter._on_generate_report(
        {
            'default': [
                AggScore(metric_name='success_rate', aggregation_name='mean', score=0.0, num=1),
                AggScore(metric_name='error_rate', aggregation_name='mean', score=0.0, num=1),
            ]
        },
        model_name='mock',
    )

    assert report.metadata['profile'] == 'openenv_v0.4.1_miniwob_all_9_steps'
    assert report.metadata['max_steps'] == 9
    assert report.metadata['repeats'] == 1
    assert report.metadata['official_browsergym_evaluation_protocol'] is False


def test_report_marks_limited_five_seed_run_as_non_official():
    adapter = get_benchmark(
        'miniwob',
        TaskConfig(
            model='mock',
            datasets=['miniwob'],
            eval_type='mock_llm',
            repeats=5,
            limit=10,
        ),
    )
    report = adapter._on_generate_report(
        {
            'default': [
                AggScore(metric_name='success_rate', aggregation_name='mean', score=0.0, num=50),
                AggScore(metric_name='error_rate', aggregation_name='mean', score=0.0, num=50),
            ]
        },
        model_name='mock',
    )

    assert report.metadata['repeats'] == 5
    assert report.metadata['official_browsergym_evaluation_protocol'] is False


def test_strategy_rejects_multiple_tool_calls():
    calls = [
        ToolCall(id='one', function=ToolFunction(name='browser_action', arguments={'action': 'click("1")'})),
        ToolCall(id='two', function=ToolFunction(name='browser_action', arguments={'action': 'click("2")'})),
    ]
    output = ModelOutput(
        model='mock',
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content='', tool_calls=calls),
                stop_reason='tool_calls',
            )
        ],
    )
    parsed = _MiniWobStrategy().parse_output(
        output,
        AgentContext(sample_id=0, messages=[]),
    )
    assert parsed.error == 'Call exactly one browser_action tool per turn.'


def test_miniwob_rejects_unsupported_runtime_and_local_image_is_prepared_once():
    import evalscope.benchmarks.miniwob.miniwob_adapter as adapter_module

    adapter_module._RUNTIME_IMAGE_TAG = None
    with patch.object(MiniWobAdapter, '_prepare_runtime_image') as prepare_unsupported:
        with pytest.raises(ValueError, match="supports only task_environment.runtime.name='ms_enclave_docker'"):
            _adapter(
                task_environment={
                    'backend': 'openenv',
                    'runtime': {
                        'name': 'remote',
                        'config': {'base_url': 'http://trusted.example:8000'},
                    },
                }
            )
        prepare_unsupported.assert_not_called()

    local = _adapter()
    with patch.object(adapter_module, 'prepare_docker_image', return_value=SimpleNamespace(image_tag='pinned:image')) as build:
        assert local._prepare_runtime_image() == 'pinned:image'
        assert local._prepare_runtime_image() == 'pinned:image'
        build.assert_called_once()
        spec = build.call_args.args[0]
        assert Path(spec.context_dir).name == 'runtime'
        assert spec.build_args['OPENENV_COMMIT'] == '65c506ef94bb1f7279cb4359673b3ef81031d01f'
        assert spec.build_args['EVALSCOPE_PIP_INDEX_URL'] == 'https://pypi.org/simple'

    adapter_module._RUNTIME_IMAGE_TAG = None
    with patch.dict(adapter_module.os.environ, {'EVALSCOPE_PIP_INDEX_URL': 'https://mirror.example/simple'}), \
            patch.object(
                adapter_module,
                'prepare_docker_image',
                return_value=SimpleNamespace(image_tag='custom:index'),
            ) as build:
        assert local._prepare_runtime_image() == 'custom:index'
        spec = build.call_args.args[0]
        assert spec.build_args['EVALSCOPE_PIP_INDEX_URL'] == 'https://mirror.example/simple'
    adapter_module._RUNTIME_IMAGE_TAG = None


def test_runtime_dependencies_pin_openenv_transitive_imports():
    repo_root = Path(__file__).parents[2]
    benchmark_dir = repo_root / 'evalscope' / 'benchmarks' / 'miniwob'
    dockerfile = (benchmark_dir / 'runtime' / 'Dockerfile').read_text(encoding='utf-8')
    requirements = (repo_root / 'requirements' / 'miniwob.txt').read_text(encoding='utf-8')

    for dependency in ('fastmcp==3.0.0', 'gradio==6.20.0', 'openenv==0.4.1'):
        assert dependency in dockerfile
        assert dependency in requirements
    assert 'PLAYWRIGHT_BROWSERS_PATH=/ms-playwright' in dockerfile
    assert 'playwright install chromium' not in dockerfile


def test_five_parallel_samples_never_activate_more_than_four_environments(tmp_path: Path):
    adapter = _adapter()
    lock = threading.Lock()
    active = 0
    maximum = 0

    class TrackingSession(FakeMiniWobSession):

        def __init__(self):
            nonlocal active, maximum
            super().__init__()
            with lock:
                active += 1
                maximum = max(maximum, active)

        async def reset(self, **kwargs):
            time.sleep(0.05)
            return await super().reset(**kwargs)

        async def close(self):
            nonlocal active
            with lock:
                active -= 1
            await super().close()

    def run_sample(index):
        sample = adapter.record_to_sample(_records()[0])
        sample.id = index
        return adapter.run_inference(_final_answer_model(), sample, str(tmp_path))

    with patch.object(
        adapter,
        'build_task_environment',
        side_effect=lambda sample: _environment_pair(TrackingSession()),
    ):
        with ThreadPoolExecutor(max_workers=5) as executor:
            states = list(executor.map(run_sample, range(5)))

    assert len(states) == 5
    assert maximum == 4
    assert active == 0


def test_mock_model_runs_through_full_evaluator_pipeline(tmp_path: Path, monkeypatch):
    call = ToolCall(
        id='browser-1',
        function=ToolFunction(name='browser_action', arguments={'action': 'click("1")'}),
    )
    output = ModelOutput(
        model='mock_llm',
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content='', tool_calls=[call]),
                stop_reason='tool_calls',
            )
        ],
    )
    original_init = MockLLM.__init__

    def patched_init(self, *args, **kwargs):
        kwargs['custom_outputs'] = [output]
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(MockLLM, '__init__', patched_init)
    monkeypatch.setattr(
        'evalscope.benchmarks.miniwob.miniwob_adapter.load_miniwob_records',
        lambda repeats=1: (_records(repeats=repeats), Path('/cache/miniwob.csv')),
    )
    monkeypatch.setattr(
        MiniWobAdapter,
        'build_task_environment',
        lambda self, sample: _environment_pair(),
    )

    reports = run_task(
        TaskConfig(
            model='mock_llm',
            datasets=['miniwob'],
            eval_type='mock_llm',
            limit=1,
            eval_batch_size=1,
            work_dir=str(tmp_path),
            no_timestamp=True,
            analysis_report=False,
        )
    )

    report = reports['miniwob']
    assert report.score == 1.0
    assert report.metadata['profile'] == 'openenv_v0.4.1_miniwob_all_10_steps'
    assert report.metadata['observation_mode'] == 'axtree_screenshot'
    prediction_files = list(tmp_path.glob('predictions/**/*.jsonl'))
    review_files = list(tmp_path.glob('reviews/**/*.jsonl'))
    report_files = list(tmp_path.glob('reports/**/*.json'))
    artifact_dirs = list(tmp_path.glob('artifacts/miniwob/*'))
    assert prediction_files and review_files and report_files and artifact_dirs

    review = json.loads(review_files[0].read_text().splitlines()[0])
    saved_report = json.loads(report_files[0].read_text())
    assert review['agent_trace']['task_environment'] == 'openenv'
    assert review['agent_trace']['task_environment_runtime'] == 'ms_enclave_docker'
    assert review['agent_trace']['environment'] is None
    assert saved_report['metadata']['official_browsergym_action_config'] is True
    assert saved_report['metadata']['repeats'] == 1
    assert saved_report['metadata']['official_browsergym_evaluation_protocol'] is False
