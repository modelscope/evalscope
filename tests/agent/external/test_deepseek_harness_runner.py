"""Tests for the DeepSeek Harness external runner."""

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import pytest

from evalscope.agent.external import ExternalAgentConfig
from evalscope.agent.external.adapter import run_external_agent
from evalscope.agent.external.runners.base import BridgeEndpoint, ExternalAgentTask, RunnerTimeoutError
from evalscope.agent.external.runners.deepseek_harness import DeepSeekHarnessRunner
from evalscope.api.agent import EventType
from evalscope.api.agent.types import ExecResult
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.models.mockllm import MockLLM
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


_load_env_file()
DASHSCOPE_BASE_URL = os.environ.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
TARGET_MODEL = os.environ.get('EVALSCOPE_QWEN_MODEL', 'qwen-plus')


class FakeEnvironment:

    name = 'fake'

    def __init__(self, results: List[ExecResult]) -> None:
        self._results = list(results)
        self.calls: List[Tuple[List[str], Optional[float], Optional[Dict[str, str]]]] = []

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        del cwd, input
        self.calls.append((cmd, timeout, env))
        return self._results.pop(0)


def _run(coro: Any) -> Any:
    return AsyncioLoopRunner.run(coro)


def _task() -> ExternalAgentTask:
    return ExternalAgentTask(instruction='Reply with 42.', timeout=20.0, metadata={'sample_id': 'sample-1'})


def _bridge() -> BridgeEndpoint:
    return BridgeEndpoint(base_url='http://127.0.0.1:12345', trial_token='trial-secret')


def test_setup_requires_dsh_when_auto_install_disabled() -> None:
    runner = DeepSeekHarnessRunner(auto_install=False)
    env = FakeEnvironment([ExecResult(returncode=1)])

    with pytest.raises(RuntimeError, match='dsh CLI not found'):
        _run(runner.setup(env))


def test_run_writes_isolated_openai_provider_settings() -> None:
    runner = DeepSeekHarnessRunner(model_name='qwen-plus', home_override='/tmp/evalscope-dsh')
    env = FakeEnvironment([ExecResult(), ExecResult(stdout='42\n', duration=1.25)])

    result = _run(runner.run(_task(), env, _bridge()))

    assert result.output == '42'
    assert result.metrics == {'wall_time': 1.25, 'returncode': 0}
    settings_cmd, _, settings_env = env.calls[0]
    settings_text = ' '.join(settings_cmd)
    assert 'openai-completions' in settings_text
    assert 'http://127.0.0.1:12345/openai/v1' in settings_text
    assert 'supportsDeveloperRole' in settings_text
    assert 'maxTokensField' in settings_text
    assert 'agent-default-model' in settings_text
    assert 'trial-secret' not in settings_text
    assert settings_env == {
        'OPENAI_API_KEY': 'trial-secret',
        'DSH_PERMISSION_MODE': 'danger-full-access',
        'DSH_HOME': '/tmp/evalscope-dsh',
    }
    command, timeout, command_env = env.calls[1]
    assert command == ['dsh', '--profile', 'headless', 'Reply with 42.']
    assert timeout == 20.0
    assert command_env == settings_env


def test_run_raises_timeout_error() -> None:
    runner = DeepSeekHarnessRunner(home_override='/tmp/evalscope-dsh')
    env = FakeEnvironment([ExecResult(), ExecResult(returncode=137, timed_out=True)])

    with pytest.raises(RunnerTimeoutError, match='timed out'):
        _run(runner.run(_task(), env, _bridge()))


def test_run_reports_cli_failure() -> None:
    runner = DeepSeekHarnessRunner(home_override='/tmp/evalscope-dsh')
    env = FakeEnvironment([ExecResult(), ExecResult(returncode=1, stderr='bad config')])

    with pytest.raises(RuntimeError, match='bad config'):
        _run(runner.run(_task(), env, _bridge()))


@pytest.fixture(autouse=True)
def _release_bridge_loop() -> None:
    yield
    AsyncioLoopRunner.shutdown_for_thread()


@pytest.mark.skipif(
    os.environ.get('EVALSCOPE_DSH_E2E') != '1' or shutil.which('dsh') is None,
    reason='DSH e2e test; set EVALSCOPE_DSH_E2E=1 with dsh installed',
)
def test_dsh_headless_through_chat_completions_bridge() -> None:
    output = ModelOutput(
        model='mock-dsh',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content='42'), stop_reason='stop')],
    )
    model = Model(api=MockLLM(model_name='mock-dsh', custom_outputs=[output, output]), config=GenerateConfig(max_tokens=64))
    config = ExternalAgentConfig(
        framework='deepseek-harness',
        environment='local',
        timeout=60.0,
        kwargs={'model_name': 'mock-dsh', 'auto_install': False},
    )
    result = run_external_agent(
        config=config,
        model=model,
        sample=Sample(input='What is 6 * 7? Reply with just the number.', id=1),
    )

    assert result.output.message.text.strip() == '42'
    assert result.trace.framework == 'deepseek-harness'
    assert any(event.type == EventType.MODEL_GENERATE for event in result.trace.events)


def _image_exists(name: str) -> bool:
    if shutil.which('docker') is None:
        return False
    try:
        result = subprocess.run(['docker', 'images', '-q', name], capture_output=True, text=True)
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


@pytest.mark.skipif(
    os.environ.get('EVALSCOPE_DSH_DOCKER_E2E') != '1',
    reason='DSH Docker e2e test; set EVALSCOPE_DSH_DOCKER_E2E=1 to enable',
)
@pytest.mark.skipif(
    not _image_exists('evalscope-deepseek-harness:0.1.5-rc.2'),
    reason='evalscope-deepseek-harness:0.1.5-rc.2 image not built',
)
def test_dsh_docker_through_chat_completions_bridge() -> None:
    from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

    output = ModelOutput(
        model='mock-dsh',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content='42'), stop_reason='stop')],
    )
    model = Model(api=MockLLM(model_name='mock-dsh', custom_outputs=[output, output]), config=GenerateConfig(max_tokens=64))
    environment = EnclaveAgentEnvironment(
        engine='docker',
        sandbox_config={
            'image': 'evalscope-deepseek-harness:0.1.5-rc.2',
            'working_dir': '/workspace',
            'network_enabled': True,
        },
        timeout=120.0,
    )
    config = ExternalAgentConfig(
        framework='deepseek-harness',
        environment='enclave',
        timeout=90.0,
        kwargs={'model_name': 'mock-dsh', 'auto_install': False},
    )
    result = run_external_agent(
        config=config,
        model=model,
        sample=Sample(input='What is 6 * 7? Reply with just the number.', id=1),
        environment_override=environment,
    )

    assert result.output.message.text.strip() == '42'
    assert result.trace.framework == 'deepseek-harness'
    assert result.trace.environment == 'enclave'
    assert any(event.type == EventType.MODEL_GENERATE for event in result.trace.events)


@pytest.mark.skipif(
    os.environ.get('EVALSCOPE_REAL_QWEN') != '1' or not DASHSCOPE_API_KEY,
    reason='real API test; set EVALSCOPE_REAL_QWEN=1 and DASHSCOPE_API_KEY',
)
@pytest.mark.skipif(
    os.environ.get('EVALSCOPE_DSH_DOCKER_E2E') != '1',
    reason='DSH Docker e2e test; set EVALSCOPE_DSH_DOCKER_E2E=1 to enable',
)
@pytest.mark.skipif(
    not _image_exists('evalscope-deepseek-harness:0.1.5-rc.2'),
    reason='evalscope-deepseek-harness:0.1.5-rc.2 image not built',
)
def test_dsh_docker_through_real_qwen_bridge() -> None:
    from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

    model = Model(
        api=OpenAICompatibleAPI(
            model_name=TARGET_MODEL,
            base_url=DASHSCOPE_BASE_URL,
            api_key=DASHSCOPE_API_KEY,
        ),
        config=GenerateConfig(max_tokens=256),
    )
    environment = EnclaveAgentEnvironment(
        engine='docker',
        sandbox_config={
            'image': 'evalscope-deepseek-harness:0.1.5-rc.2',
            'working_dir': '/workspace',
            'network_enabled': True,
        },
        timeout=240.0,
    )
    config = ExternalAgentConfig(
        framework='deepseek-harness',
        environment='enclave',
        timeout=180.0,
        kwargs={'model_name': TARGET_MODEL, 'auto_install': False},
    )
    result = run_external_agent(
        config=config,
        model=model,
        sample=Sample(input='What is 6 * 7? Reply with just the number.', id=1),
        environment_override=environment,
    )

    text = result.output.message.text.strip()
    trace = result.trace
    assert '42' in text
    assert trace.framework == 'deepseek-harness'
    assert trace.total_usage is not None
    assert trace.total_usage.total_tokens > 0
    assert any(event.type == EventType.MODEL_GENERATE for event in trace.events)
    print(f'DSH real API result={text!r} usage={trace.total_usage.model_dump()} steps={trace.step_count}')
