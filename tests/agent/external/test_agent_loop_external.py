"""Test that ``AgentLoopAdapter._on_inference`` routes
``ExternalAgentConfig`` runs through the bridge with the adapter's
per-sample environment + ``_external_extract_prediction`` hook.

This pins the wiring SWE-bench Pro relies on: the adapter's
``build_environment(sample)`` becomes the runner's sandbox, and the
``_external_extract_prediction`` hook fires *inside* that env (so it
can still query the sandbox before close).
"""

import pytest
from typing import Any

from evalscope.agent.environments.local import LocalAgentEnvironment
from evalscope.agent.external import ExternalAgentConfig
from evalscope.api.agent import AgentEnvironment, EventType
from evalscope.api.benchmark.adapters.agent_loop_adapter import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.config import TaskConfig
from evalscope.models.mockllm import MockLLM
from evalscope.utils.function_utils import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def _build_mock_model(text: str) -> Model:
    api = MockLLM(
        model_name='mock-model',
        custom_outputs=[ModelOutput.from_content(model='mock-model', content=text)],
    )
    return Model(api=api, config=GenerateConfig())


class _ProbingAdapter(AgentLoopAdapter):
    """Minimal AgentLoopAdapter subclass that records hook calls.

    Constructed via ``__new__`` so we can side-step the
    ``BenchmarkMeta`` requirements of the real adapter pipeline; the
    branch under test only reads ``self._task_config`` and the
    ``build_*`` / ``_external_extract_prediction`` hooks.
    """

    def __init__(self) -> None:  # noqa: D401 - test scaffold, not API
        # Intentionally unused; AgentLoopAdapter.__init__ pulls from
        # BenchmarkMeta which we don't have. Instances are built via
        # ``__new__`` + manual attribute injection in the tests below.
        raise NotImplementedError('use _ProbingAdapter.build() in tests')

    @classmethod
    def build(cls, task_config: TaskConfig, env: LocalAgentEnvironment) -> '_ProbingAdapter':
        adapter = cls.__new__(cls)
        adapter._task_config = task_config
        adapter.max_steps = AgentLoopAdapter.max_steps_default
        adapter._captured_env = None  # type: ignore[attr-defined]
        adapter._captured_runner_output = None  # type: ignore[attr-defined]
        adapter._provided_env = env  # type: ignore[attr-defined]
        return adapter

    def build_environment(self, sample: Sample):  # type: ignore[override]
        return self._provided_env  # type: ignore[attr-defined]

    async def _external_extract_prediction(  # type: ignore[override]
        self,
        env: AgentEnvironment,
        run_result: Any,
        sample: Sample,
    ) -> str:
        # Capture the env identity to prove the per-sample sandbox flowed
        # through; capture the runner output to prove timing (hook fires
        # after the runner returns).
        self._captured_env = env  # type: ignore[attr-defined]
        self._captured_runner_output = run_result.output  # type: ignore[attr-defined]
        return f'PATCH<{run_result.output}>'


def test_agent_loop_adapter_routes_external_config_through_bridge():
    """When ``agent_config.mode == 'external'`` the AgentLoop branch is
    skipped and the bridge stack runs instead, with the adapter's env
    flowing through and the hook's return value becoming the final text."""
    expected = 'mocked-llm-response'
    model = _build_mock_model(expected)
    sample = Sample(input='ignored — overridden by build_initial_messages', id=7)
    env = LocalAgentEnvironment()

    task_cfg = TaskConfig(
        model='mock_llm',
        agent_config=ExternalAgentConfig(framework='mock', environment='local'),
    )
    adapter = _ProbingAdapter.build(task_cfg, env)

    result: InferenceResult = adapter._on_inference(model, sample)

    # Hook ran with the same env we provided (per-sample sandbox flowed through).
    assert adapter._captured_env is env  # type: ignore[attr-defined]
    # Runner stdout is the mock LLM's text (proves bridge → model → bridge).
    assert adapter._captured_runner_output == expected  # type: ignore[attr-defined]
    # InferenceResult.output text is the *hook's* return, not the runner's stdout.
    assert isinstance(result, InferenceResult)
    assert isinstance(result.output, ModelOutput)
    assert result.output.choices[0].message.text == f'PATCH<{expected}>'

    # Trace records the external framework, not 'native' / 'function_calling'.
    assert result.trace is not None
    assert result.trace.framework == 'mock'
    types = [ev.type for ev in result.trace.events]
    assert EventType.RUN_START in types
    assert EventType.RUN_END in types
    assert EventType.MODEL_GENERATE in types


def test_agent_loop_adapter_ignores_native_agent_config():
    """Sanity check: ``mode='native'`` still falls through to the
    AgentLoop path; the new branch only fires for ExternalAgentConfig.
    The actual loop is exercised by ``test_agent_integration``; here we
    only assert that the external-bridge dispatch did *not* run.
    """
    task_cfg = TaskConfig(
        model='mock_llm',
        agent_config={'mode': 'native', 'strategy': 'function_calling'},
    )
    env = LocalAgentEnvironment()
    adapter = _ProbingAdapter.build(task_cfg, env)

    # Replace ``run_external_agent`` symbol *inside* the adapter module so
    # any accidental dispatch fails loudly. We can't patch the import in
    # agent_loop_adapter because it's a function-local import; instead we
    # assert the hook was never invoked.
    sample = Sample(input='hi', id=1)
    model = _build_mock_model('unused')

    # Native path will fail because we have no strategy registered for an
    # un-initialised adapter, but the failure must NOT be from the
    # external bridge code (i.e. capture state stays untouched).
    with pytest.raises(Exception):
        adapter._on_inference(model, sample)

    assert adapter._captured_env is None  # type: ignore[attr-defined]
    assert adapter._captured_runner_output is None  # type: ignore[attr-defined]
