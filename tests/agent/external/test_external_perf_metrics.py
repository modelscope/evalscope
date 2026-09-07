"""External-agent runs must keep per-turn ``perf_metrics`` on the transcript.

``DefaultEvaluator._record_perf`` reads ``perf_metrics`` off the assistant
messages in ``TaskState.messages`` and only falls back to ``task_state.output``.
For bridge-driven runs both come from the bridge, so a recorder that drops the
field leaves the perf table empty for every external-agent evaluation while
``collect_perf`` defaults to True.
"""

import pytest

from evalscope.agent.external import ExternalAgentConfig
from evalscope.agent.external.adapter import run_external_agent
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import GenerateConfig, Model, ModelOutput
from evalscope.models.mockllm import MockLLM
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner


@pytest.fixture(autouse=True)
def _release_bridge_loop():
    yield
    AsyncioLoopRunner.shutdown_for_thread()


def test_external_run_keeps_per_turn_perf_metrics_on_the_transcript():
    perf = PerformanceMetrics(latency=0.123, ttft=0.05, input_tokens=10, output_tokens=2)
    output = ModelOutput.from_content(model='mock-model', content='42')
    output.message.perf_metrics = perf
    model = Model(api=MockLLM(model_name='mock-model', custom_outputs=[output]), config=GenerateConfig())

    result = run_external_agent(
        ExternalAgentConfig(framework='mock', environment='local', timeout=60.0),
        model,
        Sample(input='What is 6 * 7?', id=1),
    )

    assistants = [m for m in result.messages if isinstance(m, ChatMessageAssistant)]
    assert [m.perf_metrics for m in assistants] == [perf]
    assert result.output.completion == '42'
