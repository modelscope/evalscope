"""Glue between :class:`DefaultDataAdapter` and the bridge / runner stack.

The adapter layer is intentionally a single function (no wrapper class) so
that the branch added to ``DefaultDataAdapter._on_inference`` stays narrow
and every benchmark gets external-agent support for free.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from evalscope.api.agent import AgentEnvironment, AgentTrace
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.model import Model, ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.registry import get_environment
from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.logger import get_logger
from .bridge import ModelProxyServer
from .config import ExternalAgentConfig
from .runners import ExternalAgentTask, get_runner

if TYPE_CHECKING:
    from evalscope.api.dataset import Sample

logger = get_logger()


def run_external_agent(
    config: ExternalAgentConfig,
    model: Model,
    sample: 'Sample',
) -> InferenceResult:
    """Synchronously drive one sample through an external agent runner.

    Returns an :class:`InferenceResult` whose ``output`` text is the
    agent's final stdout, ``messages`` is the bridge-reconstructed
    transcript, and ``trace`` is the shared :class:`AgentTrace` (same
    shape as native AgentLoop runs, distinguished by ``framework``).

    Uses :class:`AsyncioLoopRunner` to submit the coroutine to the calling
    thread's long-lived background loop.  That loop is reused across
    samples so the :class:`ModelProxyServer` singleton (which binds to it)
    only spins up once per worker thread instead of once per sample.
    """
    instruction = _instruction_from_sample(sample)
    return AsyncioLoopRunner.run(_run_async(config, model, sample, instruction))


async def _run_async(
    config: ExternalAgentConfig,
    model: Model,
    sample: 'Sample',
    instruction: str,
) -> InferenceResult:
    proxy = await ModelProxyServer.get_or_start(
        host=config.bridge.proxy_host,
        port=config.bridge.proxy_port,
    )
    runner_cls = get_runner(config.framework)
    runner_kwargs = dict(config.kwargs)
    runner_kwargs.setdefault('model_name', getattr(model, 'name', '') or '')
    runner = runner_cls(**runner_kwargs)

    env_cls = get_environment(config.environment)

    async with proxy.trial_session(
        model=model,
        framework=config.framework,
        bridge_config=config.bridge,
    ) as session:
        env: AgentEnvironment = env_cls(**config.environment_extra)
        run_started = time.monotonic()
        run_error: Optional[str] = None
        run_returncode = 0
        try:
            await env.__aenter__()
            await runner.setup(env)
            task = ExternalAgentTask(
                instruction=instruction,
                cwd=None,
                timeout=config.timeout,
                metadata={'sample_id': getattr(sample, 'id', None)},
            )
            session.recorder.record_run_start(
                framework=config.framework,
                cmd_summary=runner_cls.__name__,
                env_summary=[],
                cwd=task.cwd,
            )
            try:
                result = await runner.run(task=task, env=env, bridge=session.endpoint_view())
            except Exception as exc:
                run_error = repr(exc)
                run_returncode = -1
                raise
            else:
                run_returncode = int(result.metrics.get('returncode', 0))
        finally:
            session.recorder.record_run_end(
                returncode=run_returncode,
                timed_out=False,
                wall_time=time.monotonic() - run_started,
                error=run_error,
            )
            await env.__aexit__(None, None, None)

        trace: AgentTrace = session.recorder.snapshot()
        trace.environment = config.environment
        messages = session.recorder.messages()

    output = _to_model_output(result.output, model_name=getattr(model, 'name', '') or '')
    return InferenceResult(output=output, messages=messages, trace=trace)


def _instruction_from_sample(sample: 'Sample') -> str:
    """Render the sample input as a single instruction string."""
    inp = sample.input
    if isinstance(inp, str):
        return inp
    parts = []
    for msg in inp or []:
        if isinstance(msg, (ChatMessageSystem, ChatMessageUser)):
            parts.append(msg.text)
        else:
            parts.append(getattr(msg, 'text', '') or '')
    return '\n\n'.join(p for p in parts if p)


def _to_model_output(text: str, *, model_name: str) -> ModelOutput:
    """Wrap the agent's final answer in a single-choice ModelOutput."""
    choice = ChatCompletionChoice.from_content(text)
    meta: Dict[str, Any] = {'source': 'external_agent'}
    return ModelOutput(model=model_name, choices=[choice], metadata=meta)
