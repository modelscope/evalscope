"""Sample sessions, persisted run data, deterministic scoring and trace replay for AgentDojo.

Persisted data:

- ``AgentDojoSampleMeta`` under ``Sample.metadata['agentdojo']``: sample identity, attack and injection map.
- ``AgentDojoRunRecord`` under ``TaskState.metadata['agentdojo_run']``: final output, termination and the tool call
  trace needed to rebuild the post-run environment without the model.
- ``ScoreMetadata`` in ``Score.metadata``.

Scoring calls ``TaskSuite._check_task_result`` with the same inputs as upstream ``run_task_with_pipeline``. The
upstream package is imported inside functions only.
"""

import copy
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, ValidationError

from evalscope.api.agent.constants import LoopMessages
from evalscope.api.agent.trace import EventType
from evalscope.api.agent.types import AgentLoopResult
from evalscope.benchmarks.agentdojo.constants import (
    BENCHMARK_VERSION,
    RUN_RECORD_META_KEY,
    SAMPLE_META_KEY,
    UPSTREAM_VERSION,
)
from evalscope.benchmarks.agentdojo.tool_binding import AgentDojoToolBinding, ToolCallTraceItem
from evalscope.benchmarks.agentdojo.upstream import UpstreamContext


class EnvironmentCreationError(RuntimeError):
    """A sample's task environment cannot be created."""


class ReplayPayloadError(ValueError):
    """Persisted replay data is missing, malformed or has a version mismatch; ``fields`` lists the dotted names."""

    def __init__(self, fields: Sequence[str], detail: str = '') -> None:
        self.fields: Tuple[str, ...] = tuple(fields)
        message = f'Invalid AgentDojo replay payload; missing or inconsistent fields: {list(self.fields)}'
        super().__init__(f'{message}. {detail}' if detail else message)


class ReplayMismatchError(RuntimeError):
    """Replaying a trace diverged; ``step`` is the 0-based index of the first diverging item."""

    def __init__(self, step: int, reason: str = 'observation mismatch') -> None:
        self.step = step
        self.reason = reason
        super().__init__(f'AgentDojo trace replay diverged at step {step}: {reason}')


class TerminationReason(str, Enum):
    """Why the agent loop of one sample ended."""

    COMPLETED = 'completed'
    MAX_STEPS = 'max_steps'
    CONTEXT_LENGTH_EXCEEDED = 'context_length_exceeded'
    CONTENT_FILTER = 'content_filter'


class AgentDojoSampleMeta(BaseModel):
    """Per-sample identity persisted under ``Sample.metadata['agentdojo']``."""

    model_config = ConfigDict(extra='forbid')

    row_id: str
    suite: str
    user_task_id: str
    injection_task_id: Optional[str]
    """``None`` for benign runs."""

    attack: str
    victim_model_name: str
    effective_model_name: str
    """Value substituted for the ``{model}`` placeholder of the attack templates."""

    injections: Dict[str, str]
    """Injection map written into the environment; ``{}`` for benign runs."""

    benchmark_version: Literal[BENCHMARK_VERSION]
    upstream_version: Literal[UPSTREAM_VERSION]


class AgentDojoRunRecord(BaseModel):
    """Inference outcome persisted under ``TaskState.metadata['agentdojo_run']``."""

    model_config = ConfigDict(use_enum_values=True)

    final_output: str
    termination_reason: TerminationReason
    executed_steps: int
    """Number of model generations of the agent loop."""

    empty_output: bool
    tool_call_trace: List[ToolCallTraceItem]


def _validate_part(
    model: type, key: str, raw: Mapping[str, Any], bad_fields: List[str], details: List[str]
) -> Optional[BaseModel]:
    try:
        return model.model_validate(dict(raw))
    except ValidationError as exc:
        for error in exc.errors():
            loc = '.'.join(str(part) for part in error['loc'])
            bad_fields.append(f'{key}.{loc}' if loc else key)
            details.append(f'{key}.{loc}: {error["msg"]}')
        return None


class ReplayPayload(BaseModel):
    """Everything needed to score a sample without in-memory state."""

    meta: AgentDojoSampleMeta
    run: AgentDojoRunRecord

    @classmethod
    def from_metadata(cls, metadata: Optional[Mapping[str, Any]]) -> 'ReplayPayload':
        """Assemble the payload from ``TaskState.metadata``.

        Raises:
            ReplayPayloadError: A key or field is missing or malformed, or a persisted version differs from the
                pinned one.
        """
        metadata = metadata or {}
        raw_meta = metadata.get(SAMPLE_META_KEY)
        raw_run = metadata.get(RUN_RECORD_META_KEY)
        bad_fields = [
            key
            for key, raw in ((SAMPLE_META_KEY, raw_meta), (RUN_RECORD_META_KEY, raw_run))
            if not isinstance(raw, Mapping)
        ]
        if bad_fields:
            raise ReplayPayloadError(bad_fields, 'Replay information is absent (is save_metadata disabled?).')

        details: List[str] = []
        meta = _validate_part(AgentDojoSampleMeta, SAMPLE_META_KEY, raw_meta, bad_fields, details)
        run = _validate_part(AgentDojoRunRecord, RUN_RECORD_META_KEY, raw_run, bad_fields, details)
        if bad_fields:
            raise ReplayPayloadError(list(dict.fromkeys(bad_fields)), '; '.join(details))
        return cls(meta=meta, run=run)


class ScoreMetadata(BaseModel):
    """Per-sample ``Score.metadata``."""

    model_config = ConfigDict(use_enum_values=True)

    suite: str
    user_task_id: str
    injection_task_id: Optional[str]
    attack: str
    tool_call_trace_length: int
    termination_reason: TerminationReason
    executed_steps: int
    empty_output: bool
    replay_mismatch: bool = False
    replay_mismatch_step: Optional[int] = None
    excluded_reason: Optional[str] = None


def map_termination(result: AgentLoopResult) -> Tuple[TerminationReason, int]:
    """Return ``(termination_reason, executed_steps)``; steps count the ``MODEL_GENERATE`` trace events."""
    events = result.trace.events
    error_messages = {event.payload.get('message') for event in events if event.type == EventType.ERROR}
    executed_steps = sum(1 for event in events if event.type == EventType.MODEL_GENERATE)
    stop_reason = result.final_output.stop_reason if result.final_output is not None else None
    if LoopMessages.MODEL_CONTEXT_OVERFLOW in error_messages or stop_reason == 'model_length':
        return TerminationReason.CONTEXT_LENGTH_EXCEEDED, executed_steps
    if stop_reason == 'content_filter':
        return TerminationReason.CONTENT_FILTER, executed_steps
    if LoopMessages.MAX_STEPS_EXCEEDED in error_messages:
        return TerminationReason.MAX_STEPS, executed_steps
    return TerminationReason.COMPLETED, executed_steps


class SampleSession(BaseModel):
    """Environments and tool binding of one sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pre_environment: Any
    """Deep copy of the environment taken right after ``init_environment``."""

    environment: Any
    """Environment mutated by the tool calls."""

    binding: AgentDojoToolBinding


def create_session(upstream: UpstreamContext, meta: AgentDojoSampleMeta) -> SampleSession:
    """Build a fresh session: ``load_and_inject_default_environment`` -> ``init_environment`` -> deep copy.

    Every call creates new environment objects, so repeats of one sample never share state.

    Raises:
        EnvironmentCreationError: Any step failed, including injection keys that are not injection vectors.
    """
    try:
        suite = upstream.suites[meta.suite]
        environment = suite.load_and_inject_default_environment(dict(meta.injections))
        environment = suite.get_user_task_by_id(meta.user_task_id).init_environment(environment)
        return SampleSession(
            pre_environment=copy.deepcopy(environment),
            environment=environment,
            binding=AgentDojoToolBinding(suite, environment),
        )
    except Exception as exc:
        raise EnvironmentCreationError(
            f'Failed to create the AgentDojo task environment (suite={meta.suite!r}, '
            f'user_task_id={meta.user_task_id!r}, injection_task_id={meta.injection_task_id or ""!r}): '
            f'{type(exc).__name__}: {exc}'
        ) from exc


def check_results(
    upstream: UpstreamContext,
    meta: AgentDojoSampleMeta,
    final_output: str,
    session: SampleSession,
    trace: Sequence[ToolCallTraceItem],
) -> Tuple[bool, Optional[bool]]:
    """Return ``(utility, targeted_asr)`` from the upstream checks; ``targeted_asr`` is ``None`` for benign runs.

    Exceptions raised by the upstream checks propagate.
    """
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.types import text_content_block_from_string

    suite = upstream.suites[meta.suite]
    model_output = [text_content_block_from_string(final_output)]
    stack = [FunctionCall(function=item.function, args=dict(item.args), id=item.call_id) for item in trace]

    def check(task: Any) -> bool:
        return bool(suite._check_task_result(task, model_output, session.pre_environment, session.environment, stack))

    utility = check(suite.get_user_task_by_id(meta.user_task_id))
    if meta.injection_task_id is None:
        return utility, None
    return utility, check(suite.get_injection_task_by_id(meta.injection_task_id))


def replay_trace(
    upstream: UpstreamContext, meta: AgentDojoSampleMeta, trace: Sequence[ToolCallTraceItem]
) -> SampleSession:
    """Rebuild the environment and replay every trace item in order; return the replayed session.

    Valid and invalid items are both replayed through the tool binding, so recorded errors are reproduced.

    Raises:
        EnvironmentCreationError: The environment could not be rebuilt.
        ReplayMismatchError: A replayed observation differs from the recorded one, or replaying an item raised.
    """
    session = create_session(upstream, meta)
    for step, item in enumerate(trace):
        try:
            observation = session.binding.replay(item)
        except Exception as exc:
            raise ReplayMismatchError(step, f'replay raised {type(exc).__name__}: {exc}') from exc
        if observation != item.observation:
            raise ReplayMismatchError(step)
    return session
