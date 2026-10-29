"""Task environments, persisted run data and deterministic scoring for AgentDojo.

Persisted data:

- ``AgentDojoSampleMeta`` under ``Sample.metadata['agentdojo']``: sample identity, attack and injection map.
- ``AgentDojoRunRecord`` under ``TaskState.metadata['agentdojo_run']``: final output, termination, the tool call
  trace and the final environment, so a sample is scored without the model and without re-executing tools.
- ``ScoreMetadata`` in ``Score.metadata``.

Scoring calls ``TaskSuite._check_task_result`` with the same inputs as upstream ``run_task_with_pipeline``. The
upstream package is imported inside functions only.
"""

import functools
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Type

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from evalscope.api.agent.constants import LoopMessages
from evalscope.api.agent.trace import EventType
from evalscope.api.agent.types import AgentLoopResult
from evalscope.benchmarks.agentdojo.constants import (
    BENCHMARK_VERSION,
    RUN_RECORD_META_KEY,
    SAMPLE_META_KEY,
    UPSTREAM_VERSION,
)
from evalscope.benchmarks.agentdojo.tool_binding import ToolCallTraceItem
from evalscope.benchmarks.agentdojo.upstream import UpstreamContext


class EnvironmentCreationError(RuntimeError):
    """A sample's task environment cannot be created."""


class ScoringPayloadError(ValueError):
    """Persisted scoring data is missing, malformed or inconsistent; ``fields`` lists the dotted names."""

    def __init__(self, fields: Sequence[str], detail: str = '') -> None:
        self.fields: Tuple[str, ...] = tuple(fields)
        message = f'Invalid AgentDojo scoring payload; missing or inconsistent fields: {list(self.fields)}'
        super().__init__(f'{message}. {detail}' if detail else message)


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
    post_environment: Dict[str, Any]
    """Final task environment, ``environment.model_dump(mode='json', round_trip=True)``."""


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


@functools.lru_cache(maxsize=None)
def _type_adapter(annotation: Any) -> TypeAdapter:
    return TypeAdapter(annotation)


def restore_environment(environment_type: Type[BaseModel], data: Any) -> BaseModel:
    """Rebuild a task environment from its ``model_dump(mode='json', round_trip=True)`` output.

    ``model_validate`` would not restore it: upstream ``Inbox``, ``Calendar`` and ``CloudDrive`` rebuild ``emails``,
    ``events`` and ``files`` from their ``initial_*`` fields in an ``after`` validator, which undoes every change of
    the run. Model-typed fields are therefore rebuilt recursively with ``model_construct``, which skips those
    validators, and every other field is validated against its annotation.

    Raises:
        ValueError: ``data`` does not match ``environment_type``.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f'{environment_type.__name__} expects an object, got {type(data).__name__}.')
    values = {}
    for name, field in environment_type.model_fields.items():
        if name not in data:
            raise ValueError(f'{environment_type.__name__}.{name} is missing.')
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            values[name] = restore_environment(annotation, data[name])
        else:
            values[name] = _type_adapter(annotation).validate_python(data[name])
    return environment_type.model_construct(**values)


class ScoringPayload(BaseModel):
    """Everything needed to score a sample without in-memory state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    meta: AgentDojoSampleMeta
    run: AgentDojoRunRecord
    post_environment: Any
    """``run.post_environment`` restored as the suite's environment type."""

    @classmethod
    def from_metadata(cls, metadata: Optional[Mapping[str, Any]], upstream: UpstreamContext) -> 'ScoringPayload':
        """Assemble the payload from ``TaskState.metadata``.

        Raises:
            ScoringPayloadError: A key or field is missing or malformed, a persisted version differs from the pinned
                one, or the final environment cannot be restored.
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
            raise ScoringPayloadError(bad_fields, 'Scoring information is absent (is save_metadata disabled?).')

        details: List[str] = []
        meta = _validate_part(AgentDojoSampleMeta, SAMPLE_META_KEY, raw_meta, bad_fields, details)
        run = _validate_part(AgentDojoRunRecord, RUN_RECORD_META_KEY, raw_run, bad_fields, details)
        if bad_fields:
            raise ScoringPayloadError(list(dict.fromkeys(bad_fields)), '; '.join(details))
        suite = upstream.suites.get(meta.suite)
        if suite is None:
            raise ScoringPayloadError([f'{SAMPLE_META_KEY}.suite'], f'Unknown suite {meta.suite!r}.')
        try:
            post_environment = restore_environment(suite.environment_type, run.post_environment)
        except ValueError as exc:
            raise ScoringPayloadError([f'{RUN_RECORD_META_KEY}.post_environment'], str(exc)) from exc
        return cls(meta=meta, run=run, post_environment=post_environment)


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


def create_environment(upstream: UpstreamContext, meta: AgentDojoSampleMeta) -> Any:
    """Build a fresh task environment: ``load_and_inject_default_environment`` -> ``init_environment``.

    Every call creates new environment objects, so repeats of one sample never share state. The result only depends
    on ``meta``, so scoring rebuilds the pre-run environment with the same call.

    Raises:
        EnvironmentCreationError: Any step failed, including injection keys that are not injection vectors.
    """
    try:
        suite = upstream.suites[meta.suite]
        environment = suite.load_and_inject_default_environment(dict(meta.injections))
        return suite.get_user_task_by_id(meta.user_task_id).init_environment(environment)
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
    pre_environment: Any,
    post_environment: Any,
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
        return bool(suite._check_task_result(task, model_output, pre_environment, post_environment, stack))

    utility = check(suite.get_user_task_by_id(meta.user_task_id))
    if meta.injection_task_id is None:
        return utility, None
    return utility, check(suite.get_injection_task_by_id(meta.injection_task_id))
