"""Task environments, persisted run data and deterministic scoring for AgentDojo.

Persisted data:

- ``AgentDojoSampleMeta`` under ``Sample.metadata['agentdojo']``: sample identity, attack and injection map.
- ``AgentDojoRunRecord`` under ``TaskState.metadata['agentdojo_run']``: final output, termination, the tool call
  trace and the final environment, so a sample is scored without the model and without re-executing tools.
- ``ScoreMetadata`` in ``Score.metadata``.

Scoring calls ``TaskSuite._check_task_result`` with the same inputs as upstream ``run_task_with_pipeline``. The
upstream package is imported inside functions only.
"""

import collections.abc
import functools
import types
import typing
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, EmailStr, TypeAdapter, ValidationError

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


def _is_subclass(annotation: Any, base: type) -> bool:
    # Python 3.10 reports parametrized generics such as ``list[int]`` as types.
    return isinstance(annotation, type) and typing.get_origin(annotation) is None and issubclass(annotation, base)


def _is_model(annotation: Any) -> bool:
    return _is_subclass(annotation, BaseModel)


def _is_string(annotation: Any) -> bool:
    return annotation is EmailStr or (_is_subclass(annotation, str) and not _is_subclass(annotation, Enum))


def _restore(annotation: Any, value: Any) -> Any:
    """Rebuild ``value`` from its JSON form as ``annotation`` without validating string values."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if annotation is Any:
        return value
    if origin is Annotated:
        return _restore(args[0], value)
    if _is_model(annotation):
        return restore_environment(annotation, value)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        # A stable sort tries the model members first for an object and last for anything else.
        prefer_models = isinstance(value, Mapping)
        for member in sorted(args, key=lambda arg: _is_model(arg) != prefer_models):
            try:
                return _restore(member, value)
            except ValueError:
                continue
        raise ValueError(f'{value!r} matches no member of {annotation}.')
    if origin in (list, tuple, set, frozenset):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'{annotation} expects an array, got {type(value).__name__}.')
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(args) != len(value):
                raise ValueError(f'{annotation} expects {len(args)} items, got {len(value)}.')
            return tuple(_restore(arg, item) for arg, item in zip(args, value))
        item_annotation = args[0] if args else Any
        return origin(_restore(item_annotation, item) for item in value)
    if origin in (dict, collections.abc.Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f'{annotation} expects an object, got {type(value).__name__}.')
        key_annotation, item_annotation = args or (Any, Any)
        return {_restore(key_annotation, key): _restore(item_annotation, item) for key, item in value.items()}
    if _is_string(annotation) and isinstance(value, str):
        return value
    # Only JSON-lossy leaves (datetime, date, enums) and other scalars are converted.
    return _type_adapter(annotation).validate_python(value)


def restore_environment(environment_type: Type[BaseModel], data: Any) -> BaseModel:
    """Rebuild a task environment exactly from its ``model_dump(mode='json', round_trip=True)`` output.

    ``model_validate`` would not restore it: upstream ``Inbox``, ``Calendar`` and ``CloudDrive`` rebuild ``emails``,
    ``events`` and ``files`` from their ``initial_*`` fields in an ``after`` validator, which undoes every change of
    the run, and upstream tools such as ``share_file`` write unvalidated strings into ``EmailStr`` fields, which
    validation would normalize or reject. The data is therefore rebuilt along the field annotations: models with
    ``model_construct``, containers and unions element by element, string values as they are, and only the remaining
    leaves (datetime, date, enums, numbers) through their ``TypeAdapter``.

    Raises:
        ValueError: ``data`` does not match ``environment_type``.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f'{environment_type.__name__} expects an object, got {type(data).__name__}.')
    values = {}
    for name, field in environment_type.model_fields.items():
        if name not in data:
            raise ValueError(f'{environment_type.__name__}.{name} is missing.')
        values[name] = _restore(field.annotation, data[name])
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
