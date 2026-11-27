"""Data contracts shared by the trace analyzer and benchmark signal providers.

These models are deliberately benchmark-agnostic: a provider only adds optional,
more precise signals on top of what can be read from any persisted review row.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from evalscope.api.evaluator.cache import ReviewResult


class Attribution(str, Enum):
    """Who is responsible for a finding."""

    MODEL = 'model'
    GUARD = 'guard'
    ENVIRONMENT = 'environment'
    BENCHMARK = 'benchmark'
    UNKNOWN = 'unknown'


class Confidence(str, Enum):
    """How much a finding can be trusted."""

    HIGH = 'high'
    """Read from a structured field or provider ground truth."""

    MEDIUM = 'medium'
    """Derived from a text pattern or a heuristic."""

    LOW = 'low'
    """Weak heuristic; validate before relying on it."""


class FindingCategory(str, Enum):
    """Top-level kind of a finding."""

    TOOL_FAILURE = 'tool_failure'
    REPETITION = 'repetition'
    TERMINATION = 'termination'
    SIDE_EFFECT = 'side_effect'
    """Informational: a side-effect tool call happened."""

    TAINTED_SIDE_EFFECT = 'tainted_side_effect'
    ATTACK_GOAL_ACTION = 'attack_goal_action'
    RISKY_COMMAND = 'risky_command'
    REFUSAL = 'refusal'


INFORMATIONAL_CATEGORIES = frozenset({FindingCategory.SIDE_EFFECT})
"""Categories that never make a sample count as failed on their own."""


class SampleRef(BaseModel):
    """Stable reference to one persisted sample."""

    run_dir: str
    model: str
    benchmark: str
    subset: str
    index: int

    def key(self) -> str:
        """Return a compact, unique key for this sample within an analysis."""
        return f'{self.model}/{self.benchmark}/{self.subset}/{self.index}'


class TraceToolCall(BaseModel):
    """One tool call with its observation, normalized across benchmarks."""

    position: int
    """0-based order of the call within the sample."""

    step: Optional[int] = None
    """AgentLoop step when known."""

    call_event_index: Optional[int] = None
    result_event_index: Optional[int] = None
    """Positions in the ordered agent trace, when the corresponding events are present."""

    message_index: int
    """Index of the assistant message that issued the call; ``-1`` for trace-only calls."""

    call_id: Optional[str] = None
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    parse_error: Optional[str] = None
    result_message_index: Optional[int] = None
    result_text: Optional[str] = None
    error_type: Optional[str] = None
    """``ToolCallError.type`` or a provider / text-derived label."""

    error_message: Optional[str] = None
    error_source: Optional[Literal['structured', 'text_pattern', 'provider']] = None


class LoopEvent(BaseModel):
    """A loop-level event (error, nudge, submit, run start/end) taken from ``AgentTrace``."""

    step: int
    type: str
    source: Optional[str] = None
    message: Optional[str] = None


class Outcome(BaseModel):
    """Resolved outcome of one sample, independent of the raw score shape."""

    task_success: Optional[bool] = None
    safety_violation: Optional[bool] = None
    refused: Optional[bool] = None
    source: Literal['provider', 'binary_score', 'threshold', 'none'] = 'none'


class TraceView(BaseModel):
    """Benchmark-agnostic view of one sample's trajectory."""

    ref: SampleRef
    system_text: str = ''
    """Concatenated system messages (trusted instructions)."""

    user_text: str = ''
    """Concatenated user messages (trusted instructions); tool attachments are excluded."""

    tool_calls: List[TraceToolCall] = Field(default_factory=list)
    events: List[LoopEvent] = Field(default_factory=list)
    assistant_texts: List[str] = Field(default_factory=list)
    """Visible assistant text per assistant message; reasoning parts are excluded."""

    final_answer: Optional[str] = None
    has_nontext_output: bool = False
    score_value: Dict[str, float] = Field(default_factory=dict)
    main_score_name: Optional[str] = None
    score_status: str = 'success'


class Evidence(BaseModel):
    """A short excerpt supporting a finding."""

    message_index: Optional[int] = None
    position: Optional[int] = None
    excerpt: str


class Finding(BaseModel):
    """One problem detected in one sample."""

    category: FindingCategory
    subtype: str
    positions: List[int] = Field(default_factory=list)
    call_ids: List[str] = Field(default_factory=list)
    first_position: Optional[int] = None
    attribution: Attribution = Attribution.UNKNOWN
    rule_id: str = 'fallback'
    confidence: Confidence
    detector: str
    evidence: List[Evidence] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    text: str = ''
    """Text used by attribution rules (error message, observation snippet); not exported verbatim."""


class ExpectedCall(BaseModel):
    """A tool call a provider expects for some goal (e.g. an injection task's ground truth)."""

    function: str
    args: Dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[str] = None


class ToolStatusOverride(BaseModel):
    """Authoritative success / failure flag for one tool call."""

    failed: bool
    error_type: Optional[str] = None
    message: Optional[str] = None


class ProviderSignals(BaseModel):
    """Benchmark-specific signals returned by a :class:`TraceSignalProvider`."""

    labels: Dict[str, Any] = Field(default_factory=dict)
    tool_status: Dict[int, ToolStatusOverride] = Field(default_factory=dict)
    """Keyed by :attr:`TraceToolCall.position`."""

    side_effect_tools: Optional[List[str]] = None
    """Authoritative side-effect tool names; overrides the name heuristics when set."""

    attack_goal_calls: Optional[List[ExpectedCall]] = None
    outcome: Optional[Outcome] = None
    stop_reason: Optional[str] = None
    """Provider-known stop reason (e.g. ``content_filter``, ``model_length``) when the trace lacks it."""

    notes: List[str] = Field(default_factory=list)


class FailureCase(BaseModel):
    """Trace observations consumed by the unified diagnostic pipeline."""

    labels: Dict[str, Any] = Field(default_factory=dict)
    outcome: Outcome
    score_value: Dict[str, float] = Field(default_factory=dict)
    score_status: str = 'success'
    findings: List[Finding] = Field(default_factory=list)
    tool_call_count: int = 0
    notes: List[str] = Field(default_factory=list)


class TraceSignalProvider(ABC):
    """Adds benchmark-specific signals to the generic trace analysis.

    Implementations live next to their benchmark adapter and register with
    :func:`evalscope.api.registry.register_trace_signal_provider`. They must not import optional
    upstream packages at module import time; load them inside :meth:`collect` and report
    degradations through :attr:`ProviderSignals.notes`.
    """

    @abstractmethod
    def collect(self, record: 'ReviewResult', view: TraceView) -> ProviderSignals:
        """Return the signals of one persisted sample.

        Args:
            record: The raw review row, including ``sample_score.sample_metadata``.
            view: The generic, already normalized view of the same sample.
        """


__all__ = [
    'TraceSignalProvider',
    'INFORMATIONAL_CATEGORIES',
    'Attribution',
    'Confidence',
    'Evidence',
    'ExpectedCall',
    'FailureCase',
    'Finding',
    'FindingCategory',
    'LoopEvent',
    'Outcome',
    'ProviderSignals',
    'SampleRef',
    'ToolStatusOverride',
    'TraceToolCall',
    'TraceView',
]
