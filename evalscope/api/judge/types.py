"""Contracts for one judge review: what is asked, what came back, and what the adapter must do."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence

from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage
from evalscope.constants import ScoreStatus
from .contracts import OutputContract

if TYPE_CHECKING:
    from evalscope.api.metric import Score


class Placement(str, Enum):
    """Which side of a pairwise comparison a request presents first."""

    ORIGINAL = 'original'
    SWAPPED = 'swapped'


class JudgeContext(BaseModel):
    """Everything the adapter's judge hooks may read about one sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_state: TaskState
    original_prediction: str
    filtered_prediction: str
    reference: str


class JudgeCase(BaseModel):
    """One thing the judge is asked about a sample: a rubric, a claim, a criterion, a cell."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: str
    """Stable identifier within the sample."""

    output_contract: OutputContract
    """Declares the verdict shape and is the only thing allowed to parse the response."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Adapter-defined data carried through to ``build_judge_request`` and onto the verdict."""


class JudgeRequest(BaseModel):
    """What to send the judge for one case. The adapter decides the content; the executor owns
    the identity (judge, repeat, placement)."""

    messages: List[ChatMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JudgeAttempt(BaseModel):
    """One request/response round trip, including the ones that failed."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: ScoreStatus
    case_id: str
    judge_id: str
    repeat_id: int = 0
    placement: Placement = Placement.ORIGINAL
    attempt_index: int = 0
    """0 for the first try; higher for a contract-declared parse retry."""

    raw_response: Optional[str] = None
    parsed_value: Any = None
    error: Optional[str] = None
    latency: Optional[float] = None


class CaseVerdict(BaseModel):
    """The usable outcome of one case, after any retry and after both placements."""

    case_id: str
    value: Any
    status: ScoreStatus = ScoreStatus.SUCCESS
    placements: Dict[str, Any] = Field(default_factory=dict)
    """Per-placement parsed values, present when the case was judged on both sides."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Copied from the case, so a reduce step reads its context instead of parsing ``case_id``."""


class ReducedVerdict(BaseModel):
    """One observation's worth of verdicts folded into per-metric values by the adapter."""

    value: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JudgeObservation(BaseModel):
    """One (judge, repeat) pass over all of a sample's cases."""

    judge_id: str
    repeat_id: int = 0
    status: ScoreStatus = ScoreStatus.SUCCESS
    case_verdicts: List[CaseVerdict] = Field(default_factory=list)
    reduced: Optional[ReducedVerdict] = None
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status.is_usable and self.reduced is not None


class JudgeReview(BaseModel):
    """Everything the judge produced for one sample."""

    status: ScoreStatus = ScoreStatus.SUCCESS
    observations: List[JudgeObservation] = Field(default_factory=list)
    attempts: List[JudgeAttempt] = Field(default_factory=list)
    value: Dict[str, float] = Field(default_factory=dict)
    """Aggregated per-metric values; empty when no observation was usable."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Diagnostics the adapter's reduce step attached, carried onto ``Score.metadata``."""

    failure_counts: Dict[str, int] = Field(default_factory=dict)
    """Attempt counts keyed by :class:`ScoreStatus` value."""

    error: Optional[str] = None

    @property
    def valid_observations(self) -> List[JudgeObservation]:
        return [obs for obs in self.observations if obs.is_valid]


class JudgeProtocol(Protocol):
    """The adapter-side hooks :class:`JudgeExecutor` drives.

    An adapter declares what to ask and how to fold the answers into a score. It never calls the
    judge model and never parses a response.
    """

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        """Return the cases for one sample. May depend on the prediction, not on judge output."""
        ...

    def build_judge_request(
        self,
        case: JudgeCase,
        placement: Placement,
        completed_cases: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> JudgeRequest:
        """Render one case into messages. ``completed_cases`` holds earlier stages' verdicts."""
        ...

    def expand_judge_cases(
        self,
        stage: int,
        completed_cases: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> List[JudgeCase]:
        """Return cases derived from a finished stage, e.g. per-column checks that depend on an
        alignment produced by stage 0. Return an empty list when nothing is derived."""
        ...

    def judge_fallback_verdict(self, case: JudgeCase, context: JudgeContext) -> Optional[CaseVerdict]:
        """Return a rule-derived verdict for a case the judge could not answer, or ``None``."""
        ...

    def reduce_judge_verdicts(self, case_verdicts: Sequence[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
        """Fold one observation's verdicts into per-metric values."""
        ...

    def finalize_judge_score(self, review: JudgeReview, context: JudgeContext) -> 'Score':
        """Turn the aggregated review into the sample's ``Score``."""
        ...


__all__: Sequence[str] = (
    'CaseVerdict',
    'JudgeAttempt',
    'JudgeCase',
    'JudgeContext',
    'JudgeObservation',
    'JudgeProtocol',
    'JudgeRequest',
    'JudgeReview',
    'Placement',
    'ReducedVerdict',
)
