"""Adapter-facing declarations for one Native LLM judge review."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Type

from pydantic import BaseModel

from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import Score

from .contracts import OutputContract
from .types import CaseVerdict, JudgeCase, JudgeContext, JudgeRequest, JudgeReview, Placement, ReducedVerdict

RequestBuilder = Callable[[JudgeCase, Placement, Sequence[CaseVerdict], JudgeContext], JudgeRequest]
VerdictReducer = Callable[[Sequence[CaseVerdict], JudgeContext], ReducedVerdict]
CaseExpander = Callable[[int, Sequence[CaseVerdict], JudgeContext], Sequence[JudgeCase]]
FallbackBuilder = Callable[[JudgeCase, JudgeContext], Optional[CaseVerdict]]
ScoreFinalizer = Callable[[Score, JudgeReview, JudgeContext], Score]


@dataclass
class JudgeDefinition:
    """A complete, sample-scoped judge declaration.

    ``labels`` and ``numeric`` cover ordinary single-verdict tasks. ``workflow`` keeps the
    rare multi-case and staged tasks expressible without exposing executor hooks on adapters.
    """

    cases: Sequence[JudgeCase]
    request: RequestBuilder
    reduce: VerdictReducer
    main_score_name: Optional[str] = None
    expand: Optional[CaseExpander] = None
    fallback: Optional[FallbackBuilder] = None
    immediate_score: Optional[Score] = None
    skip_reason: Optional[str] = None
    finalize: Optional[ScoreFinalizer] = None

    def __post_init__(self) -> None:
        """Keep rule short-circuits observable and distinct from judge workflows."""
        if self.immediate_score is not None and not self.skip_reason:
            raise ValueError('A skipped judge definition requires a non-empty skip_reason.')
        if self.immediate_score is None and self.skip_reason is not None:
            raise ValueError('skip_reason is only valid for a skipped judge definition.')

    @classmethod
    def labels(
        cls,
        *,
        prompt: str,
        schema_model: Type[BaseModel],
        scores: Dict[str, Dict[str, float]],
        case_id: str = 'judge',
        system_prompt: Optional[str] = None,
        verdict_field: str = 'verdict',
        fallback_verdict: Optional[str] = None,
        main_score_name: Optional[str] = None,
        finalize: Optional[ScoreFinalizer] = None,
    ) -> 'JudgeDefinition':
        """Declare a one-case label judge with a label-to-metrics mapping."""
        contract = OutputContract(schema_model=schema_model)
        case = JudgeCase(case_id=case_id, output_contract=contract)

        def request_builder(
            current: JudgeCase,
            placement: Placement,
            completed: Sequence[CaseVerdict],
            context: JudgeContext,
        ) -> JudgeRequest:
            messages = []
            if system_prompt:
                messages.append(ChatMessageSystem(content=system_prompt))
            messages.append(ChatMessageUser(content=prompt + current.output_contract.instruction()))
            return JudgeRequest(messages=messages)

        def reducer(verdicts: Sequence[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
            label = str(getattr(verdicts[0].value, verdict_field))
            return ReducedVerdict(value=dict(scores[label]))

        fallback = None
        if fallback_verdict is not None:
            def fallback_builder(current: JudgeCase, context: JudgeContext) -> CaseVerdict:
                return CaseVerdict(
                    case_id=current.case_id,
                    value=schema_model.model_validate({verdict_field: fallback_verdict}),
                )
            fallback = fallback_builder
        return cls(
            cases=[case],
            request=request_builder,
            reduce=reducer,
            fallback=fallback,
            main_score_name=main_score_name,
            finalize=finalize,
        )

    @classmethod
    def numeric(
        cls,
        *,
        prompt: str,
        schema_model: Type[BaseModel],
        metric_name: str = 'acc',
        case_id: str = 'judge',
        system_prompt: Optional[str] = None,
        score_field: str = 'score',
        main_score_name: Optional[str] = None,
        finalize: Optional[ScoreFinalizer] = None,
    ) -> 'JudgeDefinition':
        """Declare a one-case numeric judge that copies a schema field into one metric."""
        contract = OutputContract(schema_model=schema_model)
        case = JudgeCase(case_id=case_id, output_contract=contract)

        def request_builder(
            current: JudgeCase,
            placement: Placement,
            completed: Sequence[CaseVerdict],
            context: JudgeContext,
        ) -> JudgeRequest:
            messages = []
            if system_prompt:
                messages.append(ChatMessageSystem(content=system_prompt))
            messages.append(ChatMessageUser(content=prompt + current.output_contract.instruction()))
            return JudgeRequest(messages=messages)

        def reducer(verdicts: Sequence[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
            return ReducedVerdict(value={metric_name: float(getattr(verdicts[0].value, score_field))})

        return cls(
            cases=[case],
            request=request_builder,
            reduce=reducer,
            main_score_name=main_score_name or metric_name,
            finalize=finalize,
        )

    @classmethod
    def workflow(
        cls,
        *,
        cases: Sequence[JudgeCase],
        request: RequestBuilder,
        reduce: VerdictReducer,
        main_score_name: Optional[str] = None,
        expand: Optional[CaseExpander] = None,
        fallback: Optional[FallbackBuilder] = None,
        finalize: Optional[ScoreFinalizer] = None,
    ) -> 'JudgeDefinition':
        """Declare multiple or staged cases with callbacks scoped to this definition."""
        return cls(
            cases=cases,
            request=request,
            reduce=reduce,
            main_score_name=main_score_name,
            expand=expand,
            fallback=fallback,
            finalize=finalize,
        )

    @classmethod
    def skip(cls, score: Score, *, reason: str) -> 'JudgeDefinition':
        """Return a rule-derived score without opening a judge session.

        ``reason`` is recorded in the resulting score metadata so consumers can distinguish a
        deterministic short-circuit from an LLM judge verdict.
        """
        return cls(
            cases=[],
            request=_unreachable_request,
            reduce=_unreachable_reduce,
            immediate_score=score,
            skip_reason=reason,
        )

    # The following methods are executor-facing only.  They deliberately live on the definition,
    # rather than on benchmark adapters, so an adapter exposes one public extension point.
    def build_cases(self, context: JudgeContext) -> Sequence[JudgeCase]:
        """Return the initial cases declared for this review."""
        return self.cases

    def build_request(
        self,
        case: JudgeCase,
        placement: Placement,
        completed_cases: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> JudgeRequest:
        """Render one declared case."""
        return self.request(case, placement, completed_cases, context)

    def expand_cases(
        self,
        stage: int,
        completed_cases: Sequence[CaseVerdict],
        context: JudgeContext,
    ) -> Sequence[JudgeCase]:
        """Return derived cases for the next stage, when configured."""
        return self.expand(stage, completed_cases, context) if self.expand is not None else []

    def fallback_verdict(self, case: JudgeCase, context: JudgeContext) -> Optional[CaseVerdict]:
        """Return the declared rule fallback, when configured."""
        return self.fallback(case, context) if self.fallback is not None else None

    def reduce_verdicts(self, verdicts: Sequence[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
        """Fold parsed case verdicts into this review's metric values."""
        return self.reduce(verdicts, context)


def _unreachable_request(
    case: JudgeCase,
    placement: Placement,
    completed: Sequence[CaseVerdict],
    context: JudgeContext,
) -> JudgeRequest:
    raise RuntimeError('A skipped judge definition cannot build a request.')


def _unreachable_reduce(verdicts: Sequence[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
    raise RuntimeError('A skipped judge definition cannot reduce verdicts.')


__all__ = ['JudgeDefinition']
