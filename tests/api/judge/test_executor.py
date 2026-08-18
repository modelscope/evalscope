"""JudgeExecutor drives every judge call; these tests use a scripted judge, never a network.

The behaviours locked here are the ones the old per-adapter code got wrong: a failed judge
becoming a 0, one side of a position swap being enough, and parse retries being unbounded.
"""
import pytest
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional, Sequence

from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeExecutor,
    JudgeExecutorConfig,
    JudgeRequest,
    JudgeReview,
    OutputContract,
    Placement,
    ReducedVerdict,
)
from evalscope.api.judge.executor import MAX_STAGES
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.constants import ScoreStatus
from evalscope.metrics.judge.llm_judge import JUDGE_ERROR_PREFIX

ERROR_RESPONSE = f'{JUDGE_ERROR_PREFIX} connection refused'
YES_REPLY = '{"verdict": "yes"}'
NO_REPLY = '{"verdict": "no"}'


class Verdict(BaseModel):
    verdict: Literal['yes', 'no']


YES_NO = OutputContract(schema_model=Verdict, parse_retries=0)


class ScriptedJudge:
    """Returns queued responses in order; the last one repeats once the queue is exhausted."""

    def __init__(self, responses: Sequence[str], model_id: str = 'scripted-judge') -> None:
        self.responses = list(responses)
        self.model_id = model_id
        self.calls: List[List[Any]] = []

    def judge(self, prompt: str = '', system_prompt: Optional[str] = None, messages: Any = None) -> str:
        self.calls.append(messages)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class SimpleAdapter:
    """One yes/no case per sample; ``yes`` scores 1.0."""

    def __init__(
        self,
        contract: OutputContract = YES_NO,
        case_ids: Sequence[str] = ('only', ),
        required: bool = True,
        fallback: Optional[float] = None,
    ) -> None:
        self.contract = contract
        self.case_ids = list(case_ids)
        self.required = required
        self.fallback = fallback

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        return [
            JudgeCase(case_id=case_id, output_contract=self.contract, required=self.required)
            for case_id in self.case_ids
        ]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        return JudgeRequest(messages=[ChatMessageUser(content=f'{case.case_id}/{placement.value}')])

    def expand_judge_cases(self, stage, completed_cases, context) -> List[JudgeCase]:
        return []

    def judge_fallback_verdict(self, case, context) -> Optional[CaseVerdict]:
        if self.fallback is None:
            return None
        return CaseVerdict(case_id=case.case_id, value=Verdict(verdict='yes' if self.fallback >= 0.5 else 'no'))

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        hits = [1.0 if _is_yes(verdict.value) else 0.0 for verdict in case_verdicts]
        return ReducedVerdict(value={'acc': sum(hits) / len(hits)})

    def finalize_judge_score(self, review: JudgeReview, context) -> Score:
        return Score(value=dict(review.value), main_score_name='acc')


class TwoStageAdapter(SimpleAdapter):
    """Stage 0 decides how many stage-1 cases exist, like WideSearch's key alignment."""

    def build_judge_cases(self, context):
        return [JudgeCase(case_id='align', output_contract=self.contract)]

    def expand_judge_cases(self, stage, completed_cases, context):
        if stage != 1:
            return []
        if not any(verdict.case_id == 'align' and _is_yes(verdict.value) for verdict in completed_cases):
            return []
        return [JudgeCase(case_id=f'column-{i}', output_contract=self.contract) for i in range(2)]


def _is_yes(value: Any) -> bool:
    if isinstance(value, list):
        return all(item.verdict == 'yes' for item in value)
    return value.verdict == 'yes'


def make_executor(responses: Sequence[str], **config: Any):
    judge = ScriptedJudge(responses)
    return JudgeExecutor([judge], JudgeExecutorConfig(**config)), judge


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_single_case_produces_a_score():
    executor, judge = make_executor([YES_REPLY])

    review = executor.execute(SimpleAdapter(), {})

    assert review.status is ScoreStatus.SUCCESS
    assert review.value == {'acc': 1.0}
    assert len(judge.calls) == 1
    assert [attempt.status for attempt in review.attempts] == [ScoreStatus.SUCCESS]


def test_raw_response_is_persisted_for_inspection():
    executor, _ = make_executor([YES_REPLY])

    review = executor.execute(SimpleAdapter(), {})

    assert review.attempts[0].raw_response == YES_REPLY
    assert review.attempts[0].latency is not None


def test_save_io_off_drops_the_raw_response():
    executor, _ = make_executor([YES_REPLY], save_io=False)

    review = executor.execute(SimpleAdapter(), {})

    assert review.attempts[0].raw_response is None


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_transport_failure_excludes_the_sample_instead_of_scoring_zero():
    executor, _ = make_executor([ERROR_RESPONSE])

    review = executor.execute(SimpleAdapter(), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.value == {}
    assert review.failure_counts == {'transport_error': 1}


def test_transport_failure_is_not_retried_by_the_executor():
    """The model layer owns network retries; the executor must not multiply them."""
    executor, judge = make_executor([ERROR_RESPONSE], )

    executor.execute(SimpleAdapter(contract=OutputContract(schema_model=Verdict)), {})

    assert len(judge.calls) == 1


def test_parse_failure_excludes_the_sample():
    executor, _ = make_executor(['I am not sure about this one'])

    review = executor.execute(SimpleAdapter(), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.value == {}
    assert review.failure_counts == {'parse_error': 1}


def test_parse_retries_are_bounded_by_the_contract():
    contract = OutputContract(schema_model=Verdict, parse_retries=2)
    executor, judge = make_executor(['unparseable'])

    review = executor.execute(SimpleAdapter(contract=contract), {})

    assert len(judge.calls) == 3
    assert review.failure_counts == {'parse_error': 3}
    assert [attempt.attempt_index for attempt in review.attempts] == [0, 1, 2]


def test_zero_parse_retries_means_one_attempt():
    executor, judge = make_executor(['unparseable'])

    executor.execute(SimpleAdapter(), {})

    assert len(judge.calls) == 1


def test_a_retry_that_parses_is_used():
    contract = OutputContract(schema_model=Verdict, parse_retries=1)
    executor, judge = make_executor(['unparseable', YES_REPLY])

    review = executor.execute(SimpleAdapter(contract=contract), {})

    assert review.value == {'acc': 1.0}
    assert len(judge.calls) == 2
    assert review.failure_counts == {'parse_error': 1}


def test_rule_fallback_fills_in_a_case_the_judge_could_not_answer():
    executor, _ = make_executor([ERROR_RESPONSE])

    review = executor.execute(SimpleAdapter(fallback=1.0), {})

    assert review.status is ScoreStatus.FALLBACK
    assert review.value == {'acc': 1.0}


def test_an_optional_case_failure_does_not_invalidate_the_observation():
    executor, _ = make_executor(['unparseable', YES_REPLY])
    adapter = SimpleAdapter(case_ids=('a', 'b'), required=False)

    review = executor.execute(adapter, {})

    assert review.status is ScoreStatus.SUCCESS
    assert review.value == {'acc': 1.0}


def test_a_required_case_failure_invalidates_the_whole_observation():
    executor, _ = make_executor(['unparseable', YES_REPLY])
    adapter = SimpleAdapter(case_ids=('a', 'b'))

    review = executor.execute(adapter, {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.observations[0].status is ScoreStatus.INVALID_SESSION


# ---------------------------------------------------------------------------
# Position swap
# ---------------------------------------------------------------------------


def test_swap_asks_both_placements():
    executor, judge = make_executor([YES_REPLY], position_swap=True)

    review = executor.execute(SimpleAdapter(), {})

    assert len(judge.calls) == 2
    assert {attempt.placement for attempt in review.attempts} == {Placement.ORIGINAL, Placement.SWAPPED}
    assert review.value == {'acc': 1.0}


def test_one_successful_side_of_a_swap_is_not_a_verdict():
    """AIR-Bench used to emit a score from a single parsed pass; a swap is atomic."""
    executor, _ = make_executor([YES_REPLY, 'unparseable'], position_swap=True)

    review = executor.execute(SimpleAdapter(), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.value == {}


def test_swap_records_both_parsed_placements():
    executor, _ = make_executor([YES_REPLY], position_swap=True)

    review = executor.execute(SimpleAdapter(), {})
    verdict = review.observations[0].case_verdicts[0]

    assert {name: value.verdict for name, value in verdict.placements.items()} == {'original': 'yes', 'swapped': 'yes'}


# ---------------------------------------------------------------------------
# Stage expansion
# ---------------------------------------------------------------------------


def test_stage_one_cases_are_derived_from_stage_zero():
    executor, judge = make_executor([YES_REPLY])

    review = executor.execute(TwoStageAdapter(), {})

    assert [call[0].content for call in judge.calls] == [
        'align/original',
        'column-0/original',
        'column-1/original',
    ]
    assert len(review.observations[0].case_verdicts) == 3


def test_stage_expansion_stops_when_stage_zero_says_so():
    executor, judge = make_executor([NO_REPLY])

    review = executor.execute(TwoStageAdapter(), {})

    assert len(judge.calls) == 1
    assert review.value == {'acc': 0.0}


def test_expansion_that_ends_exactly_at_the_stage_limit_is_not_an_error():
    """Regression: a ``for/else`` guard used to raise whenever the loop ran to completion."""

    class DeepAdapter(SimpleAdapter):

        def build_judge_cases(self, context):
            return [JudgeCase(case_id='s0', output_contract=self.contract)]

        def expand_judge_cases(self, stage, completed_cases, context):
            if stage >= MAX_STAGES:
                return []
            return [JudgeCase(case_id=f's{stage}', output_contract=self.contract)]

    executor, judge = make_executor([YES_REPLY])

    review = executor.execute(DeepAdapter(), {})

    assert len(judge.calls) == MAX_STAGES
    assert review.value == {'acc': 1.0}


def test_expansion_beyond_the_stage_limit_is_an_adapter_bug():
    class RunawayAdapter(SimpleAdapter):

        def expand_judge_cases(self, stage, completed_cases, context):
            return [JudgeCase(case_id=f'endless-{stage}', output_contract=self.contract)]

    executor, _ = make_executor([YES_REPLY])

    with pytest.raises(RuntimeError, match='exceeded'):
        executor.execute(RunawayAdapter(), {})


# ---------------------------------------------------------------------------
# Batch entry point and reserved capabilities
# ---------------------------------------------------------------------------


def test_execute_batch_scores_every_context():
    executor, _ = make_executor([YES_REPLY])

    reviews = executor.execute_batch(SimpleAdapter(), [{}, {}, {}], max_workers=2)

    assert [review.value for review in reviews] == [{'acc': 1.0}] * 3


def test_execute_batch_on_no_contexts():
    executor, _ = make_executor([YES_REPLY])

    assert executor.execute_batch(SimpleAdapter(), []) == []


def test_multiple_judges_are_rejected_until_supported():
    with pytest.raises(ValueError, match='Multiple judge models'):
        JudgeExecutor([ScriptedJudge(['yes']), ScriptedJudge(['yes'])])


def test_repeats_are_rejected_until_supported():
    with pytest.raises(ValueError, match='repeats'):
        JudgeExecutor([ScriptedJudge(['yes'])], JudgeExecutorConfig(repeats=2))


def test_no_judge_is_a_configuration_error():
    with pytest.raises(ValueError, match='at least one judge'):
        JudgeExecutor([])


# ---------------------------------------------------------------------------
# Score construction
# ---------------------------------------------------------------------------


def test_build_score_attaches_status_and_diagnostics():
    executor, _ = make_executor([YES_REPLY])
    adapter = SimpleAdapter()
    review = executor.execute(adapter, {})

    score = executor.build_score(adapter, review, {})

    assert score.status is ScoreStatus.SUCCESS
    assert score.judge_detail.judge_models == ['scripted-judge']
    assert score.judge_detail.valid_observations == 1
    assert score.judge_detail.total_observations == 1


def test_build_score_reports_an_unavailable_sample():
    executor, _ = make_executor([ERROR_RESPONSE])
    adapter = SimpleAdapter()
    review = executor.execute(adapter, {})

    score = executor.build_score(adapter, review, {})

    assert score.status is ScoreStatus.EXCLUDED
    assert not score.status.is_usable
    assert score.value == {}
    assert score.judge_detail.failures == {'transport_error': 1}


def test_build_score_drops_values_an_adapter_invents_for_an_unusable_review():
    """The invariant is enforced, not merely documented: a failed judge is never a number."""

    class SloppyAdapter(SimpleAdapter):

        def finalize_judge_score(self, review, context) -> Score:
            return Score(value={'acc': 0.0}, main_score_name='acc')

    executor, _ = make_executor([ERROR_RESPONSE])
    adapter = SloppyAdapter()
    review = executor.execute(adapter, {})

    score = executor.build_score(adapter, review, {})

    assert score.value == {}


def test_an_adapter_declaring_no_cases_is_scored_not_excluded():
    """Rule-first adapters settle some samples without asking the judge anything."""

    class RuleOnlyAdapter(SimpleAdapter):

        def build_judge_cases(self, context):
            return []

        def reduce_judge_verdicts(self, case_verdicts, context):
            assert case_verdicts == []
            return ReducedVerdict(value={'acc': 1.0})

    executor, judge = make_executor([YES_REPLY])

    review = executor.execute(RuleOnlyAdapter(), {})

    assert judge.calls == []
    assert review.value == {'acc': 1.0}
    assert review.status is ScoreStatus.SUCCESS


def test_declared_cases_that_all_fail_are_still_excluded():
    executor, _ = make_executor(['unparseable'])

    review = executor.execute(SimpleAdapter(required=False), {})

    assert review.value == {}
    assert review.status is ScoreStatus.EXCLUDED
