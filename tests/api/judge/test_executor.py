"""JudgeExecutor tests use a scripted transport and never call a provider."""
import pytest
from pydantic import BaseModel
from typing import Any, List, Literal, Optional, Sequence

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
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import ModelOutput
from evalscope.constants import ScoreStatus

YES_REPLY = '{"verdict": "yes"}'
NO_REPLY = '{"verdict": "no"}'


class Verdict(BaseModel):
    verdict: Literal['yes', 'no']


YES_NO = OutputContract(schema_model=Verdict)


class ScriptedJudge:
    def __init__(self, responses: Sequence[Any], judge_id: str = 'scripted-judge') -> None:
        self.responses = list(responses)
        self.judge_id = judge_id
        self.model_id = judge_id
        self.calls: List[List[Any]] = []

    def generate(self, messages: List[Any]) -> ModelOutput:
        self.calls.append(messages)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return ModelOutput.from_content(model=self.model_id, content=response)


class SimpleAdapter:
    def __init__(self, fallback: Optional[float] = None) -> None:
        self.fallback = fallback

    def build_judge_cases(self, context: JudgeContext) -> List[JudgeCase]:
        return [JudgeCase(case_id='only', output_contract=YES_NO)]

    def build_judge_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        return JudgeRequest(messages=[ChatMessageUser(content=f'{case.case_id}/{placement.value}')])

    def expand_judge_cases(self, stage, completed_cases, context) -> List[JudgeCase]:
        return []

    def judge_fallback_verdict(self, case, context) -> Optional[CaseVerdict]:
        if self.fallback is None:
            return None
        return CaseVerdict(case_id=case.case_id, value=Verdict(verdict='yes' if self.fallback else 'no'))

    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        return ReducedVerdict(value={'acc': float(case_verdicts[0].value.verdict == 'yes')})

    def finalize_judge_score(self, review: JudgeReview, context) -> Score:
        return Score(value=dict(review.value), main_score_name='acc')


def make_executor(responses: Sequence[Any], **config: Any):
    judge = ScriptedJudge(responses)
    return JudgeExecutor([judge], JudgeExecutorConfig(**config)), judge


def test_single_request_parses_and_persists_full_io():
    executor, judge = make_executor([YES_REPLY])

    review = executor.execute(SimpleAdapter(), {})

    assert review.status is ScoreStatus.SUCCESS
    assert review.value == {'acc': 1.0}
    assert judge.calls[0][0].content == 'only/original'
    attempt = review.attempts[0]
    assert attempt.messages == judge.calls[0]
    assert attempt.model_output.completion == YES_REPLY
    assert attempt.raw_response == YES_REPLY


def test_parse_failure_is_one_call_without_automatic_correction():
    executor, judge = make_executor(['not JSON', YES_REPLY])

    review = executor.execute(SimpleAdapter(), {})

    assert len(judge.calls) == 1
    assert review.status is ScoreStatus.EXCLUDED
    assert review.failure_counts == {'parse_error': 1}


def test_transport_failure_is_typed_and_not_retried_by_executor():
    executor, judge = make_executor([ConnectionError('offline')])

    review = executor.execute(SimpleAdapter(), {})

    assert len(judge.calls) == 1
    assert review.status is ScoreStatus.EXCLUDED
    assert review.failure_counts == {'transport_error': 1}
    assert review.attempts[0].error.startswith('ConnectionError:')


def test_atomic_swap_excludes_a_half_completed_pair():
    executor, _ = make_executor([YES_REPLY, 'not JSON'], position_swap=True)

    review = executor.execute(SimpleAdapter(), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.value == {}


def test_fallback_is_usable_but_degraded():
    executor, _ = make_executor([ConnectionError('offline')])

    review = executor.execute(SimpleAdapter(fallback=1.0), {})

    assert review.status is ScoreStatus.DEGRADED
    assert review.value == {'acc': 1.0}


@pytest.mark.parametrize(
    ('aggregation', 'expected'),
    [('mean', 0.5), ('median', 0.5), ('majority_vote', 1.0)],
)
def test_cross_judge_aggregation(aggregation, expected):
    executor = JudgeExecutor(
        [ScriptedJudge([YES_REPLY], 'primary'), ScriptedJudge([NO_REPLY], 'secondary')],
        JudgeExecutorConfig(aggregation=aggregation),
    )

    review = executor.execute(SimpleAdapter(), {})

    assert review.value == {'acc': expected}
    assert review.status is (ScoreStatus.DEGRADED if aggregation == 'majority_vote' else ScoreStatus.SUCCESS)


def test_missing_repeat_degrades_without_scoring_it_as_zero():
    executor = JudgeExecutor(
        [ScriptedJudge([YES_REPLY, 'not JSON'], 'judge-a')],
        JudgeExecutorConfig(repeats=2),
    )

    review = executor.execute(SimpleAdapter(), {})

    assert review.value == {'acc': 1.0}
    assert review.status is ScoreStatus.DEGRADED
    assert review.disagreement['numeric']['all_observations']['acc']['range'] == 0.0


def test_minimum_valid_judges_excludes_when_quorum_is_not_met():
    executor = JudgeExecutor(
        [ScriptedJudge([YES_REPLY], 'judge-a'), ScriptedJudge(['not JSON'], 'judge-b')],
        JudgeExecutorConfig(min_valid_judges=2),
    )

    review = executor.execute(SimpleAdapter(), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.value == {}


def test_build_score_never_turns_an_unavailable_review_into_zero():
    executor, _ = make_executor(['not JSON'])
    adapter = SimpleAdapter()

    score = executor.build_score(adapter, executor.execute(adapter, {}), {})

    assert score.status is ScoreStatus.EXCLUDED
    assert score.value == {}
    assert score.metadata['judge_attempts'][0]['raw_response'] == 'not JSON'
