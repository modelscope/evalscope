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
    PairwiseOutcome,
    PairwisePlacementOutcome,
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


class PairwiseAdapter(SimpleAdapter):
    def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
        result = case_verdicts[0].value.verdict
        outcome = PairwiseOutcome(metric_name='win_rate', result=result)
        return ReducedVerdict(value={'win_rate': outcome.score}, outcome=outcome)

    def finalize_judge_score(self, review: JudgeReview, context) -> Score:
        return Score(value=dict(review.value), main_score_name='win_rate')


def make_executor(responses: Sequence[Any], **config: Any):
    judge = ScriptedJudge(responses)
    return JudgeExecutor([judge], JudgeExecutorConfig(**config)), judge


def test_executor_rejects_an_impossible_judge_quorum():
    with pytest.raises(ValueError, match='cannot exceed'):
        JudgeExecutor([ScriptedJudge([YES_REPLY])], JudgeExecutorConfig(min_valid_judges=2))


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
    assert review.valid_observations == []
    assert len(review.fallback_observations) == 1


def test_fallback_does_not_satisfy_judge_quorum():
    executor = JudgeExecutor(
        [ScriptedJudge([ConnectionError('offline')], 'judge-a'), ScriptedJudge([ConnectionError('offline')], 'judge-b')],
        JudgeExecutorConfig(min_valid_judges=2),
    )

    review = executor.execute(SimpleAdapter(fallback=1.0), {})

    assert review.status is ScoreStatus.DEGRADED
    assert review.value == {'acc': 1.0}
    assert review.valid_observations == []
    assert len(review.fallback_observations) == 2


def test_fallback_applies_consistently_when_partial_judges_miss_quorum():
    executor = JudgeExecutor(
        [ScriptedJudge([YES_REPLY], 'judge-a'), ScriptedJudge(['not JSON'], 'judge-b')],
        JudgeExecutorConfig(min_valid_judges=2),
    )

    review = executor.execute(SimpleAdapter(fallback=0.0), {})

    assert review.status is ScoreStatus.DEGRADED
    assert review.value == {'acc': 0.0}
    assert len(review.valid_observations) == 1
    assert len(review.fallback_observations) == 1


def test_swap_never_uses_a_one_sided_fallback():
    executor, judge = make_executor([ConnectionError('offline')], position_swap=True)

    review = executor.execute(SimpleAdapter(fallback=1.0), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.value == {}
    assert len(judge.calls) == 1


def test_empty_reducer_output_never_satisfies_judge_quorum():
    class EmptyAdapter(SimpleAdapter):
        def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
            return ReducedVerdict()

    executor = JudgeExecutor(
        [ScriptedJudge([YES_REPLY], 'judge-a'), ScriptedJudge([YES_REPLY], 'judge-b')],
        JudgeExecutorConfig(min_valid_judges=2),
    )

    review = executor.execute(EmptyAdapter(), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.value == {}


def test_metrics_without_individual_quorum_are_excluded():
    class DisjointMetricAdapter(SimpleAdapter):
        def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
            return ReducedVerdict(
                value={'yes': 1.0} if case_verdicts[0].value.verdict == 'yes' else {'no': 1.0}
            )

    executor = JudgeExecutor(
        [ScriptedJudge([YES_REPLY], 'judge-a'), ScriptedJudge([NO_REPLY], 'judge-b')],
        JudgeExecutorConfig(min_valid_judges=2),
    )

    review = executor.execute(DisjointMetricAdapter(), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.value == {}


def test_outcome_without_a_numeric_metric_is_not_a_usable_score():
    class OutcomeOnlyAdapter(SimpleAdapter):
        def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
            return ReducedVerdict(outcome=PairwiseOutcome(metric_name='win_rate', result='win'))

    executor = JudgeExecutor([ScriptedJudge([YES_REPLY])], JudgeExecutorConfig())

    review = executor.execute(OutcomeOnlyAdapter(), {})

    assert review.status is ScoreStatus.EXCLUDED
    assert review.valid_observations == []


def test_unresolved_metric_tie_does_not_discard_other_metrics():
    class PartialMetricAdapter(SimpleAdapter):
        def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
            verdict = case_verdicts[0].value.verdict
            if verdict == 'yes':
                return ReducedVerdict(value={'settled': 1.0, 'tied': 1.0})
            return ReducedVerdict(value={'settled': 1.0, 'tied': 0.0})

    class PrimaryAdapter(PartialMetricAdapter):
        def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
            return ReducedVerdict(value={'settled': 1.0})

    primary = ScriptedJudge([YES_REPLY], 'primary')
    executor = JudgeExecutor(
        [primary, ScriptedJudge([YES_REPLY], 'judge-a'), ScriptedJudge([NO_REPLY], 'judge-b')],
        JudgeExecutorConfig(aggregation='majority_vote'),
    )
    adapter = PartialMetricAdapter()
    original_reduce = adapter.reduce_judge_verdicts
    calls = 0

    def reduce_by_judge(case_verdicts, context):
        nonlocal calls
        calls += 1
        return PrimaryAdapter().reduce_judge_verdicts(case_verdicts, context) if calls == 1 else original_reduce(
            case_verdicts, context
        )

    adapter.reduce_judge_verdicts = reduce_by_judge
    review = executor.execute(adapter, {})

    assert review.status is ScoreStatus.DEGRADED
    assert review.value == {'settled': 1.0}
    assert review.metadata['metrics_without_primary_tiebreak'] == ['tied']


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


@pytest.mark.parametrize(
    ('responses', 'expected'),
    [([YES_REPLY, NO_REPLY], 1.0), ([NO_REPLY, YES_REPLY], 0.0)],
)
def test_repeat_majority_tie_uses_first_valid_observation_and_degrades(responses, expected):
    executor, _ = make_executor(responses, repeats=2, aggregation='majority_vote')

    review = executor.execute(SimpleAdapter(), {})

    assert review.value == {'acc': expected}
    assert review.status is ScoreStatus.DEGRADED
    assert review.metadata['repeat_tie_broken_by_first_observation'] == {'scripted-judge': ['acc']}


def test_pairwise_cross_judge_tie_uses_primary_semantic_result():
    class PairVerdict(BaseModel):
        verdict: Literal['win', 'loss']

    class Adapter(PairwiseAdapter):
        def build_judge_cases(self, context):
            return [JudgeCase(case_id='only', output_contract=OutputContract(schema_model=PairVerdict))]

    executor = JudgeExecutor(
        [ScriptedJudge(['{"verdict": "win"}'], 'primary'), ScriptedJudge(['{"verdict": "loss"}'], 'secondary')],
        JudgeExecutorConfig(aggregation='majority_vote'),
    )

    review = executor.execute(Adapter(), {})

    assert review.outcome.result == 'win'
    assert review.value['win_rate'] == 0.75
    assert review.status is ScoreStatus.DEGRADED
    assert review.metadata['tie_broken_by_primary'] is True


def test_pairwise_repeat_tie_becomes_a_semantic_draw():
    class PairVerdict(BaseModel):
        verdict: Literal['win', 'loss']

    class Adapter(PairwiseAdapter):
        def build_judge_cases(self, context):
            return [JudgeCase(case_id='only', output_contract=OutputContract(schema_model=PairVerdict))]

    executor = JudgeExecutor(
        [ScriptedJudge(['{"verdict": "win"}', '{"verdict": "loss"}'])],
        JudgeExecutorConfig(repeats=2, aggregation='majority_vote'),
    )

    review = executor.execute(Adapter(), {})

    assert review.outcome.result == 'tie'
    assert review.value['win_rate'] == 0.5
    assert review.status is ScoreStatus.SUCCESS


def test_pairwise_placements_are_aggregated_as_separate_games():
    class PairVerdict(BaseModel):
        verdict: Literal['win', 'loss']

    class Adapter(PairwiseAdapter):
        def build_judge_cases(self, context):
            return [JudgeCase(case_id='only', output_contract=OutputContract(schema_model=PairVerdict))]

        def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
            placements = {
                'original': PairwisePlacementOutcome(result='win', strength='strong'),
                'swapped': PairwisePlacementOutcome(result='loss'),
            }
            return ReducedVerdict(
                value={'win_rate': 0.5},
                outcome=PairwiseOutcome(metric_name='win_rate', result='tie', placements=placements),
            )

    executor = JudgeExecutor([ScriptedJudge(['{"verdict": "win"}', '{"verdict": "loss"}'])],
                             JudgeExecutorConfig(position_swap=True))

    review = executor.execute(Adapter(), {})

    assert review.outcome.result == 'tie'
    assert review.outcome.placements['original'].result == 'win'
    assert review.outcome.placements['original'].strength == 'strong'
    assert review.outcome.placements['swapped'].result == 'loss'


def test_display_metadata_comes_from_primary_first_valid_observation():
    class MetadataAdapter(SimpleAdapter):
        def reduce_judge_verdicts(self, case_verdicts, context) -> ReducedVerdict:
            return ReducedVerdict(
                value={'acc': float(case_verdicts[0].value.verdict == 'yes')},
                metadata={'verdict': case_verdicts[0].value.verdict},
            )

    executor = JudgeExecutor(
        [ScriptedJudge([YES_REPLY], 'primary'), ScriptedJudge([NO_REPLY], 'secondary')],
        JudgeExecutorConfig(),
    )

    review = executor.execute(MetadataAdapter(), {})
    score = executor.build_score(MetadataAdapter(), review, {})

    assert review.metadata == {'verdict': 'yes'}
    assert score.metadata['judge_observation_metadata'][1]['metadata'] == {'verdict': 'no'}


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
