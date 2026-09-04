"""Metric failures must not become successful zero scores."""
from typing import Any, List

import pytest

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage
from evalscope.api.metric import JudgeSummary, Metric, SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.config import TaskConfig
from evalscope.constants import JudgeScoreType, ScoreStatus, ScoringPolicy
from evalscope.metrics.aggregators.aggregators import Mean
from evalscope.metrics.judge.llm_judge import DEFAULT_PROMPT_TEMPLATE, LLMJudge


class FailingMetric(Metric):
    """Simulate a plugin that fails on a particular sample."""

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        if predictions[0] == 'crash':
            raise RuntimeError('metric unavailable')
        return [1.0]


class BrokenConstructorMetric(FailingMetric):
    """Simulate a plugin that cannot initialize."""

    def __init__(self) -> None:
        raise RuntimeError('initialization failed')


def make_adapter(
    monkeypatch: pytest.MonkeyPatch, metrics: List[Any], strategy: str = 'auto'
) -> DefaultDataAdapter:
    """Install local metric doubles without changing the registry."""
    from evalscope.api.benchmark.adapters import default_data_adapter

    original_get_metric = default_data_adapter.get_metric

    def get_metric(name: str) -> Any:
        if name == 'flaky':
            return FailingMetric
        if name == 'broken_init':
            return BrokenConstructorMetric
        return original_get_metric(name)

    monkeypatch.setattr(default_data_adapter, 'get_metric', get_metric)
    return DefaultDataAdapter(
        benchmark_meta=BenchmarkMeta(name='metric_failure', dataset_id='unused', metric_list=metrics),
        task_config=TaskConfig(
            datasets=['metric_failure'], judge={'strategy': strategy, 'models': [{'model_id': 'offline'}]}
        ),
    )


def calculate(adapter: DefaultDataAdapter, prediction: str, sample_id: int = 0) -> SampleScore:
    """Exercise the complete per-sample rule scoring path."""
    state = TaskState(
        model='offline',
        sample=Sample(id=sample_id, input='question', target='answer'),
        output=ModelOutput.from_content('offline', prediction),
        completed=True,
    )
    return adapter.calculate_metrics(state)


@pytest.mark.parametrize('metric', ['flaky', {'flaky': {}}, 'broken_init', 'issue1697_unknown_metric'])
def test_metric_errors_are_unavailable(monkeypatch: pytest.MonkeyPatch, metric: Any) -> None:
    adapter = make_adapter(monkeypatch, [metric])
    score = calculate(adapter, 'crash').score
    metric_name = metric if isinstance(metric, str) else next(iter(metric))

    assert score.value == {}
    assert score.status is ScoreStatus.EXCLUDED
    assert score.metadata[metric_name].startswith('error: ')
    assert score.prediction == 'crash'


@pytest.mark.parametrize('metrics', [['flaky', 'accuracy'], ['accuracy', 'flaky']])
def test_partial_metric_failure_retains_other_scores(
    monkeypatch: pytest.MonkeyPatch, metrics: List[str]
) -> None:
    adapter = make_adapter(monkeypatch, metrics)
    score = calculate(adapter, 'crash').score

    assert score.value == {'accuracy': 0.0}
    assert score.status is ScoreStatus.DEGRADED
    assert score.metadata['flaky'] == 'error: metric unavailable'


def test_failed_metric_is_excluded_only_from_its_own_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = make_adapter(monkeypatch, ['flaky', 'accuracy'])
    samples = [calculate(adapter, 'answer', 0), calculate(adapter, 'crash', 1)]
    results = {result.metric_name: result for result in Mean()(samples)}

    assert results['flaky'].score == 1.0
    assert results['flaky'].num == 1
    assert results['flaky'].ids == [0]
    assert results['accuracy'].score == 0.5
    assert results['accuracy'].num == 2
    assert samples[0].score.status is ScoreStatus.SUCCESS


def test_unavailable_rule_and_judge_stay_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = make_adapter(monkeypatch, ['flaky'])
    rule = calculate(adapter, 'crash').score
    judge = Score(status=ScoreStatus.PARSE_ERROR, judge_summary=JudgeSummary(status=ScoreStatus.PARSE_ERROR))

    for score in [adapter.fallback_to_rule_score(rule, judge), adapter._merge_scores(rule, judge)]:
        assert score.value == {}
        assert not score.status.is_usable
        assert not score.judge_summary.status.is_usable
        assert score.metadata['flaky'] == 'error: metric unavailable'


def test_judge_recovers_failed_rule_using_the_judge_metric_name(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = make_adapter(monkeypatch, ['flaky'])
    rule = calculate(adapter, 'crash').score
    judge = Score(value={'acc': 1.0})

    score = adapter._merge_scores(rule, judge)

    assert score.value == {'acc': 1.0}
    assert score.status is ScoreStatus.SUCCESS
    assert score.metadata['flaky'] == 'error: metric unavailable'


class ScriptedJudge:
    """Exercise the judge executor with a deterministic offline response."""

    score_type = JudgeScoreType.PATTERN
    score_mapping = {'A': 1.0, 'B': 0.0}
    prompt_template = DEFAULT_PROMPT_TEMPLATE
    system_prompt = None
    judge_id = model_id = 'offline'
    build_prompt = LLMJudge.build_prompt

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, messages: List[ChatMessage]) -> ModelOutput:
        """Return the configured reply without contacting an external service."""
        self.calls += 1
        return ModelOutput.from_content('offline', self.reply)


@pytest.mark.parametrize(('prediction', 'surviving_value'), [('answer', 1.0), ('wrong', 0.0)])
def test_recall_does_not_replace_a_failed_primary_metric_with_a_surviving_metric(
    monkeypatch: pytest.MonkeyPatch, prediction: str, surviving_value: float
) -> None:
    adapter = make_adapter(monkeypatch, ['broken_init', 'exact_match'], strategy='llm_recall')
    judge = ScriptedJudge('{"verdict":"A"}')
    adapter.llm_judge = judge

    score = calculate(adapter, prediction).score

    assert judge.calls == 1
    assert score.value == {'acc': 1.0, 'exact_match': surviving_value}
    assert score.status is ScoreStatus.DEGRADED
    assert score.judge_summary.status is ScoreStatus.DEGRADED
    assert score.metadata['broken_init'] == 'error: initialization failed'


@pytest.mark.parametrize(('prediction', 'reply'), [('wrong', '{"verdict":"A"}'), ('answer', '{"verdict":"B"}')])
def test_recall_merges_judge_alias_into_the_existing_metric(
    monkeypatch: pytest.MonkeyPatch, prediction: str, reply: str
) -> None:
    adapter = make_adapter(monkeypatch, ['broken_init', 'accuracy'], strategy='llm_recall')
    judge = ScriptedJudge(reply)
    adapter.llm_judge = judge

    sample_score = calculate(adapter, prediction)

    assert judge.calls == 1
    assert sample_score.score.value == {'accuracy': 1.0}
    assert sample_score.score.main_score_name == 'accuracy'
    assert sample_score.score.status is ScoreStatus.DEGRADED
    aggregated = Mean()([sample_score])
    assert len(aggregated) == 1
    assert aggregated[0].metric_name == 'accuracy'
    assert aggregated[0].score == 1.0
    assert aggregated[0].num == 1


@pytest.mark.parametrize(('prediction', 'judge_calls'), [('answer', 0), ('wrong', 1)])
def test_recall_preserves_secondary_metric_failures(
    monkeypatch: pytest.MonkeyPatch, prediction: str, judge_calls: int
) -> None:
    adapter = make_adapter(monkeypatch, ['accuracy', 'broken_init'], strategy='llm_recall')
    judge = ScriptedJudge('{"verdict":"A"}')
    adapter.llm_judge = judge

    score = calculate(adapter, prediction).score

    assert judge.calls == judge_calls
    assert score.value == {'accuracy': 1.0}
    assert score.status is ScoreStatus.DEGRADED
    if score.judge_summary is not None:
        assert score.judge_summary.status is ScoreStatus.DEGRADED


@pytest.mark.parametrize('strategy', ['llm_recall', 'auto'])
@pytest.mark.parametrize('metrics', [['broken_init', 'accuracy'], ['accuracy', 'broken_init']])
def test_failed_judge_preserves_partial_rule_evidence(
    monkeypatch: pytest.MonkeyPatch, strategy: str, metrics: List[str]
) -> None:
    adapter = make_adapter(monkeypatch, metrics, strategy=strategy)
    adapter.scoring_policy = ScoringPolicy.JUDGE_DEFAULT
    judge = ScriptedJudge('not JSON')
    adapter.llm_judge = judge

    score = calculate(adapter, 'wrong').score

    assert judge.calls == 1
    assert score.value == {'accuracy': 0.0}
    assert score.status is ScoreStatus.DEGRADED
    assert score.judge_summary.status is ScoreStatus.DEGRADED
    assert score.metadata['broken_init'] == 'error: initialization failed'
