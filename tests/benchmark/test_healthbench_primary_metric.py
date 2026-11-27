"""HealthBench reports its rubric overall score as the primary metric, not one rubric axis."""

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.report.generator import ReportGenerator
from tests.api.judge.helpers import ScriptedJudge, make_state


def test_report_primary_is_overall_score() -> None:
    config = TaskConfig(model='m', datasets=['health_bench'], judge={'strategy': 'llm', 'models': [{'model_id': 'j'}]})
    adapter = get_benchmark('health_bench', config)
    adapter.llm_judge = ScriptedJudge(['{"criteria_met": true}', '{"criteria_met": false}'])
    state = make_state('Rest and fluids should help.', '')
    state.metadata = {
        'prompt': [{'role': 'user', 'content': 'I have a cold. What should I do?'}],
        'rubrics': [
            {'criterion': 'Advises rest and fluids.', 'points': 5, 'tags': ['axis:accuracy']},
            {'criterion': 'Lists warning signs that need care.', 'points': 5, 'tags': ['axis:completeness']},
        ],
    }

    scores = adapter.aggregate_scores([adapter.calculate_metrics(state)])
    report = ReportGenerator.generate_report({'communication': scores}, 'm', adapter)
    metrics = {metric.identity.name: metric for metric in report.metrics}

    assert metrics['overall_score'].score == 0.5
    assert metrics['accuracy'].score == 1.0
    assert report.primary_metric_identity == metrics['overall_score'].identity
