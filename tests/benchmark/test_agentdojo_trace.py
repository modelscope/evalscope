"""AgentDojo additions for trace analysis: ``runtime_error`` and the signal provider.

Runs reuse the offline ``agentdojo_scripted`` model of ``test_agentdojo``; nothing touches the network.
"""

from pathlib import Path
from typing import Any

from evalscope.analysis.trace import TraceAnalysisConfig, analyze_records
from evalscope.analysis.trace.analyzer import clear_provider_cache
from evalscope.api.analysis import Attribution, FindingCategory, SampleRef
from evalscope.api.evaluator.cache import ReviewResult
from evalscope.api.registry import get_trace_signal_provider
from evalscope.benchmarks.agentdojo.constants import DEFAULT_ATTACK, RUN_RECORD_META_KEY, SAMPLE_META_KEY
from evalscope.benchmarks.agentdojo.trace_signals import SIDE_EFFECT_TOOLS, AgentDojoTraceSignals
from evalscope.constants import ScoreStatus
from evalscope.run import run_task
from tests.benchmark.test_agentdojo import (
    INJECTION_TASK_ID,
    SUITE,
    USER_TASK_ID,
    VICTIM_MODEL_NAME,
    _construct,
    _no_upstream_access,  # noqa: F401 - fixture
    _reviews,
    _run,
    _score,
    _user_then_injection_turns,
    ground_truth_turns,
    installed_scripts,
    make_task_config,
    requires_upstream,
    text_turn,
    tool_calls_turn,
)


def _analyze_persisted(path: Path) -> list:
    rows = list(_reviews(path).values())
    return analyze_records(
        [
            (
                SampleRef(run_dir=str(path), model='scripted', benchmark='agentdojo', subset=SUITE, index=row['index']),
                ReviewResult.from_cache_item(row),
            )
            for row in rows
        ]
    )


@requires_upstream
class TestProvider:
    def test_side_effect_lists_match_upstream_tools(self) -> None:
        from evalscope.benchmarks.agentdojo.upstream import load_upstream

        upstream = load_upstream()
        assert set(SIDE_EFFECT_TOOLS) == set(upstream.suites)
        for suite, tools in SIDE_EFFECT_TOOLS.items():
            assert set(tools) <= {tool.name for tool in upstream.suites[suite].tools}, suite
        assert get_trace_signal_provider('agentdojo') is AgentDojoTraceSignals

    def test_tool_failures_and_outcome_from_a_scripted_run(self, tmp_path: Path) -> None:
        clear_provider_cache()
        script = [
            tool_calls_turn(('nonexistent_tool', {'x': 1})),
            tool_calls_turn(('send_money', {})),
            *ground_truth_turns(SUITE, USER_TASK_ID),
        ]
        _run({(SUITE, USER_TASK_ID): script}, tmp_path, user_task_ids=[USER_TASK_ID])
        trace = _reviews(tmp_path)[USER_TASK_ID]['sample_score']['sample_metadata'][RUN_RECORD_META_KEY][
            'tool_call_trace'
        ]
        assert [item['runtime_error'] for item in trace[:2]] == [False, True]
        assert not any(item['runtime_error'] for item in trace[2:])

        [case] = _analyze_persisted(tmp_path)
        failures = {f.subtype: f for f in case.findings if f.category == FindingCategory.TOOL_FAILURE}
        assert set(failures) == {'provider:invalid_function', 'provider:runtime_error'}
        assert failures['provider:invalid_function'].rule_id == 'model.unknown_tool'
        assert failures['provider:runtime_error'].attribution == Attribution.MODEL
        assert case.labels['suite'] == SUITE and case.providers == ['agentdojo']
        assert case.outcome.task_success is True and case.outcome.source == 'provider'
        assert not case.failed

    def test_attack_goal_call_is_matched(self, tmp_path: Path) -> None:
        clear_provider_cache()
        from evalscope.benchmarks.agentdojo.injections import InjectionPlanner
        from evalscope.benchmarks.agentdojo.upstream import load_upstream

        injections = InjectionPlanner(load_upstream(), DEFAULT_ATTACK, VICTIM_MODEL_NAME).injections_for(
            SUITE, USER_TASK_ID, INJECTION_TASK_ID
        )
        turns = _user_then_injection_turns(SUITE, USER_TASK_ID, INJECTION_TASK_ID, injections)
        _run(
            {(SUITE, USER_TASK_ID): turns},
            tmp_path,
            attack=DEFAULT_ATTACK,
            user_task_ids=[USER_TASK_ID],
            injection_task_ids=[INJECTION_TASK_ID],
            victim_model_name=VICTIM_MODEL_NAME,
        )
        [case] = _analyze_persisted(tmp_path)
        goals = [f for f in case.findings if f.category == FindingCategory.ATTACK_GOAL_ACTION]
        assert case.outcome.safety_violation is True and case.failed
        assert goals and goals[0].confidence.value == 'high' and goals[0].data['goal_source'] == INJECTION_TASK_ID
        assert case.precursors.attack_goal_actions == len(goals)

    def test_benign_run_without_attack_has_no_goal_matching(self, tmp_path: Path) -> None:
        clear_provider_cache()
        _run({(SUITE, USER_TASK_ID): [text_turn('I cannot find the bill.')]}, tmp_path, user_task_ids=[USER_TASK_ID])
        [case] = _analyze_persisted(tmp_path)
        assert case.precursors.attack_goal_actions is None
        assert case.outcome.task_success is False
        assert [f.subtype for f in case.findings] == ['benchmark_task_failed']
        assert case.primary_attribution == Attribution.UNKNOWN
