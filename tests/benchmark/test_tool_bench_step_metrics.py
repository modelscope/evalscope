"""ToolBench scores each step with the metrics that apply to it, as the official alpha-UMi scorer
does: Act.EM / F1 over 'call' steps and Rouge-L over 'finish' steps, so a perfect model scores 1."""
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.tool_bench.utils import calculate_metrics
from evalscope.metrics.semantics.legacy_identity import migrate_legacy_identity

TOOLS = [{'name': 'get_place_by_ip_for_spott'}, {'name': 'Finish'}]
CALL = 'Thought: \nAction: get_place_by_ip_for_spott\nAction Input: {\n  "is_id": "200.194.51.97"\n}'
FINISH = (
    'Thought: \nAction: Finish\nAction Input: {\n  "return_type": "give_answer",\n'
    '  "final_answer": "The IP address is located in Mexico City"\n}'
)


def _metrics(step: str) -> dict:
    return calculate_metrics({'target': step, 'predictions': step, 'tools': TOOLS})


def test_call_step_has_no_rouge():
    metrics = _metrics(CALL)
    assert metrics['Act.EM'] == 1 and metrics['F1'] == 1
    assert 'Rouge-L' not in metrics


def test_finish_step_has_no_action_metrics():
    metrics = _metrics(FINISH)
    assert metrics['Rouge-L'] == 1
    assert 'Act.EM' not in metrics and 'F1' not in metrics


def test_emitted_metric_keys_cover_declared_metric_list():
    # Plan.EM was emitted as 'plan_em', so the declared metric never got an aggregate bucket (#2024).
    # Compare canonical identities: registration rewrites 'F1' to 'f1', and 'Act.EM' / 'Plan.EM' differ only by target.
    def identities(names: set) -> set:
        return {
            migrate_legacy_identity(name, 'mean', benchmark_name='tool_bench').model_dump_json() for name in names
        }

    emitted = set(_metrics(CALL)) | set(_metrics(FINISH))
    assert identities(set(BENCHMARK_REGISTRY['tool_bench'].metric_list)) <= identities(emitted)
    assert 'plan_em' not in emitted


def test_evaluation_version():
    assert BENCHMARK_REGISTRY['tool_bench'].evaluation_version == 'v1.2'
