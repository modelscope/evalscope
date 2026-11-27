"""AgentDojo signals for the generic trace analyzer (``evalscope analyze``).

Only the evalscope API is imported at module level; the upstream ``agentdojo`` package is
loaded lazily for the injection-task ground truth and is optional.
"""

import importlib.util
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

from evalscope.api.analysis import (
    ExpectedCall,
    Outcome,
    ProviderSignals,
    ToolStatusOverride,
    TraceSignalProvider,
    TraceView,
)
from evalscope.api.registry import register_trace_signal_provider
from evalscope.benchmarks.agentdojo.constants import RUN_RECORD_META_KEY, SAMPLE_META_KEY

# State-changing tools per suite of agentdojo==0.1.35 (checked against the upstream tool list in tests).
# ``get_webpage`` is listed for slack because visiting an attacker URL is the suite's exfiltration channel.
SIDE_EFFECT_TOOLS: Dict[str, Tuple[str, ...]] = {
    'workspace': (
        'add_calendar_event_participants',
        'append_to_file',
        'cancel_calendar_event',
        'create_calendar_event',
        'create_file',
        'delete_email',
        'delete_file',
        'reschedule_calendar_event',
        'send_email',
        'share_file',
    ),
    'travel': (
        'cancel_calendar_event',
        'create_calendar_event',
        'reserve_car_rental',
        'reserve_hotel',
        'reserve_restaurant',
        'send_email',
    ),
    'banking': (
        'schedule_transaction',
        'send_money',
        'update_password',
        'update_scheduled_transaction',
        'update_user_info',
    ),
    'slack': (
        'add_user_to_channel',
        'get_webpage',
        'invite_user_to_slack',
        'post_webpage',
        'remove_user_from_slack',
        'send_channel_message',
        'send_direct_message',
    ),
}

_TERMINATION_STOP_REASONS = {
    'content_filter': 'content_filter',
    'context_length_exceeded': 'model_length',
    'max_steps': 'max_steps',
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


@lru_cache(maxsize=None)
def _goal_calls(meta_json: str) -> Tuple[Tuple[str, str], ...]:
    """Ground-truth calls of the injection task, cached per sample identity (JSON-encoded meta)."""
    import json

    from evalscope.benchmarks.agentdojo.scoring import AgentDojoSampleMeta, create_environment
    from evalscope.benchmarks.agentdojo.upstream import load_upstream

    values = json.loads(meta_json)
    meta = AgentDojoSampleMeta.model_validate(
        {key: values[key] for key in AgentDojoSampleMeta.model_fields if key in values}
    )
    upstream = load_upstream()
    pre_environment = create_environment(upstream, meta)
    task = upstream.suites[meta.suite].get_injection_task_by_id(meta.injection_task_id)
    calls = task.ground_truth(pre_environment)
    return tuple((call.function, json.dumps(_jsonable(dict(call.args)), sort_keys=True)) for call in calls)


@register_trace_signal_provider('agentdojo')
class AgentDojoTraceSignals(TraceSignalProvider):
    """Tool-call status, side-effect tools, injection goals and outcome for AgentDojo samples."""

    def collect(self, record: Any, view: TraceView) -> ProviderSignals:
        import json

        metadata = record.sample_score.sample_metadata or {}
        meta = metadata.get(SAMPLE_META_KEY) or {}
        run = metadata.get(RUN_RECORD_META_KEY) or {}
        signals = ProviderSignals()
        suite = meta.get('suite')
        signals.labels = {
            key: meta.get(key) for key in ('suite', 'user_task_id', 'injection_task_id', 'attack') if key in meta
        }
        if suite in SIDE_EFFECT_TOOLS:
            signals.side_effect_tools = list(SIDE_EFFECT_TOOLS[suite])
        elif suite is not None:
            signals.notes.append(f'agentdojo: unknown suite {suite!r}')

        signals.tool_status = self._tool_status(run.get('tool_call_trace') or [], view)
        signals.stop_reason = _TERMINATION_STOP_REASONS.get(
            str((record.sample_score.score.metadata or {}).get('termination_reason', run.get('termination_reason')))
        )
        signals.outcome = self._outcome(view)

        if meta.get('injection_task_id'):
            if importlib.util.find_spec('agentdojo') is None:
                signals.notes.append('agentdojo: upstream package not installed; attack-goal matching skipped')
            else:
                try:
                    goals = _goal_calls(json.dumps(meta, sort_keys=True))
                    signals.attack_goal_calls = [
                        ExpectedCall(function=function, args=json.loads(args), source_id=meta['injection_task_id'])
                        for function, args in goals
                    ]
                except Exception as exc:  # noqa: BLE001 - degrade to generic analysis
                    signals.notes.append(f'agentdojo: ground truth unavailable ({type(exc).__name__})')
        return signals

    @staticmethod
    def _tool_status(trace: List[Mapping[str, Any]], view: TraceView) -> Dict[int, ToolStatusOverride]:
        by_id = {call.call_id: call.position for call in view.tool_calls if call.call_id}
        status: Dict[int, ToolStatusOverride] = {}
        used: set = set()
        for order, item in enumerate(trace):
            position = by_id.get(item.get('call_id'))
            if position is None:
                # Calls with unparseable arguments are not traced, so order-based matching is a fallback only.
                position = next(
                    (
                        call.position
                        for call in view.tool_calls
                        if call.position not in used and call.name == item.get('function')
                    ),
                    None,
                )
            if position is None:
                continue
            used.add(position)
            observation = str(item.get('observation') or '')
            if not item.get('valid_function', True):
                status[position] = ToolStatusOverride(failed=True, error_type='invalid_function', message=observation)
            elif item.get('formatter_error'):
                status[position] = ToolStatusOverride(failed=True, error_type='formatter_error', message=observation)
            elif 'runtime_error' in item:
                status[position] = (
                    ToolStatusOverride(failed=True, error_type='runtime_error', message=observation)
                    if item['runtime_error']
                    else ToolStatusOverride(failed=False)
                )
            # Older records without ``runtime_error`` fall back to the generic text patterns.
        return status

    @staticmethod
    def _outcome(view: TraceView) -> Optional[Outcome]:
        if 'utility' not in view.score_value:
            return None
        outcome = Outcome(task_success=view.score_value['utility'] == 1.0)
        if 'targeted_asr' in view.score_value:
            outcome.safety_violation = view.score_value['targeted_asr'] == 1.0
        return outcome


__all__ = ['SIDE_EFFECT_TOOLS', 'AgentDojoTraceSignals']
