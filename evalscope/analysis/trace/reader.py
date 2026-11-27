"""Read-only access to run outputs, and normalization of one review row into a :class:`TraceView`."""

import json
import os
from typing import Any, Dict, Iterator, List, Tuple

from evalscope.api.agent.trace import AgentTrace, EventType
from evalscope.api.analysis import LoopEvent, SampleRef, TraceToolCall, TraceView
from evalscope.api.evaluator.cache import ReviewResult
from evalscope.report.diagnosis import ReviewFile
from evalscope.utils.io_utils import OutputsStructure
from evalscope.utils.logger import get_logger

logger = get_logger()

_KNOWN_EVENT_TYPES = frozenset(event.value for event in EventType)
_LOOP_EVENT_TYPES = frozenset(
    event.value
    for event in (
        EventType.ERROR,
        EventType.NUDGE,
        EventType.SUBMIT,
        EventType.RUN_START,
        EventType.RUN_END,
        EventType.ENV_RESET,
    )
)


# ---------------------------------------------------------------------------
# Run outputs
# ---------------------------------------------------------------------------


def _is_run_dir(path: str) -> bool:
    return any(
        os.path.isdir(os.path.join(path, folder))
        for folder in (
            OutputsStructure.REVIEWS_DIR,
            OutputsStructure.REPORTS_DIR,
        )
    )


def discover_runs(path: str) -> List[str]:
    """Discover complete nested run trees without silently dropping deeper test directories."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Run directory not found: {path}')
    if _is_run_dir(path):
        return [path]
    found: List[str] = []
    for root, dirs, _ in os.walk(path):
        if _is_run_dir(root):
            found.append(root)
            dirs[:] = []
        else:
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ('trace_analysis', 'analysis'))
    return sorted(found)


def load_review_rows(review_file: ReviewFile) -> Tuple[List[ReviewResult], int]:
    """Return the readable rows of one review file and the number of unreadable rows."""
    results: List[ReviewResult] = []
    unreadable = 0
    with open(review_file.path, encoding='utf-8-sig') as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError('Expected a JSON object')
                results.append(ReviewResult.from_cache_item(row))
            except (ValueError, TypeError) as exc:
                unreadable += 1
                logger.warning(
                    f'Skipping unreadable review row {line_number} in {review_file.path}: {type(exc).__name__}'
                )
    return results, unreadable


def iter_report_pairs(run_dir: str) -> Iterator[Tuple[str, str]]:
    """Discover report-only benchmarks; their contents are parsed once by the input loader."""
    reports_root = os.path.join(run_dir, OutputsStructure.REPORTS_DIR)
    if not os.path.isdir(reports_root):
        return
    for model in sorted(os.listdir(reports_root)):
        model_dir = os.path.join(reports_root, model)
        if os.path.isdir(model_dir):
            for name in sorted(os.listdir(model_dir)):
                if name.endswith('.json'):
                    yield model, name[:-5]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _numeric_scores(value: Dict[str, Any]) -> Dict[str, float]:
    return {key: float(item) for key, item in (value or {}).items() if isinstance(item, (bool, int, float))}


def has_nontext_output(record: ReviewResult) -> bool:
    """Media and structured outputs are not eligible for text-only generation checks."""
    return any(
        part.type not in ('text', 'reasoning')
        for message in record.messages
        if message.role == 'assistant' and isinstance(message.content, list)
        for part in message.content
    )


def _message_text(message: Any) -> str:
    """Visible text only; ``ChatMessage.text`` already skips reasoning parts."""
    try:
        return message.text or ''
    except Exception:  # noqa: BLE001 - unusual content shapes degrade to empty text
        return ''


def _calls_from_messages(record: ReviewResult) -> List[TraceToolCall]:
    calls: List[TraceToolCall] = []
    by_id: Dict[str, TraceToolCall] = {}
    for index, message in enumerate(record.messages):
        if message.role == 'assistant':
            for tool_call in message.tool_calls or []:
                call = TraceToolCall(
                    position=len(calls),
                    message_index=index,
                    call_id=tool_call.id,
                    name=tool_call.function.name or '',
                    arguments=dict(tool_call.function.arguments or {}),
                    parse_error=tool_call.parse_error,
                )
                calls.append(call)
                if tool_call.id:
                    by_id[tool_call.id] = call
        elif message.role == 'tool':
            call = by_id.get(message.tool_call_id) if message.tool_call_id else None
            if not message.tool_call_id:
                # Only legacy results without an ID use order-based matching.
                call = next((item for item in calls if item.result_message_index is None), None)
            if call is None or call.result_message_index is not None:
                continue
            call.result_message_index = index
            call.result_text = _message_text(message)
            if message.error is not None:
                call.error_type, call.error_message, call.error_source = (
                    message.error.type,
                    message.error.message,
                    'structured',
                )
    return calls


def _merge_trace(trace: AgentTrace, view: TraceView) -> None:
    """Add steps, latencies, loop events and token usage; create calls that only exist in the trace."""
    by_id = {call.call_id: call for call in view.tool_calls if call.call_id}
    for event_index, event in enumerate(trace.events):
        event_type = event.type.value if isinstance(event.type, EventType) else str(event.type)
        payload = event.payload or {}
        call_id = payload.get('id')
        if event_type == EventType.TOOL_CALL.value:
            call = by_id.get(call_id) if call_id else None
            if call is None:
                arguments = payload.get('arguments')
                call = TraceToolCall(
                    position=len(view.tool_calls),
                    message_index=-1,
                    call_id=call_id,
                    name=str(payload.get('name') or ''),
                    arguments=arguments if isinstance(arguments, dict) else {'raw': arguments},
                )
                view.tool_calls.append(call)
                if call_id:
                    by_id[call_id] = call
            call.step = event.step
            call.call_event_index = event_index
        elif event_type == EventType.TOOL_RESULT.value:
            call = by_id.get(call_id) if call_id else None
            if call is None:
                continue
            call.result_event_index = event_index
            if call.result_text is None and isinstance(payload.get('preview'), str):
                call.result_text = payload['preview']
            if call.error_type is None and payload.get('error'):
                call.error_type, call.error_source = str(payload['error']), 'structured'
        elif event_type in _LOOP_EVENT_TYPES or event_type not in _KNOWN_EVENT_TYPES:
            source, message = payload.get('source'), payload.get('message')
            view.events.append(
                LoopEvent(
                    step=event.step,
                    type=event_type,
                    source=None if source is None else str(source),
                    message=None if message is None else str(message),
                )
            )


def build_view(record: ReviewResult, ref: SampleRef) -> TraceView:
    """Build the generic view of one sample from its messages and, when present, its agent trace."""
    score = record.sample_score.score
    view = TraceView(
        ref=ref,
        has_nontext_output=has_nontext_output(record),
        score_value=_numeric_scores(score.value),
        main_score_name=score.main_score_name,
        score_status=getattr(score.status, 'value', str(score.status)),
    )
    system_parts: List[str] = []
    user_parts: List[str] = []
    for message in record.messages:
        text = _message_text(message)
        if message.role == 'system':
            system_parts.append(text)
        elif message.role == 'user' and not getattr(message, 'tool_call_id', None):
            user_parts.append(text)
        elif message.role == 'assistant':
            view.assistant_texts.append(text)
    view.system_text = '\n'.join(system_parts)
    view.user_text = '\n'.join(user_parts)
    view.tool_calls = _calls_from_messages(record)
    if record.agent_trace is not None:
        _merge_trace(record.agent_trace, view)
    if isinstance(score.prediction, str):
        view.final_answer = score.prediction
    else:
        view.final_answer = next((text for text in reversed(view.assistant_texts) if text.strip()), None)
    return view


__all__ = ['build_view', 'discover_runs', 'iter_report_pairs', 'load_review_rows']
