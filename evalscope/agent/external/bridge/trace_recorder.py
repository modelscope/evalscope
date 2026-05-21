"""Bridge-side recorder that writes :class:`AgentTrace` directly.

Replaces the legacy ``TrajectoryRecorder`` which maintained its own
``Trajectory`` / ``Step`` schema.  This recorder emits the same
:class:`AgentTraceEvent` shapes the native :class:`AgentLoop` does, so
downstream metric / serialization layers see a single trace format
regardless of whether the run was driven by AgentLoop or by an external
CLI through the bridge.

Step semantics mirror :class:`AgentLoop`:

* Each ``MODEL_GENERATE`` increments ``step``.
* All ``TOOL_CALL`` events emitted from the same assistant message share
  that step.
* ``tool_result`` blocks that arrive in the *next* request body are
  recorded as ``TOOL_RESULT`` events under ``step + 1`` (i.e. they live
  on the step they were observed by, matching native ordering).

Wall-clock ``RUN_START`` / ``RUN_END`` events bracket the CLI lifecycle
and are emitted by :mod:`evalscope.agent.external.adapter`.
"""

import threading
from typing import Any, Dict, List, Optional

from evalscope.api.agent.trace import AgentTrace, EventType
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction
from .translate_anthropic import unpack_tool_call


class BridgeTraceRecorder:
    """Per-trial accumulator owned by the bridge session.

    Thread-safety: ``record_anthropic_turn`` may be invoked concurrently
    from the bridge's aiohttp handlers, so all mutating operations take
    ``self._lock``.
    """

    def __init__(self, trial_id: str, framework: str, model_name: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._trace = AgentTrace(
            framework=framework,
            trial_id=trial_id,
            strategy=None,
            environment=None,
            max_steps=0,
            total_usage=None,
        )
        self._model_name = model_name
        self._step = -1
        self._messages: List[ChatMessage] = []
        self._seen_first_user = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_anthropic_turn(
        self,
        request_body: Dict[str, Any],
        output: ModelOutput,
        *,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Append events for one Anthropic request/response round-trip.

        Order: first surface any ``tool_result`` blocks from the latest
        user message as ``TOOL_RESULT`` events on the *next* step, then
        emit ``MODEL_GENERATE`` (incrementing step) + per-tool-call
        ``TOOL_CALL`` events on the new step.
        """
        with self._lock:
            self._ingest_initial_user_message(request_body)

            tool_results = self._extract_tool_results(request_body.get('messages') or [])
            if tool_results:
                next_step = self._step + 1
                for tc_id, text, is_error in tool_results:
                    msg = ChatMessageTool(
                        content=text,
                        tool_call_id=tc_id,
                        error=ToolCallError(type='unknown', message=text) if is_error else None,
                    )
                    self._messages.append(msg)
                    self._trace.add_event(
                        step=next_step,
                        type=EventType.TOOL_RESULT,
                        message_id=msg.id,
                        payload={
                            'id': tc_id,
                            'error': 'unknown' if is_error else None
                        },
                    )

            self._step += 1

            if output.usage is not None:
                self._trace.total_usage = (
                    output.usage if self._trace.total_usage is None else self._trace.total_usage + output.usage
                )

            assistant_msg = self._build_assistant_message(output)
            self._messages.append(assistant_msg)
            self._trace.add_event(
                step=self._step,
                type=EventType.MODEL_GENERATE,
                message_id=assistant_msg.id,
                latency_ms=latency_ms,
                token_usage=_flat_usage(output),
                payload={'stop_reason': output.stop_reason},
            )

            for tc in assistant_msg.tool_calls or []:
                name, args = unpack_tool_call(tc)
                self._trace.add_event(
                    step=self._step,
                    type=EventType.TOOL_CALL,
                    message_id=assistant_msg.id,
                    payload={
                        'name': name,
                        'arguments': args,
                        'id': tc.id
                    },
                )

    def record_run_start(
        self,
        *,
        framework: str,
        cmd_summary: str,
        env_summary: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        """Record CLI launch.  Step stays -1 because no generate happened yet."""
        with self._lock:
            self._trace.add_event(
                step=max(self._step, 0),
                type=EventType.RUN_START,
                payload={
                    'framework': framework,
                    'cmd': cmd_summary,
                    'env_keys': env_summary or [],
                    'cwd': cwd,
                },
            )

    def record_run_end(
        self,
        *,
        returncode: int,
        timed_out: bool,
        wall_time: float,
        error: Optional[str] = None,
    ) -> None:
        """Record CLI exit.  Step pinned to the most recent generate step."""
        with self._lock:
            payload: Dict[str, Any] = {
                'returncode': returncode,
                'timed_out': timed_out,
                'wall_time': wall_time,
            }
            if error:
                payload['error'] = error
            self._trace.add_event(
                step=max(self._step, 0),
                type=EventType.RUN_END,
                payload=payload,
            )

    def snapshot(self) -> AgentTrace:
        """Deep copy of the accumulated trace."""
        with self._lock:
            return self._trace.model_copy(deep=True)

    def messages(self) -> List[ChatMessage]:
        """Shallow copy of the reconstructed message transcript."""
        with self._lock:
            return list(self._messages)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ingest_initial_user_message(self, request_body: Dict[str, Any]) -> None:
        """Capture the very first user prompt the agent sent — used for
        the messages() transcript so downstream consumers see the original
        instruction, not just observations + assistant turns."""
        if self._seen_first_user:
            return
        messages = request_body.get('messages') or []
        for entry in messages:
            if not isinstance(entry, dict) or entry.get('role') != 'user':
                continue
            text = _user_text_from_content(entry.get('content'))
            if text:
                self._messages.append(ChatMessageUser(content=text))
                self._seen_first_user = True
            return

    @staticmethod
    def _extract_tool_results(messages: List[Any]) -> List[tuple]:
        if not messages:
            return []
        last = messages[-1]
        if not isinstance(last, dict) or last.get('role') != 'user':
            return []
        content = last.get('content')
        if not isinstance(content, list):
            return []
        out: List[tuple] = []
        for block in content:
            if not isinstance(block, dict) or block.get('type') != 'tool_result':
                continue
            raw = block.get('content', '')
            if isinstance(raw, list):
                text = ''.join(b.get('text', '') for b in raw if isinstance(b, dict) and b.get('type') == 'text')
            else:
                text = str(raw)
            out.append((block.get('tool_use_id'), text, bool(block.get('is_error', False))))
        return out

    @staticmethod
    def _build_assistant_message(output: ModelOutput) -> ChatMessageAssistant:
        if not output.choices:
            return ChatMessageAssistant(content='')
        src = output.message
        tool_calls: List[ToolCall] = []
        for tc in src.tool_calls or []:
            if isinstance(tc.function, ToolFunction):
                fn = tc.function
            else:
                fn = ToolFunction(name=str(tc.function), arguments={})
            tool_calls.append(ToolCall(id=tc.id, function=fn, type='function'))
        return ChatMessageAssistant(
            content=src.text or '',
            tool_calls=tool_calls or None,
        )


def _user_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text').strip()
    return ''


def _flat_usage(output: ModelOutput) -> Optional[Dict[str, int]]:
    usage = output.usage
    if usage is None:
        return None
    return {
        'input': int(usage.input_tokens or 0),
        'output': int(usage.output_tokens or 0),
        'total': int(usage.total_tokens or 0),
    }


__all__ = ['BridgeTraceRecorder']
