"""Bridge-side recorder that writes :class:`AgentTrace` directly.

Emits the same :class:`AgentTraceEvent` shapes the native
:class:`AgentLoop` does, so downstream metric / serialization layers see
a single trace format regardless of whether the run was driven by
AgentLoop or by an external CLI through the bridge.

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
from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction
from evalscope.utils.logger import get_logger
from .translate_anthropic import unpack_tool_call

logger = get_logger()


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

    @property
    def current_step(self) -> int:
        """Latest step that emitted a ``MODEL_GENERATE``; ``-1`` before any turn."""
        return self._step

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
        """Append events for one Anthropic request/response round-trip."""
        self._record_turn(
            request_body,
            output,
            latency_ms=latency_ms,
            extract_tool_results=self._extract_tool_results,
        )

    def record_openai_turn(
        self,
        request_body: Dict[str, Any],
        output: ModelOutput,
        *,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Append events for one OpenAI Chat Completions round-trip."""
        self._record_turn(
            request_body,
            output,
            latency_ms=latency_ms,
            extract_tool_results=self._extract_openai_tool_results,
        )

    def record_responses_turn(
        self,
        request_body: Dict[str, Any],
        output: ModelOutput,
        *,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Append events for one OpenAI Responses API round-trip."""
        self._record_turn(
            request_body,
            output,
            latency_ms=latency_ms,
            extract_tool_results=self._extract_responses_tool_results,
            messages_key='input',
        )

    def _record_turn(
        self,
        request_body: Dict[str, Any],
        output: ModelOutput,
        *,
        latency_ms: Optional[float],
        extract_tool_results,
        messages_key: str = 'messages',
    ) -> None:
        """Shared per-turn recorder for all three wire protocols.

        Order: surface any tool results from the previous turn as
        ``TOOL_RESULT`` events on the *next* step, then ``MODEL_GENERATE``
        (incrementing step) + per-tool-call ``TOOL_CALL`` events on the new
        step. ``extract_tool_results`` returns ``(tc_id, text, is_error)``
        tuples — protocols that don't surface a structured error flag pass
        ``is_error=False`` (the slot is reserved for future use).

        ``messages_key`` selects the request-body field that holds the
        per-turn message list: ``'messages'`` for anthropic / openai chat,
        ``'input'`` for openai responses.
        """
        with self._lock:
            self._ingest_initial_messages(request_body, messages_key=messages_key)

            tool_results = extract_tool_results(request_body.get(messages_key) or [])
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

    def record_run_start(self, *, framework: str, cmd_summary: str) -> None:
        """Record CLI launch.  Step stays -1 because no generate happened yet."""
        with self._lock:
            self._trace.add_event(
                step=max(self._step, 0),
                type=EventType.RUN_START,
                payload={
                    'framework': framework,
                    'cmd': cmd_summary,
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
        """Deep copy of the reconstructed message transcript.

        Mirrors :meth:`AgentLoop._snapshot_assistant_message` so callers can
        freely mutate the returned messages without polluting recorder state
        (e.g. an extractor that rewrites assistant text in-place must not
        retroactively alter the recorded turn).
        """
        with self._lock:
            return [m.model_copy(deep=True) for m in self._messages]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ingest_initial_messages(
        self,
        request_body: Dict[str, Any],
        *,
        messages_key: str = 'messages',
    ) -> None:
        """Capture every initial context message the agent sent on its very
        first turn — top-level ``instructions``, all ``role: system /
        developer`` items, AND **all** ``role: user`` items, in document
        order — so the transcript shows the full setup downstream consumers
        need (the model itself sees all of it).

        Why "all user items", not just the first one: codex splits its
        initial prompt across multiple ``role: 'user'`` messages, e.g.

        * ``input[1]`` = auto-discovered ``AGENTS.md`` + environment context
        * ``input[2]`` = the actual task description (positional argv prompt)

        Capturing only the first user item drops the task description from
        the transcript — exactly the SWE-bench Pro symptom that motivated
        this rewrite. Subsequent turns re-send the same initial messages
        plus tool calls + results; we use ``self._step == -1`` to detect
        "still on the first turn" and skip on every later request.

        Responses ``input[]`` items carry a ``type`` discriminator
        (``message`` / ``function_call`` / ``function_call_output`` /
        ``reasoning``); chat-style ``messages[]`` items don't. Entries
        with a non-``'message'`` ``type`` are skipped — those are tool
        calls / reasoning, not initial setup. ``role: 'assistant'``
        message items are also skipped: model output is captured from
        :class:`ModelOutput` via :meth:`_build_assistant_message`.
        """
        if self._step >= 0:
            return  # past the first turn; later requests just re-send history.

        if instr := request_body.get('instructions'):
            self._messages.append(ChatMessageSystem(content=str(instr)))

        for entry in request_body.get(messages_key) or []:
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get('type')
            if entry_type is not None and entry_type != 'message':
                continue
            role = entry.get('role')
            text = _user_text_from_content(entry.get('content'))
            if not text:
                continue
            if role in ('system', 'developer'):
                self._messages.append(ChatMessageSystem(content=text))
            elif role == 'user':
                self._messages.append(ChatMessageUser(content=text))

    @staticmethod
    def _extract_openai_tool_results(messages: List[Any]) -> List[tuple]:
        """Surface tail ``role:'tool'`` entries — they're the tool_result
        payloads the agent observed after the previous assistant turn.

        Walks backwards from the end of ``messages[]`` collecting consecutive
        tool entries; stops at the first non-tool message. Preserves order.

        Assumes the agent appends tool results immediately before the next
        request (true for codex / claude-code style loops). Tool results that
        are followed by a user message in the same request are silently
        dropped from this turn's TOOL_RESULT events — they were already
        recorded on the prior turn that surfaced them.

        Returns ``(tool_call_id, text, is_error)``; ``is_error`` is always
        ``False`` because the OpenAI Chat Completions spec has no structured
        error flag on tool messages (slot kept for protocol parity).
        """
        if not messages:
            return []
        tail: List[Dict[str, Any]] = []
        for entry in reversed(messages):
            if not isinstance(entry, dict) or entry.get('role') != 'tool':
                break
            tail.append(entry)
        tail.reverse()
        out: List[tuple] = []
        for entry in tail:
            text = _user_text_from_content(entry.get('content', ''))
            out.append((entry.get('tool_call_id'), text, False))
        return out

    def _extract_responses_tool_results(self, items: List[Any]) -> List[tuple]:
        """Return ``*_call_output`` entries that haven't been recorded yet.

        codex re-sends the **entire** transcript on every turn — each
        ``(function_call, function_call_output)`` pair stays wherever it
        originally appeared, NOT clumped at the tail. So we can't use the
        anthropic / openai-chat tail-only shortcut; instead we scan the
        whole list and dedup against ``self._messages`` (every previously
        recorded tool result is a :class:`ChatMessageTool` carrying its
        ``tool_call_id``).

        Recognised output types must match the ``_TOOL_OUTPUT_ITEM_TYPES``
        set in :mod:`translate_responses`. ``is_error`` is always ``False``
        (Responses spec has no structured error flag, parity with the
        openai chat path).
        """
        if not items:
            return []
        OUTPUT_TYPES = {
            'function_call_output',
            'custom_tool_call_output',
            'computer_call_output',
            'local_shell_call_output',
        }
        already_recorded = {
            m.tool_call_id
            for m in self._messages
            if isinstance(m, ChatMessageTool) and m.tool_call_id is not None
        }
        new_entries: List[Dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict) or entry.get('type') not in OUTPUT_TYPES:
                continue
            if entry.get('call_id') in already_recorded or entry.get('call_id') is None:
                continue
            new_entries.append(entry)

        out: List[tuple] = []
        for entry in new_entries:
            raw_output = entry.get('output', '')
            if isinstance(raw_output, dict):
                # computer_call_output.output is typically {'image_url': '...'}.
                # Render to placeholder text since downstream model can't see
                # screenshots (chat-only LLMs).
                if 'image_url' in raw_output:
                    text = f'[image: {raw_output["image_url"]}]'
                else:
                    text = str(raw_output)
            elif isinstance(raw_output, list):
                # function_call_output.output may be a typed content-parts list
                text = '\n'.join(
                    b.get('text', '')
                    for b in raw_output
                    if isinstance(b, dict) and b.get('type') in ('input_text', 'output_text', 'text')
                )
            else:
                text = str(raw_output)
            out.append((entry.get('call_id'), text, False))
        return out

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
            name, args = unpack_tool_call(tc)
            tool_calls.append(ToolCall(
                id=tc.id,
                function=ToolFunction(name=name, arguments=args),
                type='function',
            ))
        return ChatMessageAssistant(
            content=src.text or '',
            tool_calls=tool_calls or None,
        )


def _user_text_from_content(content: Any) -> str:
    """Flatten an Anthropic-or-OpenAI message ``content`` to plain text.

    Accepts ``str`` or a list of content-parts; recognises ``type='text'``
    (Anthropic + OpenAI) and ``type='input_text'`` (OpenAI Responses /
    multi-modal). Non-text blocks (image, tool_use, etc.) are dropped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') in ('text', 'input_text'):
                parts.append(block.get('text', '') or '')
        return '\n'.join(p for p in parts if p).strip()
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
