"""Stream LLM request/response pairs into a :class:`Trajectory`.

Each call to :meth:`record_anthropic_turn` inspects the latest user/tool
messages in the request (= observations from the previous turn) and the
content blocks in the response (= a new agent step), appending
:class:`Step` entries to the trajectory.

P0 scope: text + ``tool_use`` + ``tool_result`` blocks.  Streaming and
OpenAI translation come in P1.
"""

import threading
from typing import Any, Dict, List, Optional

from evalscope.api.model import ModelOutput
from .models import Observation, Step, ToolCallRecord, Trajectory, TurnUsage


class TrajectoryRecorder:
    """Per-trial accumulator owned by the bridge session."""

    def __init__(self, trial_id: str, framework: str, model: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._trajectory = Trajectory(trial_id=trial_id, framework=framework, model=model)
        self._next_step_id = 0

    def _new_step_id(self) -> int:
        sid = self._next_step_id
        self._next_step_id += 1
        return sid

    def record_anthropic_turn(
        self,
        request_body: Dict[str, Any],
        response: ModelOutput,
    ) -> None:
        """Append observation steps (from the latest user/tool message)
        followed by one agent step (from the assistant response).
        """
        with self._lock:
            observations = self._extract_observations(request_body.get('messages') or [])
            if observations:
                self._trajectory.steps.append(
                    Step(step_id=self._new_step_id(), source='tool', observations=observations)
                )

            text, tool_calls = self._split_response(response)
            usage = self._extract_usage(response)
            if usage is not None:
                self._trajectory.total_usage = self._trajectory.total_usage.__class__(
                    input_tokens=self._trajectory.total_usage.input_tokens + usage.input_tokens,
                    output_tokens=self._trajectory.total_usage.output_tokens + usage.output_tokens,
                    total_tokens=self._trajectory.total_usage.total_tokens + usage.total_tokens,
                )

            self._trajectory.steps.append(
                Step(
                    step_id=self._new_step_id(),
                    source='agent',
                    message=text or None,
                    tool_calls=tool_calls,
                    usage=usage,
                )
            )

    def snapshot(self) -> Trajectory:
        """Return a deep copy of the current trajectory."""
        with self._lock:
            return self._trajectory.model_copy(deep=True)

    # ---- helpers -----------------------------------------------------

    @staticmethod
    def _extract_observations(messages: List[Dict[str, Any]]) -> List[Observation]:
        if not messages:
            return []
        last = messages[-1]
        if last.get('role') != 'user':
            return []
        content = last.get('content')
        if not isinstance(content, list):
            return []
        out: List[Observation] = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                raw = block.get('content', '')
                if isinstance(raw, list):
                    text = ''.join(b.get('text', '') for b in raw if isinstance(b, dict) and b.get('type') == 'text')
                else:
                    text = str(raw)
                out.append(
                    Observation(
                        tool_call_id=block.get('tool_use_id'),
                        output=text,
                        is_error=bool(block.get('is_error', False)),
                    )
                )
        return out

    @staticmethod
    def _split_response(response: ModelOutput) -> tuple[str, List[ToolCallRecord]]:
        if not response.choices:
            return '', []
        message = response.choices[0].message
        text = message.text or ''
        tool_calls: List[ToolCallRecord] = []
        for tc in message.tool_calls or []:
            fn = tc.function
            if isinstance(fn, str):
                name = fn
                args: Dict[str, Any] = {}
            else:
                name = getattr(fn, 'name', '') or ''
                args = getattr(fn, 'arguments', {}) or {}
            tool_calls.append(ToolCallRecord(id=tc.id, name=name, arguments=args))
        return text, tool_calls

    @staticmethod
    def _extract_usage(response: ModelOutput) -> Optional[TurnUsage]:
        if response.usage is None:
            return None
        return TurnUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )
