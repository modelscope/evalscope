"""Agent-loop control-flow exceptions.

These are *not* error conditions — they are flow-control signals that a
strategy or tool can raise to short-circuit the AgentLoop.  The loop
catches them at well-defined points and converts them into normal
termination + result fields.
"""

from __future__ import annotations


class Submitted(Exception):
    """Raised by a strategy/tool to terminate the AgentLoop with a final submission.

    Mirrors mini-swe-agent's ``InterruptAgentFlow``/``Submitted`` pattern:
    when a strategy detects the completion sentinel inside a tool
    observation, it raises ``Submitted(submission=...)`` and the loop
    captures the payload as the final answer without running additional
    steps.
    """

    def __init__(self, submission: str, exit_status: str = 'Submitted') -> None:
        self.submission = submission
        self.exit_status = exit_status
        preview = (submission or '').strip().splitlines()[0:1]
        preview_text = preview[0][:120] if preview else ''
        super().__init__(f'[{exit_status}] {preview_text}')


__all__ = ['Submitted']
