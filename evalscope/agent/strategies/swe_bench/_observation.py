"""Shared XML observation formatter for SWE-bench agentic strategies.

Aligns with the original ``mini-swe-agent`` ``swebench.yaml`` /
``swebench_backticks.yaml`` ``observation_template``: long outputs are
truncated head/tail with a warning, and the result is wrapped in XML-style
tags so the model can reliably parse the bash output.

Private helper: only consumed by sibling modules under
``evalscope.agent.strategies.swe_bench``.
"""

from __future__ import annotations

DEFAULT_MAX_CHARS = 10000
DEFAULT_HEAD = 5000
DEFAULT_TAIL = 5000


def truncate_middle(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    head: int = DEFAULT_HEAD,
    tail: int = DEFAULT_TAIL,
) -> str:
    """Return ``text`` unchanged if short; otherwise head + warning + tail."""
    if text is None:
        return ''
    if len(text) <= max_chars:
        return text
    omitted = len(text) - head - tail
    warning = (
        f'\n<warning>Output truncated: {omitted} characters omitted between '
        f'head and tail. Use `head`, `tail`, `sed -n` or `grep` to inspect '
        'specific regions if needed.</warning>\n'
    )
    return text[:head] + warning + text[-tail:]


def format_exec_observation(
    observation: str,
    *,
    error_message: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    head: int = DEFAULT_HEAD,
    tail: int = DEFAULT_TAIL,
) -> str:
    """Wrap a bash observation in the SWE-bench XML envelope with truncation.

    ``observation`` is the already-merged stdout/stderr string produced by
    the ``bash`` tool (see :func:`evalscope.agent.tools.bash.run_bash`).

    If ``error_message`` is provided (tool execution failed before bash even
    ran — e.g. timeout, missing environment), it is reported as a separate
    ``<error>`` block instead of ``<output>``.
    """
    if error_message:
        return f'<error>{error_message}</error>'

    body = truncate_middle(
        observation or '',
        max_chars=max_chars,
        head=head,
        tail=tail,
    )
    return f'<output>\n{body.rstrip()}\n</output>'


__all__ = ['format_exec_observation', 'truncate_middle']
