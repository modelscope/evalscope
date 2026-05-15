"""Shared observation formatter for SWE-bench agentic strategies.

Aligned with the original ``mini-swe-agent`` ``swebench.yaml``
``observation_template``: long outputs are truncated with head/tail
windows, and the result is wrapped with ``<returncode>`` + ``<output>``
tags so the model can reliably parse the bash output.

Also exposes ``check_sentinel`` — the canonical entry point used by
strategies to detect ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` in a raw
bash observation *before* any envelope is rendered.

Private helper: only consumed by sibling modules under
``evalscope.agent.strategies.swe_bench``.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Public sentinel constant -- mirror of mini-swe-agent's submission marker.
# ---------------------------------------------------------------------------

SUBMIT_SENTINEL = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'

# Truncation thresholds (must match mini-swe-agent observation_template).
DEFAULT_MAX_CHARS = 10000
DEFAULT_HEAD = 5000
DEFAULT_TAIL = 5000

# Markers appended by ``evalscope.agent.tools.bash._format_exec_result``.
# We reverse-engineer them here to surface ``returncode`` to the
# observation template without changing the bash tool's public string
# contract.
_EXIT_TAIL_RE = re.compile(r'\n?\[exit (-?\d+)\]\s*\Z')
_TIMEOUT_TAIL_RE = re.compile(r'\n?\[TIMEOUT\]\s*\Z')
_NO_OUTPUT_LITERAL = '(no output)'

# Trailing ``[stderr]`` block appended after the sentinel-payload by the
# bash tool when stderr is non-empty.  Stripped from submissions because
# stderr is not part of the patch.
_TRAILING_STDERR_RE = re.compile(r'\n*\[stderr\][\s\S]*\Z')

_TRUNCATION_WARNING = (
    'The output of your last command was too long.\n'
    'Please try a different command that produces less output.\n'
    "If you're looking at a file you can try use head, tail or sed to view a "
    'smaller number of lines selectively.\n'
    "If you're using grep or find and it produced too much output, you can "
    'use a more selective search pattern.\n'
    "If you really need to see something from the full command's output, "
    'you can redirect output to a file and then search in that file.'
)

# Mirror of mini-swe-agent ``format_error_template`` (model.* section).
_FORMAT_ERROR_TEMPLATE = (
    'Tool call error:\n'
    '\n'
    '<error>\n'
    '{error}\n'
    '</error>\n'
    '\n'
    'Here is general guidance on how to submit correct toolcalls:\n'
    '\n'
    "Every response needs to use the 'bash' tool at least once to execute commands.\n"
    '\n'
    'Call the bash tool with your command as the argument:\n'
    '- Tool: bash\n'
    '- Arguments: {{"command": "your_command_here"}}\n'
    '\n'
    'If you have completed your assignment, please consult the first message about how to\n'
    'submit your solution (you will not be able to continue working on this task after that).'
)


def parse_exec_metadata(observation: str) -> Tuple[int, bool, str]:
    """Reverse-engineer ``(returncode, timed_out, body)`` from bash output.

    The bash tool (``evalscope.agent.tools.bash._format_exec_result``)
    encodes status into a trailing marker line:

    * ``[exit N]`` (only for non-zero return codes)
    * ``[TIMEOUT]`` when the command timed out
    * ``(no output)`` when both stdout and stderr are empty

    Anything else is a successful (returncode == 0) command.  ``body``
    is the original observation with the trailing marker stripped so it
    can be wrapped in ``<output>`` tags as raw command output.
    """
    if observation is None:
        return 0, False, ''
    if observation.strip() == _NO_OUTPUT_LITERAL:
        return 0, False, ''
    timeout_match = _TIMEOUT_TAIL_RE.search(observation)
    if timeout_match:
        return 124, True, observation[:timeout_match.start()].rstrip('\n')
    exit_match = _EXIT_TAIL_RE.search(observation)
    if exit_match:
        return int(exit_match.group(1)), False, observation[:exit_match.start()].rstrip('\n')
    return 0, False, observation


def check_sentinel(observation: str) -> Optional[str]:
    """Return the submission payload when the bash output starts with the sentinel.

    Mirrors mini-swe-agent's ``DockerEnvironment._check_finished``: the
    sentinel must appear on the *first* non-empty line of stdout and the
    command must have completed with returncode 0.  Returns ``None``
    otherwise.

    The returned payload is the raw text following the sentinel line,
    with any trailing ``[stderr]`` block (added by the bash tool when
    stderr is non-empty) removed.
    """
    if not observation:
        return None
    rc, timed_out, body = parse_exec_metadata(observation)
    if rc != 0 or timed_out:
        return None
    if not body:
        return None
    lines = body.lstrip('\n').splitlines(keepends=True)
    if not lines or lines[0].strip() != SUBMIT_SENTINEL:
        return None
    submission = ''.join(lines[1:])
    submission = _TRAILING_STDERR_RE.sub('', submission)
    return submission


def truncate_middle(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    head: int = DEFAULT_HEAD,
    tail: int = DEFAULT_TAIL,
) -> str:
    """Return ``text`` unchanged when short; otherwise head + warning + tail.

    Kept for backwards compatibility with tests that import it directly.
    The mini-swe-agent observation_template encodes truncation slightly
    differently (separate ``<output_head>``/``<output_tail>`` blocks);
    see :func:`format_exec_observation` for the rendered envelope.
    """
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
    error_message: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    head: int = DEFAULT_HEAD,
    tail: int = DEFAULT_TAIL,
) -> str:
    """Wrap a bash observation in mini-swe-agent's observation/error envelope.

    ``observation`` is the raw return value of the ``bash`` tool (combined
    stdout / ``[stderr]`` block / ``[exit N]`` or ``[TIMEOUT]`` marker, as
    produced by :func:`evalscope.agent.tools.bash._format_exec_result`).

    When ``error_message`` is provided (tool execution itself failed
    before bash even ran — e.g. timeout, missing environment, unknown
    tool), it is rendered through mini-swe-agent's ``format_error_template``
    instead of the success envelope.
    """
    if error_message is not None:
        return _FORMAT_ERROR_TEMPLATE.format(error=error_message)

    rc, timed_out, body = parse_exec_metadata(observation or '')

    if timed_out:
        return (f'<returncode>{rc}</returncode>\n'
                f'<exception>Command timed out.</exception>')

    body_text = body or ''
    if len(body_text) < max_chars:
        return (f'<returncode>{rc}</returncode>\n'
                f'<output>\n{body_text.rstrip()}\n</output>')

    elided = len(body_text) - max_chars
    return (
        f'<returncode>{rc}</returncode>\n'
        f'<warning>\n{_TRUNCATION_WARNING}\n</warning>\n'
        f'<output_head>\n{body_text[:head]}\n</output_head>\n'
        f'<elided_chars>\n{elided} characters elided\n</elided_chars>\n'
        f'<output_tail>\n{body_text[-tail:]}\n</output_tail>'
    )


__all__ = [
    'SUBMIT_SENTINEL',
    'check_sentinel',
    'format_exec_observation',
    'parse_exec_metadata',
    'truncate_middle',
]
