# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-row validator functions for MiniMax-Vendor-Verifier.

Each function returns a dict of score fields. The adapter merges these into
``Score.value`` and the aggregator turns them into rates. The functions are
intentionally stateless so the adapter can dispatch them by ``check_type``.

Ports of validators from
https://github.com/MiniMax-AI/MiniMax-Provider-Verifier/tree/main/validator
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# n-gram repeat detection defaults — matches MiniMax RepeatNGramValidator
_NGRAM_N = 3
_NGRAM_REPEAT_THRESHOLD = 4

# Cyrillic Unicode range used by Russian / other Slavic scripts
_CYRILLIC_START = 0x0400
_CYRILLIC_END = 0x04FF

# Common command prefixes that indicate a merged shell argument when followed
# by a space — see MiniMax tool_calls.py is_valid_array_command.
_COMMON_COMMANDS = (
    'ls ',
    'cat ',
    'git ',
    'npm ',
    'npx ',
    'cd ',
    'cp ',
    'mv ',
    'rm ',
    'mkdir ',
    'chmod ',
    'chown ',
    'find ',
    'grep ',
    'curl ',
    'wget ',
    'pip ',
)

# ---------------------------------------------------------------------------
# tool_calls
# ---------------------------------------------------------------------------


def _is_shell_c_invocation(cmd: list) -> bool:
    if not cmd or len(cmd) < 3:
        return False
    shell = cmd[0]
    if shell not in (
        'bash',
        'sh',
        'zsh',
        '/bin/bash',
        '/bin/sh',
        '/bin/zsh',
        '/usr/bin/bash',
        '/usr/bin/sh',
        '/usr/bin/zsh',
    ):
        return False
    for arg in cmd[1:]:
        if arg in ('-c', '-lc'):
            return True
        if arg in ('-l', '--login'):
            continue
        break
    return False


def is_valid_array_command(cmd: Any) -> bool:
    """Validate that an array command has properly separated arguments.

    Returns False when the model merged multiple arguments into a single
    string (e.g. ``["ls -la /workspace"]``) which would fail at execvp time.
    Shell ``-c`` invocations are exempted because the script string is
    expected to contain spaces.
    """
    if not isinstance(cmd, list) or not cmd:
        return False
    if _is_shell_c_invocation(cmd):
        return True
    for elem in cmd:
        if not isinstance(elem, str):
            return False
        if ' ' in elem:
            if any(elem.startswith(prefix) for prefix in _COMMON_COMMANDS):
                return False
    if len(cmd) == 1 and ' ' in cmd[0]:
        return False
    return True


def validate_tool_call_with_array_command(tool_call: Any, tools: List[Dict[str, Any]]) -> bool:
    """JSON-schema validation plus MiniMax's array-command linter.

    ``tool_call`` is a ``ToolCall`` pydantic model from evalscope.
    """
    from jsonschema import ValidationError, validate

    try:
        tool_name = tool_call.function.name
        schema = next(
            (t['function']['parameters'] for t in tools if t['function']['name'] == tool_name),
            None,
        )
        if not schema:
            return False
        args = tool_call.function.arguments
        if isinstance(args, str):
            args = json.loads(args)
        validate(instance=args, schema=schema)
        # Extra: command-array soundness check
        for param_name, param_schema in (schema.get('properties') or {}).items():
            if (
                param_name == 'command' and param_schema.get('type') == 'array'
                and (param_schema.get('items') or {}).get('type') == 'string'
            ):
                if not is_valid_array_command(args.get(param_name)):
                    return False
        return True
    except (json.JSONDecodeError, ValidationError, KeyError):
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Language following (Cyrillic absence)
# ---------------------------------------------------------------------------


def has_no_cyrillic_chars(text: str) -> bool:
    """Return True iff ``text`` contains no Cyrillic-block code points."""
    if not text:
        return True
    return not any(_CYRILLIC_START <= ord(ch) <= _CYRILLIC_END for ch in text)


# ---------------------------------------------------------------------------
# n-gram repeat
# ---------------------------------------------------------------------------


def has_no_repeated_ngram(text: str, n: int = _NGRAM_N, repeat_count: int = _NGRAM_REPEAT_THRESHOLD) -> bool:
    """Return True iff no length-``n`` substring appears ``repeat_count`` or more times.

    Uses a single-pass sliding-window ``Counter`` to narrow down candidates,
    then verifies each candidate's non-overlapping count with ``str.count``
    to preserve the upstream semantics (the original implementation used
    overlap-counting from ``text.count``, but its outer loop made it O(L²);
    the sliding count is a strict upper bound on the non-overlapping count,
    so candidates filtered out here are guaranteed to also be safe under
    ``text.count``).
    """
    from collections import Counter

    if not text or len(text) < n:
        return True
    sliding_counts = Counter(text[i:i + n] for i in range(len(text) - n + 1))
    for ngram, count in sliding_counts.items():
        if count < repeat_count:
            continue
        if text.count(ngram) >= repeat_count:
            return False
    return True


# ---------------------------------------------------------------------------
# Scenario check (JSON key order preservation)
# ---------------------------------------------------------------------------


def _strip_think_blocks(text: str) -> str:
    """Drop ``<think>...</think>`` blocks to leave the user-visible reply."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_expected_param_order(tools: List[Dict[str, Any]]) -> Optional[List[str]]:
    """Pull the declared property order from the first tool's ``parameters.properties``."""
    if not tools or not isinstance(tools, list):
        return None
    params = (tools[0].get('function', {}).get('parameters') or {})
    if 'properties' in params:
        return list((params.get('properties') or {}).keys())
    schema_keywords = {
        'type',
        'description',
        'required',
        'additionalProperties',
        '$schema',
        'items',
        'enum',
        'default',
    }
    keys = [k for k in params.keys() if k not in schema_keywords]
    return keys if keys else None


def check_param_order_preserved(text: str, expected: List[str]) -> Dict[str, Any]:
    """Compare the order of first-occurrence of expected param names in the reply.

    Returns ``{'checked': bool, 'valid': bool | None, 'expected': ..., 'actual': ...}``.
    Matches MiniMax ScenarioCheckValidator: only considered checked once we
    find at least 2 of the expected names in the visible reply.
    """
    visible = _strip_think_blocks(text or '')
    positions = []
    for name in expected:
        idx = visible.find(name)
        if idx != -1:
            positions.append((idx, name))
    positions.sort(key=lambda x: x[0])
    actual = [name for _, name in positions]
    if len(actual) < 2:
        return {'checked': False, 'valid': None, 'expected': expected, 'actual': actual}
    valid = actual == expected[:len(actual)]
    return {'checked': True, 'valid': valid, 'expected': expected, 'actual': actual}
