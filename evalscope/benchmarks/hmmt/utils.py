"""Shared utilities for HMMT benchmark adapters (answer extraction helpers)."""
from __future__ import annotations

import re
from typing import Optional

# ─── Answer extraction with multi-level fallback ───────────────────────────────


def extract_hmmt_answer(prediction: str) -> str:
    """
    Extract final answer from model output with multi-level fallback:
      1) EvalScope built-in math_parser.extract_answer
      2) Last \\boxed{...} or \\fbox{...} (nested-brace aware)
      3) Lines matching "ANSWER: xxx"
      4) Last non-empty line
    """
    pred = (prediction or '').strip()

    # 1) Prefer EvalScope's built-in math extraction logic
    try:
        from evalscope.metrics.math.parser import extract_answer as _extract_answer

        ans = _extract_answer(pred)
        if isinstance(ans, str) and ans.strip():
            return _post_clean(ans)
    except Exception:
        pass

    # 2) Fallback: parse last \boxed{...}
    boxed = _extract_last_boxed(pred)
    if boxed is not None and boxed.strip():
        return _post_clean(boxed)

    # 3) Fallback: parse lines like "ANSWER: xxx"
    ans = _extract_answer_line(pred)
    if ans is not None and ans.strip():
        return _post_clean(ans)

    # 4) Last resort: last non-empty line
    lines = [ln.strip() for ln in pred.splitlines() if ln.strip()]
    return _post_clean(lines[-1] if lines else '')


# ─── Internal helpers ──────────────────────────────────────────────────────────

_ANSWER_LINE_RE = re.compile(r'(?im)^\s*(?:final\s+answer|answer)\s*[:：]\s*(.+?)\s*$')


def _extract_answer_line(text: str) -> Optional[str]:
    matches = _ANSWER_LINE_RE.findall(text or '')
    if not matches:
        return None
    return matches[-1].strip()


def _extract_last_boxed(text: str) -> Optional[str]:
    """
    Extract content from the last occurrence of \\boxed{...} or \\fbox{...}.
    Supports nested braces/parentheses by bracket matching.
    """
    if not text:
        return None

    for token in (r'\boxed', r'\fbox'):
        idx = text.rfind(token)
        while idx != -1:
            j = idx + len(token)
            while j < len(text) and text[j].isspace():
                j += 1

            if j < len(text) and text[j] in '{(':
                inner = _read_balanced_group(text, j)
                if inner is not None:
                    return inner
            idx = text.rfind(token, 0, idx)

    return None


def _read_balanced_group(text: str, open_pos: int) -> Optional[str]:
    """
    Given text and open_pos pointing to '{' or '(',
    return the inside content with proper nesting support.
    """
    if open_pos < 0 or open_pos >= len(text):
        return None

    open_ch = text[open_pos]
    if open_ch == '{':
        close_ch = '}'
    elif open_ch == '(':
        close_ch = ')'
    else:
        return None

    depth = 0
    i = open_pos
    while i < len(text):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[open_pos + 1:i]
        i += 1

    return None


def _post_clean(ans: str) -> str:
    """
    Lightweight cleanup to reduce trivial mismatches.
    Keep LaTeX intact as much as possible.
    """
    s = (ans or '').strip()

    # strip surrounding $...$ if model wraps math mode
    if len(s) >= 2 and s[0] == '$' and s[-1] == '$':
        s = s[1:-1].strip()

    # remove trailing punctuation that often appears after boxed answer
    s = s.rstrip().rstrip('.。,')

    return s.strip()
