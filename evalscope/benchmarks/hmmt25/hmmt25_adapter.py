# evalscope/benchmarks/hmmt25/hmmt25_adapter.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

PROMPT_TEMPLATE = r"""
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.
""".lstrip()


# https://huggingface.co/datasets/MathArena/hmmt_feb_2025
@register_benchmark(
    BenchmarkMeta(
        name='hmmt25',
        pretty_name='HMMT25',
        dataset_id='evalscope/hmmt_feb_2025',
        description=(
            'HMMT February 2025 (MathArena) is a challenging evaluation benchmark '
            'derived from the Harvard-MIT Mathematics Tournament (HMMT) February '
            '2025 competition, one of the most prestigious and difficult high school math contests globally.'
            'The benchmark focuses on advanced mathematical reasoning across four '
            'primary domains: Algebra, Combinatorics, Geometry, and Number Theory.'
        ),
        tags=[Tags.MATH, Tags.REASONING],
        subset_list=['default'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # HF dataset provides split 'train'
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class HMMT25Adapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = str(record.get('problem', '')).strip()
        target = str(record.get('answer', '')).strip()

        # problem_type is sequence[string] in this dataset (can be multiple types)
        ptype = record.get('problem_type', None)

        return Sample(
            input=problem,
            target=target,
            metadata={
                'problem_idx': record.get('problem_idx', None),
                'problem_type': ptype,
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """
        Extract final answer from model output.
        Prefer EvalScope built-in math_parser.extract_answer; fallback to robust boxed-parser.
        """
        pred = (prediction or '').strip()

        # 1) Prefer EvalScope's built-in math extraction logic (recommended for math tasks).
        try:
            from evalscope.metrics.math_parser import extract_answer as _extract_answer  # type: ignore

            ans = _extract_answer(pred)
            if isinstance(ans, str) and ans.strip():
                return _post_clean(ans)
        except Exception:
            # Fall back to local extraction if import fails or parser errors
            pass

        # 2) Fallback: parse last \boxed{...} (supports nested braces)
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


# Helpers (robust parsing)

_ANSWER_LINE_RE = re.compile(r'(?im)^\s*(?:final\s+answer|answer)\s*[:：]\s*(.+?)\s*$')


def _extract_answer_line(text: str) -> Optional[str]:
    matches = _ANSWER_LINE_RE.findall(text or '')
    if not matches:
        return None
    # use the last occurrence
    return matches[-1].strip()


def _extract_last_boxed(text: str) -> Optional[str]:
    """
    Extract content from the last occurrence of \boxed{...} or \fbox{...}.
    Supports nested braces/parentheses by bracket matching.
    """
    if not text:
        return None

    for token in (r'\boxed', r'\fbox'):
        idx = text.rfind(token)
        while idx != -1:
            j = idx + len(token)
            # skip whitespace
            while j < len(text) and text[j].isspace():
                j += 1

            if j < len(text) and text[j] in '{(':
                inner = _read_balanced_group(text, j)
                if inner is not None:
                    return inner
            # try previous occurrence
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
                # content is between open_pos and i
                return text[open_pos + 1:i]
        i += 1

    # no matching close bracket
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
