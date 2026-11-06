"""Rubric parsing and sandbox command helpers for SWE-Atlas QnA."""

from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List

# ``annotations.type`` values seen across the 124 public tasks. A negative
# verifier inverts the judge's verdict: the claim being present is a failure.
POSITIVE_TYPE = 'positive hli verifier'
NEGATIVE_TYPE = 'negative hli verifier'

FINAL_ANSWER_TAG = '<<FINAL_ANSWER>>'


def parse_rubric(raw: Any) -> List[Dict[str, Any]]:
    """Normalise the ``rubric`` column into id/title/negated triples.

    The column ships as a JSON string; a already-decoded list is accepted so
    callers can reuse this on in-memory records.
    """
    if isinstance(raw, str):
        items = json.loads(raw) if raw.strip() else []
    else:
        items = raw or []

    parsed = []
    for index, item in enumerate(items):
        annotations = item.get('annotations') or {}
        item_type = (annotations.get('type') or POSITIVE_TYPE).strip().lower()
        if item_type not in {POSITIVE_TYPE, NEGATIVE_TYPE}:
            raise ValueError(f'unknown rubric verifier type {item_type!r} at index {index}')
        parsed.append(
            {
                'id': item.get('id') or f'rubric_{index}',
                'title': item.get('title') or '',
                'negated': item_type == NEGATIVE_TYPE,
            }
        )
    return parsed


def rubric_item_passed(claim_met: bool, negated: bool) -> bool:
    """Map a judge verdict onto pass/fail, honouring negative verifiers."""
    return (not claim_met) if negated else claim_met


def reset_repo_command(working_dir: str, base_commit: str) -> str:
    """Return the repository reset documented in the dataset README.

    Pinning to ``base_commit`` matters because the published images are not
    guaranteed to sit at that commit, and the rubric was written against it.
    """
    quoted_dir = shlex.quote(working_dir)
    commit = shlex.quote(base_commit)
    return (
        f'cd {quoted_dir} && '
        f'git config --global --add safe.directory {quoted_dir} && '
        f'git reset --hard {commit} && '
        'git clean -fdq'
    )


def dirty_check_command(working_dir: str) -> str:
    """Return a command printing the working tree's uncommitted changes.

    SWE-Atlas fails a task whose answer came with source edits, so the tree is
    inspected before the sandbox is torn down.
    """
    quoted_dir = shlex.quote(working_dir)
    return f'cd {quoted_dir} && git status --porcelain'


def extract_final_answer(text: str) -> str:
    """Pull the ``<<FINAL_ANSWER>>``-delimited block out of the agent's reply.

    The official scaffold asks the model to wrap its answer in two identical
    tags rather than an open/close pair, so the content between the first and
    the last occurrence is taken. Text without the tag is returned unchanged:
    a model that answered without the wrapper is still judged on what it said.
    """
    if not text:
        return ''
    first = text.find(FINAL_ANSWER_TAG)
    if first < 0:
        return text.strip()
    start = first + len(FINAL_ANSWER_TAG)
    last = text.rfind(FINAL_ANSWER_TAG)
    if last > first:
        return text[start:last].strip()
    return text[start:].strip()
