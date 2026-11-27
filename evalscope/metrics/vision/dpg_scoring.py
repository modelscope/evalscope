# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure scoring functions for DPG-Bench (ELLA, TencentQQGYLab).

Follows the per-image rules in ELLA's ``dpg_bench/compute_dpg_bench.py``:

- Every atomic question is answered by a VQA judge; the answer scores 1.0 only when it is
  exactly ``'yes'`` (case-sensitive, no punctuation), else 0.0.
- Dependency propagation: when any parent question of a proposition is adjusted to 0, the
  proposition is forced to 0 as well (``adjusted_scores``).
- The per-image score is the mean of the adjusted scores of all its questions
  (0-1; multiply by 100 for the leaderboard scale).
- The L1 category scores (Global / Entity / Attribute / Relation / Other) aggregate the
  RAW (pre-dependency) scores at question level across all images, i.e. total yes answers
  divided by total questions of that category -- corpus level, not a mean of per-image
  ratios.
"""

L1_CATEGORIES = ('global', 'entity', 'attribute', 'relation', 'other')


def parse_questions(records: list[dict]) -> list[dict]:
    """Normalize the sample's question records.

    Each record carries ``proposition_id`` (int), ``question`` (str), ``dependency``
    (list of parent ids, empty or [0] for roots), ``category_broad`` (L1) and
    ``category_detailed`` (L2).
    """
    questions = []
    for record in records:
        dependency = record.get('dependency') or []
        dependency = [int(d) for d in dependency if int(d) != 0]
        questions.append(
            {
                'qid': int(record.get('proposition_id', 0)),
                'question': str(record.get('question', '')),
                'dependency': dependency,
                'l1': str(record.get('category_broad', 'other')).strip().lower(),
                'l2': str(record.get('category_detailed', '')).strip().lower(),
            }
        )
    return questions


def raw_and_adjusted(questions: list[dict], answers: dict[int, str]) -> tuple[dict[int, float], dict[int, float]]:
    """Compute raw and dependency-adjusted per-question scores.

    ``answers`` maps question id -> judge reply. Returns ``(raw_scores, adjusted_scores)``
    as dicts keyed by qid.
    """
    raw = {q['qid']: float(answers[q['qid']] == 'yes') for q in questions}
    adjusted = raw.copy()
    for q in questions:
        # The official pass starts with raw scores, so a later parent supplies its raw verdict.
        if any(adjusted[pid] == 0.0 for pid in q['dependency']):
            adjusted[q['qid']] = 0.0
    return raw, adjusted


def overall_counts(questions: list[dict], adjusted: dict[int, float]) -> tuple[float, float, int]:
    """Overall (adjusted) counts: ``(score, yes_sum, question_count)``."""
    total = len(questions)
    if not total:
        return 0.0, 0, 0
    yes_sum = sum(adjusted[q['qid']] for q in questions)
    score = yes_sum / total
    return score, yes_sum, total


def category_raw_counts(questions: list[dict], raw: dict[int, float], category: str) -> tuple[float, float, int]:
    """Official L1 category score over RAW scores: ``(score, yes_sum, question_count)``.

    Questions whose ``category_broad`` differs from ``category`` are excluded; a category
    with no questions in this sample contributes nothing (0 numerator, 0 denominator).
    """
    relevant = [q for q in questions if q['l1'] == category]
    total = len(relevant)
    if not total:
        return 0.0, 0, 0
    yes_sum = sum(raw[q['qid']] for q in relevant)
    return yes_sum / total, yes_sum, total
