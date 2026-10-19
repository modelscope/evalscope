# Scoring for TableBench, following the official evaluation in https://github.com/TableBench/TableBench
# (eval_tablebench_script.py, metrics/qa_metrics.py, metrics/custom_em_metric.py,
# parse_tablebench_instruction_response_script.py; the 2025-04 "refined" metrics).
#
# The official pipeline normalizes prediction and reference (lowercase, drop punctuation, articles and
# extra spaces) *before* every metric, so e.g. "10.6" and "1,060" both become digit strings and a minus
# sign is dropped. This module keeps that behaviour so scores are comparable with the leaderboard.

import re
import string
from decimal import ROUND_HALF_UP, Decimal
from typing import List, Optional

_FINAL_ANSWER_RE = re.compile(r'Final Answer: (.+)')
_NUMBER_RE = re.compile(r'^-?\d+(\.\d+)?%?$')

# Data-analysis subtypes scored with a 10% numeric tolerance or strict EM; the rest use ROUGE-L.
EM_TOLERANCE_SUBTYPES = {'CorrelationAnalysis', 'TrendForecasting', 'StatisticalAnalysis'}
EM_SUBTYPES = {'ImpactAnalysis'}


def parse_final_answer(prediction: str) -> str:
    """Text after the first ``Final Answer: `` up to the end of that line ('' if absent), as the official parser."""
    match = _FINAL_ANSWER_RE.search(prediction or '')
    return match.group(1) if match else ''


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, remove punctuation, articles and extra whitespace."""
    text = (text or '').lower()
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def _is_number(value: str) -> bool:
    return bool(_NUMBER_RE.match(value.strip()))


def _to_decimal(value: str) -> Decimal:
    if value.endswith('%'):
        return (Decimal(value.strip('%')) / Decimal('100')).quantize(Decimal('1.0000'), rounding=ROUND_HALF_UP)
    return Decimal(value)


def _precision(values: List[str]) -> int:
    precisions = [len(v.split('.')[-1]) if '.' in v else 0 for v in values if not v.endswith('%')]
    return min(precisions) if precisions else 0


def _round(value: Decimal, precision: int) -> str:
    return str(value.quantize(Decimal(f'1.{"0" * precision}'), rounding=ROUND_HALF_UP))


def exact_match(prediction: str, reference: str, tolerance: Optional[float] = None) -> float:
    """Official per-sample EM on normalized strings.

    The reference is split on commas into answers of equal weight; the i-th predicted answer is compared
    with the i-th reference answer. Numbers are compared after rounding to the reference precision, or
    within ``tolerance`` (relative, e.g. 0.1 for 10%) when it is given.
    """
    refs = [x.strip() for x in normalize_answer(reference).split(',')]
    preds = [x.strip() for x in normalize_answer(prediction).split(',')]
    weight = 1.0 / len(refs)
    score = 0.0
    for i, ref in enumerate(refs):
        if i >= len(preds):
            continue
        pred = preds[i]
        if not _is_number(ref):
            score += weight if ref == pred else 0.0
            continue
        try:
            if tolerance is not None:
                gold, guess = _to_decimal(ref), _to_decimal(pred)
                ok = guess == gold if gold == 0 else abs(gold - guess) / abs(gold) <= tolerance
            elif ref.endswith('%'):
                ok = _to_decimal(ref) == _to_decimal(pred)
            else:
                precision = _precision([x for x in refs if _is_number(x) and not x.endswith('%')])
                ok = _round(_to_decimal(ref), precision) == _round(_to_decimal(pred), precision)
        except Exception:  # an unparsable predicted number scores 0 for this answer
            ok = False
        score += weight if ok else 0.0
    return score


_ROUGE_SCORER = None


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure on normalized strings with the rouge_score default tokenizer (as ``evaluate``'s rouge)."""
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from evalscope.metrics.utils.bundled_rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'])
    return _ROUGE_SCORER.score(normalize_answer(reference), normalize_answer(prediction))['rougeL'].fmeasure


def metric_for(qtype: str, qsubtype: str) -> str:
    """Which official metric scores a question: ``em``, ``em_tolerance_10`` or ``rouge_l``."""
    if qtype != 'DataAnalysis' or qsubtype in EM_SUBTYPES:
        return 'em'
    if qsubtype in EM_TOLERANCE_SUBTYPES:
        return 'em_tolerance_10'
    return 'rouge_l'


def score_answer(parsed: str, reference: str, qtype: str, qsubtype: str) -> float:
    metric = metric_for(qtype, qsubtype)
    if metric == 'em':
        return exact_match(parsed, reference)
    if metric == 'em_tolerance_10':
        return exact_match(parsed, reference, tolerance=0.1)
    return rouge_l(parsed, reference)
