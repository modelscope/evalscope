"""IE_eval_v2 scoring logic.

Goals:
1. Maximize coverage of the solution text within the completion without
    penalizing extra text in the completion.
2. Require the completion to be valid strict JSON.

Overall score:
ie_score = 0.5 * text_coverage_score + 0.5 * json_strict_score
"""

import json
import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple


def _normalize_text(text: str) -> str:
    s = (text or '').strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def _strip_markdown_json_fence(text: str) -> str:
    raw = (text or '').strip()
    match = re.fullmatch(r'```(?:json|JSON)?\s*\n([\s\S]*?)\n```', raw)
    if match:
        return match.group(1).strip()
    return raw


def _strict_json_parse(text: str) -> Tuple[Optional[object], Optional[str]]:
    """
    Strict JSON validation.
    - Only accept a complete JSON payload, allowing surrounding whitespace.
    - Strip an outer Markdown code fence before validation when present.
    - Reject additional explanatory text outside the JSON payload.
    """
    raw = (text or '').strip()
    if not raw:
        return None, 'empty'

    raw = _strip_markdown_json_fence(raw)

    try:
        return json.loads(raw), None
    except Exception as e:
        return None, str(e)


def _text_coverage_score(solution: str, completion: str) -> float:
    """
    Coverage score measured from the solution side.
    score = matched_characters / solution_characters
    Extra text in the completion does not reduce the score.
    """
    gt = _normalize_text(solution)
    pred = _normalize_text(completion)

    if not gt:
        return 1.0
    if not pred:
        return 0.0

    matcher = SequenceMatcher(None, gt, pred)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return max(0.0, min(1.0, matched / max(len(gt), 1)))


def evaluate_ie_sample(completion: str, solution: str, prompt: Optional[str] = None) -> Dict[str, object]:
    """
    IE v2 evaluation.
    - text_coverage_score: coverage of the solution text in the completion.
    - json_strict_score: whether the completion is strict JSON (1 or 0).
    - ie_score: 0.5 * text_coverage_score + 0.5 * json_strict_score.

    The prompt parameter is retained for compatibility with the existing call signature.
    """
    _ = prompt

    text_coverage = _text_coverage_score(solution=solution, completion=completion)
    pred_json, pred_err = _strict_json_parse(completion)
    json_strict_score = 1.0 if pred_json is not None else 0.0

    ie_score = 0.5 * text_coverage + 0.5 * json_strict_score
    ie_score = max(0.0, min(1.0, ie_score))

    return {
        'ie_score': float(ie_score),
        'text_coverage_score': float(text_coverage),
        'json_strict_score': float(json_strict_score),
        'pred_parse_ok': pred_json is not None,
        'gt_parse_ok': True,
        'pred_parse_error': pred_err,
        'gt_parse_error': None,
        'eval_mode': 'coverage_plus_strict_json_v2',
    }
