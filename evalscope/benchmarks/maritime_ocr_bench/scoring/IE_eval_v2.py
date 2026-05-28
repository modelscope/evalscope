"""
IE_eval_v2 计算逻辑说明

目标：
1. solution 中的文本尽量在 completion 中出现（completion 多写不扣分）。
2. completion 必须是严格 JSON 结构。

总分：
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
    严格 JSON 校验：
    - 仅允许完整 JSON 文本（可有首尾空白）
    - 若整体被 markdown code fence 包裹，则先去除 code fence 再校验
    - 不接受额外说明文字
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
    基于 solution 侧召回的覆盖分：
    score = 匹配字符数 / solution字符数
    这样 completion 额外文本不会被扣分。
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
    IE v2 评测：
    - text_coverage_score: solution 文本在 completion 中的覆盖率
    - json_strict_score: completion 是否严格 JSON（是=1，否=0）
    - ie_score: 0.5 * text_coverage_score + 0.5 * json_strict_score

    参数 prompt 保留是为了兼容现有调用签名。
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
