import re
from difflib import SequenceMatcher
from typing import Dict, Optional

OPTION_PATTERNS = [
    re.compile(r'(?:答案|正确答案|选项|请选择|应选|选择|选|answer)\s*[:：]?\s*([ABCD])\b', re.IGNORECASE),
    re.compile(r'\b([ABCD])\s*[\.、:：\)]', re.IGNORECASE),
    re.compile(r'[\(（\[]\s*([ABCD])\s*[\)）\]]', re.IGNORECASE),
    re.compile(r'\b([ABCD])\b', re.IGNORECASE),
]


def _normalize_text(text: str) -> str:
    text = text or ''
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_first_option(text: str) -> Optional[str]:
    """
    提取首个匹配到的 A/B/C/D 选项。
    匹配顺序：
    1) 明确答案语义模式；
    2) 列表风格 A. / A) / A：；
    3) 括号风格 (A)；
    4) 兜底独立字母 A/B/C/D。
    """
    normalized = _normalize_text(text)
    if not normalized:
        return None

    for pattern in OPTION_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return match.group(1).upper()
    return None


def evaluate_vqa_exam_sample(completion: str, solution: str) -> Dict[str, object]:
    """
    VQA 选择题评测：
    - 若 pred/gt 都能提取出选项，则按选项是否一致计分。
    - 若任一侧无法提取选项，则退化为轻量文本相似度。
    """
    pred_option = extract_first_option(completion)
    gt_option = extract_first_option(solution)

    if pred_option is not None and gt_option is not None:
        score = 1.0 if pred_option == gt_option else 0.0
        mode = 'option_match'
    else:
        pred_text = _normalize_text(completion).lower()
        gt_text = _normalize_text(solution).lower()
        if not pred_text and not gt_text:
            score = 1.0
        elif not pred_text or not gt_text:
            score = 0.0
        else:
            score = SequenceMatcher(None, pred_text, gt_text).ratio()
        mode = 'fallback_text_similarity'

    return {
        'vqa_exam_score': float(score),
        'pred_option': pred_option,
        'gt_option': gt_option,
        'match_mode': mode,
    }
