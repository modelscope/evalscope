# Copyright (c) Alibaba, Inc. and its affiliates.
# Scoring logic adapted from https://github.com/databricks/officeqa/blob/main/reward.py

import re

# Default tolerance for numerical matching (1% relative error)
DEFAULT_TOLERANCE = 0.01

_CURRENCY_SYMBOLS = r'$£€¥₹¢₩₽'
_NUMBER_BODY = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?'
_VALID_THOUSANDS_RE = r'(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)'


def normalize_text(text: str) -> str:
    """Normalize text for consistent parsing."""
    if not text:
        return ''
    normalized = text.replace('\u2212', '-')
    return re.sub(r'\s+', ' ', normalized).strip()


def _normalize_numeric_formatting(text: str) -> str:
    """Remove currency symbols and handle accounting notation (parentheses = negative)."""

    def _accounting_repl(match: re.Match) -> str:
        number = match.group(2)
        num_val = float(number.replace(',', ''))
        if 1900 <= num_val <= 2100 and num_val == int(num_val):
            return match.group(0)
        return f'-{number}'

    text = re.sub(
        rf'\(\s*([{_CURRENCY_SYMBOLS}])?\s*({_NUMBER_BODY})\s*\)',
        _accounting_repl,
        text,
    )
    return re.sub(rf'[{_CURRENCY_SYMBOLS}]', '', text)


def extract_numbers(text: str) -> list:
    """Extract (number_value, context) tuples from text."""
    if not text:
        return []
    text = normalize_text(text)
    text = _normalize_numeric_formatting(text)
    text_clean = re.sub(
        _VALID_THOUSANDS_RE,
        lambda m: m.group().replace(',', ''),
        text,
    )
    results = []
    for match in re.finditer(r'-?\d+\.?\d*%?', text_clean):
        matched = match.group()
        if not matched or matched == '-':
            continue
        num_text = matched.rstrip('%')
        try:
            num = float(num_text)
        except ValueError:
            continue
        start = max(0, match.start() - 20)
        end = min(len(text_clean), match.end() + 20)
        context = text_clean[start:end].lower()
        results.append((num, context))
    return results


def detect_unit(context: str) -> str:
    """Detect unit from context (million, billion, etc.)."""
    c = context.lower()
    if re.search(r'\btrillions?\b', c):
        return 'trillion'
    if re.search(r'\bbillions?\b', c):
        return 'billion'
    if re.search(r'\bmillions?\b', c):
        return 'million'
    if re.search(r'\bthousands?\b', c):
        return 'thousand'
    return ''


def is_year(num: float) -> bool:
    """Check if a number is likely a year."""
    return 1900 <= num <= 2100 and num == int(num)


def extract_final_answer(text: str) -> str:
    """Extract answer from <FINAL_ANSWER> tags if present."""
    matches = list(re.finditer(r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>', text, re.DOTALL | re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()


def score_answer(ground_truth: str, predicted: str, tolerance: float = DEFAULT_TOLERANCE) -> float:
    """
    Score predicted answer against ground truth using official OfficeQA scoring.

    Adapted from https://github.com/databricks/officeqa/blob/main/reward.py
    """
    if not ground_truth or not predicted:
        return 0.0

    predicted = extract_final_answer(predicted)
    if not predicted or 'unable to determine' in predicted.lower():
        return 0.0

    gt_numbers = extract_numbers(ground_truth)
    pred_numbers = extract_numbers(predicted)

    # Case 1: Both have numbers → numeric comparison
    if gt_numbers and pred_numbers:
        if len(gt_numbers) > 1:
            # Multi-number: all GT numbers must be found in prediction
            matched = 0
            for gt_val, gt_ctx in gt_numbers:
                gt_unit = detect_unit(gt_ctx)
                for pred_val, pred_ctx in pred_numbers:
                    pred_unit = detect_unit(pred_ctx)
                    if gt_unit and pred_unit and gt_unit != pred_unit:
                        continue
                    if gt_val == 0:
                        if pred_val == 0:
                            matched += 1
                            break
                    elif abs(gt_val - pred_val) / abs(gt_val) <= tolerance:
                        matched += 1
                        break
            return 1.0 if matched == len(gt_numbers) else 0.0
        else:
            # Single number comparison
            gt_val, gt_ctx = gt_numbers[0]
            gt_unit = detect_unit(gt_ctx)
            should_filter_years = not is_year(gt_val)

            for pred_val, pred_ctx in pred_numbers:
                if should_filter_years and is_year(pred_val):
                    continue
                pred_unit = detect_unit(pred_ctx)
                if gt_unit and pred_unit and gt_unit != pred_unit:
                    continue
                if gt_val == 0:
                    if pred_val == 0:
                        return 1.0
                    continue
                if abs(gt_val - pred_val) / abs(gt_val) <= tolerance:
                    return 1.0
            return 0.0

    # Case 2: Text comparison (case-insensitive substring match)
    gt_clean = re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)', '', ground_truth.strip().lower())).strip()
    pred_clean = re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)', '', predicted.strip().lower())).strip()

    if gt_clean in pred_clean or gt_clean == pred_clean:
        return 1.0
    return 0.0
