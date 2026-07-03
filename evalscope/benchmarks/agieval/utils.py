# Copyright (c) Alibaba, Inc. and its affiliates.
# Following official AGIEval evaluation: https://github.com/ruixiangcui/AGIEval

import re
from typing import Any, Dict

# Dataset classification following official AGIEval src/dataset_loader.py
ENGLISH_QA = [
    'aqua-rat', 'logiqa-en', 'lsat-ar', 'lsat-lr', 'lsat-rc', 'sat-math', 'sat-en', 'sat-en-without-passage',
    'gaokao-english'
]
CHINESE_QA = [
    'logiqa-zh', 'gaokao-chinese', 'gaokao-geography', 'gaokao-history', 'gaokao-biology', 'gaokao-chemistry',
    'gaokao-physics', 'gaokao-mathqa', 'jec-qa-kd', 'jec-qa-ca'
]
ENGLISH_CLOZE = ['math']
CHINESE_CLOZE = ['gaokao-mathcloze']
MULTI_CHOICE = ['jec-qa-kd', 'jec-qa-ca', 'gaokao-physics']

ALL_SUBSETS = ENGLISH_QA + CHINESE_QA + ENGLISH_CLOZE + CHINESE_CLOZE


def is_english_qa(subset: str) -> bool:
    return subset in ENGLISH_QA


def is_chinese_qa(subset: str) -> bool:
    return subset in CHINESE_QA


def is_english_cloze(subset: str) -> bool:
    return subset in ENGLISH_CLOZE


def is_chinese_cloze(subset: str) -> bool:
    return subset in CHINESE_CLOZE


def is_multi_choice(subset: str) -> bool:
    return subset in MULTI_CHOICE


def is_qa(subset: str) -> bool:
    return is_english_qa(subset) or is_chinese_qa(subset)


def is_cloze(subset: str) -> bool:
    return is_english_cloze(subset) or is_chinese_cloze(subset)


# --- Prompt building (following official dataset_loader.py convert_zero_shot) ---


def build_prompt(record: Dict[str, Any], subset: str) -> str:
    """Build prompt following official AGIEval zero-shot format."""
    passage = record.get('passage') or ''
    question = record['question']
    options = record.get('options')

    if is_english_qa(subset):
        option_str = ' '.join(options) if options else ''
        count = len(options) if options else 5
        option_string = 'ABCDEFG'
        return (
            f'{passage}Q: {question} '
            f'Answer Choices: {option_str}\n'
            f'A: Among A through {option_string[count - 1]}, the answer is'
        )
    elif is_chinese_qa(subset):
        option_str = ' '.join(options) if options else ''
        count = len(options) if options else 4
        option_string = 'ABCDEFG'
        return (f'{passage}问题：{question} '
                f'选项：{option_str}\n'
                f'答案：从A到{option_string[count - 1]}, 我们应选择')
    elif is_english_cloze(subset):
        return f'{passage}Q: {question}\nA: The answer is'
    elif is_chinese_cloze(subset):
        return f'{passage}问题：{question}\n答案：'
    else:
        return f'{passage}{question}'


# --- Answer extraction (following official post_process.py) ---


def extract_single_answer_en(prediction: str) -> str:
    """Extract single MCQ answer from English response."""
    pattern = r'answer is .*?([A-G])'
    match = re.search(pattern, prediction)
    if match:
        return match.group(1)
    return find_first_capital_letter(prediction)


def extract_single_answer_zh(prediction: str) -> str:
    """Extract single MCQ answer from Chinese response."""
    pattern = r'答案是.*?([A-G])'
    match = re.search(pattern, prediction)
    if match:
        return match.group(1)
    return find_first_capital_letter(prediction)


def find_first_capital_letter(answer: str) -> str:
    """Find first capital letter A-F in response."""
    letter_set = {'A', 'B', 'C', 'D', 'E', 'F'}
    for c in answer:
        if c in letter_set:
            return c
    return ''


def extract_multiple_answers(prediction: str) -> str:
    """Extract multiple MCQ answers."""
    pattern = r'\(*([A-F])\)*'
    matches = re.findall(pattern, prediction)
    if matches:
        return ''.join(sorted(set(matches)))
    return ''


def extract_math_answer(prediction: str) -> str:
    """Extract math answer from response."""
    from evalscope.metrics.math.parser import extract_answer
    return extract_answer(prediction)


# --- Scoring (following official evaluation.py) ---


def score_single_choice(prediction: str, label: str) -> float:
    """Score single-choice MCQ: exact letter match."""
    return 1.0 if prediction.strip().upper() == label.strip().upper() else 0.0


def score_multiple_choice(prediction: str, label) -> float:
    """Score multi-choice MCQ: set comparison (official evaluate_single_sample)."""
    pred_set = {c for c in prediction.strip().upper() if c.isalpha()}
    if isinstance(label, list):
        label_set = {l.strip().upper() for l in label}
    elif isinstance(label, str):
        label_set = {c for c in label.strip().upper() if c.isalpha()}
    else:
        label_set = set()
    return 1.0 if pred_set == label_set else 0.0


def score_math(prediction: str, label: str) -> float:
    """Score math answer: equivalence check (official is_equiv)."""
    from evalscope.metrics.math.parser import math_equal
    return 1.0 if math_equal(prediction, label) else 0.0
