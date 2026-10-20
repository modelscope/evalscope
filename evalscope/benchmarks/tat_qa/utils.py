# Scoring for TAT-QA, ported from the official evaluation in https://github.com/NExTplusplus/TAT-QA
# (tatqa_utils.py and tatqa_metric.py, MIT License, Copyright (c) 2021 Fengbin Zhu).
#
# The functions below keep the official behaviour, including how numbers are rounded and scaled before
# comparison, so that EM / F1 match `tatqa_eval.py` question by question.

import re
import string
from typing import List, Set, Tuple, Union

import numpy as np

SCALES = ['', 'thousand', 'million', 'billion', 'percent']

# `[ \t]*` (not `\s*`) after the colon keeps an empty answer line from capturing the next line.
_ANSWER_RE = re.compile(r'^\s*\**\s*(?:final\s+)?answer\s*\**\s*:[ \t]*(.*)$', re.IGNORECASE | re.MULTILINE)
_SCALE_RE = re.compile(r'^\s*\**\s*scale\s*\**\s*:[ \t]*(.*)$', re.IGNORECASE | re.MULTILINE)
SPAN_SEPARATOR = ' | '

# --- tatqa_utils.py ---------------------------------------------------------------------------------------


def scale_to_num(scale: str) -> float:
    scale = scale.lower()
    num = 1
    if 'hundred' in scale:
        num = 100
    elif 'thousand' in scale:
        num = 1000
    elif 'million' in scale:
        num = 1000000
    elif 'billion' in scale:
        num = 1000000000
    elif 'percent' in scale:
        num = 0.01
    return num


def extract_one_num_from_str(s):
    s = _clean_num(s)
    r_num = r'([+-]?\d+(\.\d+)?)|([+-]?\.\d+)'
    groups = re.findall(r_num, s)
    if len(groups) == 0:
        return None
    num = groups[0][0]
    if num == '':
        return None
    if '.' in num:
        return float(num)
    return int(num)


EXCLUDE_IN_NUM = '\'"\\$€£¥%(),[]'


def _clean_num(text: str) -> str:
    return ''.join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def is_number(text: str) -> bool:
    try:
        words = ' '.join([_clean_num(w) for w in text.split()]).split()
        if len(words) == 0:
            return False
        num = float(words[0])
        if np.isnan(num):
            return False
        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False
        return True
    except ValueError:
        return False


def negative_num_handle(x: str) -> int:
    """(134) -> -134"""
    if re.findall(r'(\([\d.\s]+\))', x.strip()):
        return -1
    return 1


def percent_num_handle(x: str) -> float:
    """12% -> 12/100"""
    if re.findall(r'([\d.\s]+%)', x.strip()):
        return 0.01
    return 1


def word_scale_handle(x: str) -> float:
    """1 million -> 1,000,000"""
    for one in re.finditer(r'([\d.]+\s?[a-zA-Z]+)', x):
        return scale_to_num(one.group(0).lower())
    return 1


def to_number(text: str):
    num = extract_one_num_from_str(text)
    scale_val = word_scale_handle(text)
    negative_flag = negative_num_handle(text)
    percent_flag = percent_num_handle(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None


def remove_articles(text: str) -> str:
    return re.sub(re.compile(r'\b(a|an|the)\b', re.UNICODE), ' ', text)


def white_space_fix(text: str) -> str:
    return ' '.join(text.split())


EXCLUDE = set(string.punctuation)


def remove_punc(text: str) -> str:
    if not is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    return text


def normalize_number(text: str) -> str:
    if is_number(text):
        return str(to_number(text))
    return text


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [
        white_space_fix(remove_articles(normalize_number(remove_punc(token.lower())))) for token in text.split(' ')
    ]
    parts = [part for part in parts if part.strip()]
    return ' '.join(parts).strip()


# --- tatqa_metric.py --------------------------------------------------------------------------------------


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    raw_spans = answer if isinstance(answer, (list, tuple)) else [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> np.ndarray:
    """Best 1-1 alignment between gold and predicted bags, then the F1 of each gold bag."""
    from scipy.optimize import linear_sum_assignment

    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = 1.0 if not predicted_bag else intersection / float(len(predicted_bag))
    recall = 1.0 if not gold_bag else intersection / float(len(gold_bag))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def get_metrics(predicted: Union[str, List[str]], gold: Union[str, List[str]]) -> Tuple[float, float]:
    """Exact match and DROP-style F1 of a prediction against one gold answer."""
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0
    f1 = round(np.mean(_align_bags(predicted_bags[1], gold_bags[1])), 2)
    return exact_match, f1


def extract_gold_answers(qa_annotation: dict) -> Tuple[str, List[str], str]:
    answer_type, scale = qa_annotation['answer_type'], qa_annotation['scale']
    answer_content = qa_annotation['answer']
    if answer_type in ['multi-span', 'span']:
        assert isinstance(answer_content, list), answer_content
        gold_answers = answer_content
    elif answer_type == 'count':
        gold_answers = [str(int(answer_content))]
    else:
        gold_answers = [str(answer_content)]
    return answer_type, gold_answers, scale


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    scores = [metric_fn(pred, ground_truth) for pred in predictions for ground_truth in ground_truths]
    if not scores:
        return 0, 0
    return max(scores)


def get_answer_str(answers: list, scale: str) -> List[str]:
    """Sort the spans, render numbers as ``%.4f`` after applying the scale and join them with spaces."""
    ans_temp = []
    for ans in sorted(answers):
        ans_str = str(ans)
        if is_number(ans_str):
            ans_num = to_number(ans_str)
            if ans_num is None:
                if scale:
                    ans_str = ans_str + ' ' + str(scale)
            else:
                if '%' in ans_str:  # the answer itself is a percentage
                    ans_str = '%.4f' % ans_num
                else:
                    ans_str = '%.4f' % (round(ans_num, 2) * scale_to_num(scale))
        else:
            if scale:
                ans_str = ans_str + ' ' + str(scale)
        ans_temp.append(ans_str)
    return [' '.join(ans_temp)]


def add_percent_pred(prediction_strings: List[str], pred_scale: str, pred: list) -> List[str]:
    """Also accept a bare number such as 0.2342 for a gold answer of 23.42 with scale ``percent``."""
    if len(pred) > 1:
        return prediction_strings
    pred_str = str(pred[0])
    if not pred_scale and '%' not in pred_str and is_number(pred_str):
        pred_num = to_number(pred_str)
        if pred_num is None:
            return prediction_strings
        prediction_strings.append('%.4f' % pred_num)
    return prediction_strings


def score_question(ground_truth: dict, prediction: Union[str, List[str]], pred_scale: str = '') -> Tuple[float, float]:
    """EM and F1 of one question, as accumulated by the official ``TaTQAEmAndF1.__call__``."""
    if not prediction:
        return 0.0, 0.0
    gold_type, gold_answer, gold_scale = extract_gold_answers(ground_truth)
    if not gold_answer:
        return 0.0, 0.0
    ground_truth_answer_strings = get_answer_str(gold_answer, gold_scale)
    prediction = prediction if isinstance(prediction, list) else [prediction]
    prediction_strings = get_answer_str(prediction, pred_scale)
    prediction_strings = add_percent_pred(prediction_strings, pred_scale, prediction)
    exact_match, f1_score = metric_max_over_ground_truths(get_metrics, prediction_strings, ground_truth_answer_strings)
    if gold_type in ['arithmetic', 'count']:
        f1_score = exact_match
    return float(exact_match), float(f1_score)


# --- model output parsing ---------------------------------------------------------------------------------


def parse_response(text: str) -> Tuple[List[str], str]:
    """Read the last ``Answer:`` and ``Scale:`` lines of a response.

    Multiple spans are separated by ``|``. An unknown or missing scale counts as no scale.
    """
    answers = _ANSWER_RE.findall(text or '')
    if not answers:
        return [], ''
    spans = [span.strip().strip('*').strip() for span in answers[-1].split('|')]
    spans = [span for span in spans if span]
    scales = _SCALE_RE.findall(text or '')
    scale = scales[-1].strip().strip('*').strip().rstrip('.').lower() if scales else ''
    return spans, scale if scale in SCALES else ''
