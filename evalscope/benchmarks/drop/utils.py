import numpy as np
import re
import string
from typing import List

_ARTICLES = re.compile(r'\b(a|an|the)\b', re.UNICODE)


def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_articles(text):
    return _ARTICLES.sub(' ', text)


def _white_space_fix(text):
    return ' '.join(text.split())


def _remove_punc(text):
    exclude = set(string.punctuation)
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in exclude)
    else:
        return text


def _fix_number(text):
    return str(float(text)) if _is_number(text) else text


def _tokenize(text):
    return re.split(' |-', text)


def _normalize(answer):
    tokens = [
        _white_space_fix(_remove_articles(_fix_number(_remove_punc(token.lower())))) for token in _tokenize(answer)
    ]
    tokens = [token for token in tokens if token.strip()]
    normalized = ' '.join(tokens).strip()
    return normalized


def _compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = ((2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0)
    return f1


def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = {word for word in gold_bag if _is_number(word)}
    predicted_numbers = {word for word in predicted_bag if _is_number(word)}
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _align_bags(predicted, gold):
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    from scipy.optimize import linear_sum_assignment

    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def parse_answer(answer):
    # NOTE: Everything is returned as a tuple for uniformity and hashability.
    if answer['number'] != '':
        return (str(answer['number']), )
    if answer['spans'] != []:
        return tuple(answer['spans'])
    return (' '.join([answer['date']['day'], answer['date']['month'], answer['date']['year']]).strip(), )


def _get_gold_answers(input_d: dict) -> List[str]:
    """
    Parse the raw input labels (gold).
    """

    def _flatten_validated_answers(validated_answers: dict) -> List[dict]:
        """
        Flatten the validated_answers structure into a list of answer dictionaries.

        Expected input:
            validated_answers: {
                'number': [...],
                'date':   [...],
                'spans':  [...]
            }

        Each returned dict has keys: 'number', 'date', 'spans'.
        If the input lists have different lengths, iteration stops at the shortest.
        """
        # Safely read lists from the input dict (default to empty lists)
        numbers = validated_answers.get('number', [])
        dates = validated_answers.get('date', [])
        spans = validated_answers.get('spans', [])

        # Ensure we only iterate as far as the shortest sequence to avoid IndexError
        length = min(len(numbers), len(dates), len(spans))

        flattened: List[dict] = []
        for num, date, sp in zip(numbers[:length], dates[:length], spans[:length]):
            flattened.append({'number': num, 'date': date, 'spans': sp})
        return flattened

    answers = []
    answers_set = set()
    candidates = [input_d['answer']] + _flatten_validated_answers(input_d['validated_answers'])
    for candidate in candidates:
        answer = parse_answer(candidate)
        if answer in answers_set:
            continue
        answers_set.add(answer)
        answers.append(answer)
    return answers
