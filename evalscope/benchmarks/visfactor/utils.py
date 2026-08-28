# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

BOOL_CATEGORY_IDS = frozenset({'CF1', 'CF2', 'MV1', 'MV2', 'MV3', 'P3', 'RL2', 'S1', 'S2', 'SS2', 'VZ1', 'VZ2'})
STRING_LIST_CATEGORY_IDS = frozenset({'CS1', 'CS2', 'CS3'})
NUMERIC_CATEGORY_IDS = frozenset({'CF3', 'I3', 'MA1', 'SS3', 'VZ3'})

_JSON_ANSWER_PATTERN = re.compile(r'\{\s*"answer"\s*:\s*(.+?)\s*\}')
_ADDITIONAL_PATTERN = re.compile(r'<ADDITIONAL_(\d+)>')
_NUMBER_PATTERN = re.compile(r'\d+')


def render_question(question: str, additional: str) -> str:
    """Render the official prompt by resolving line breaks and additional fields."""
    text = question.replace('<br>', '\n')
    if not additional:
        return text

    values = additional.replace('<br>', '\n').split(';')

    def replace(match: re.Match) -> str:
        index = int(match.group(1))
        return values[index] if index < len(values) else match.group(0)

    return _ADDITIONAL_PATTERN.sub(replace, text)


def extract_json_answer(prediction: str) -> str:
    """Extract the last JSON-shaped answer using the official VisFactor rule."""
    matches = list(_JSON_ANSWER_PATTERN.finditer(prediction))
    if not matches:
        return ''

    answer = matches[-1].group(1).strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {'"', "'"}:
        answer = answer[1:-1]
    return answer


def normalize_prediction(category_id: str, prediction: str, additional: str = '') -> str:
    """Normalize one model response according to its VisFactor subtest."""
    answer = extract_json_answer(prediction)

    if category_id in BOOL_CATEGORY_IDS or (category_id == 'VZ3' and not additional.isdigit()):
        if answer.lower() in {'t', 'y', '1', 'true', 'yes'}:
            return 'T'
        if answer.lower() in {'f', 'n', '0', 'false', 'no'}:
            return 'F'
        return ''

    if category_id in STRING_LIST_CATEGORY_IDS:
        return answer

    if category_id not in NUMERIC_CATEGORY_IDS:
        return ''

    candidate = answer or prediction
    if category_id == 'VZ3':
        return next((character for character in reversed(candidate) if character.isupper()), '')

    numbers = _NUMBER_PATTERN.findall(candidate)
    if category_id == 'CF3':
        return '' if len(numbers) < 2 else f'({numbers[-2]}, {numbers[-1]})'
    return numbers[-1] if numbers else ''


def score_prediction(category_id: str, prediction: str, reference: str, additional: str = '') -> Tuple[str, float]:
    """Return the normalized prediction and official row-level correctness."""
    normalized = normalize_prediction(category_id, prediction, additional)
    if category_id in STRING_LIST_CATEGORY_IDS:
        accepted_answers = [answer.strip().lower() for answer in reference.split(',')]
        correct = normalized.strip().lower() in accepted_answers
    else:
        correct = normalized == reference
    return normalized, float(correct)


def aggregate_item_accuracy(rows: Iterable[Tuple[str, int, float]]) -> Dict[str, Tuple[float, int]]:
    """Apply logical AND within each item and return per-category accuracy and item count."""
    grouped_rows: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for category_id, eval_index, score in rows:
        grouped_rows[(category_id, eval_index)].append(score)

    category_items: Dict[str, List[float]] = defaultdict(list)
    for (category_id, _), scores in grouped_rows.items():
        category_items[category_id].append(float(all(scores)))

    return {
        category_id: (sum(scores) / len(scores), len(scores))
        for category_id, scores in category_items.items()
        if scores
    }
