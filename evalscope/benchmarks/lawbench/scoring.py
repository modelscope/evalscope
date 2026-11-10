# Copyright (c) Alibaba, Inc. and its affiliates.
# Adapted from OpenCompass/LawBench (Apache-2.0) for per-sample scoring and explicit denominators.
import math
import re
from collections import Counter
from typing import Dict, Hashable, Iterable

from .labels import CHARGE_LABELS, DISPUTE_LABELS, EVENT_LABELS, MARITAL_LABELS, TOPIC_LABELS

ENTITY_TYPES = (
    '犯罪嫌疑人',
    '受害人',
    '被盗货币',
    '物品价值',
    '盗窃获利',
    '被盗物品',
    '作案工具',
    '时间',
    '地点',
    '组织机构',
)


def _set_f1(prediction: Iterable[Hashable], reference: Iterable[Hashable]) -> float:
    pred, gold = set(prediction), set(reference)
    return 2 * len(pred & gold) / (len(pred) + len(gold)) if pred or gold else 0.0


def _character_f1(prediction: str, reference: str) -> float:
    pred = [c for c in prediction.lower() if c.isdigit() or c.isalpha()]
    gold = [c for c in reference.lower() if c.isdigit() or c.isalpha()]
    if not pred or not gold:
        return float(pred == gold)
    return 2 * sum((Counter(pred) & Counter(gold)).values()) / (len(pred) + len(gold))


def _choice(prediction: str, reference: str, labels: Iterable[str]) -> Dict[str, float]:
    selected = {label for label in labels if label in prediction}
    return {'score': float(selected == {reference}), 'abstention_rate': float(not selected)}


def _multi_label(prediction: str, references: Iterable[str], labels: Iterable[str]) -> Dict[str, float]:
    selected = {label for label in labels if label in prediction}
    return {'score': _set_f1(selected, references), 'abstention_rate': float(not selected)}


def _rouge_l(prediction: str, reference: str) -> float:
    import jieba
    from rouge_chinese import Rouge

    pred = ' '.join(jieba.cut(prediction))
    gold = ' '.join(jieba.cut(reference))
    return Rouge().get_scores(pred if pred.strip() else '无内容', gold)[0]['rouge-l']['f']


def _entities(prediction: str, reference: str) -> Dict[str, float]:
    gold = {part.split(':')[0]: part.split(':')[1] for part in reference.split(';')} if reference.strip() else {}
    pred = {}
    for entity_type in ENTITY_TYPES:
        for match in re.finditer(entity_type, prediction):
            end = match.end()
            if end >= len(prediction) - 2 or prediction[end] not in (':', '：'):
                continue
            entity = re.split(r'\n| ', prediction[end + 1 :].strip())[0].strip()
            if entity not in ('无', '未提及'):
                pred[entity_type] = entity
    if not gold:
        return {'score': float(not pred), 'abstention_rate': 0.0}
    matched = sum(_character_f1(value, gold[key]) for key, value in pred.items() if key in gold)
    precision = matched / len(pred) if pred else 0.0
    recall = matched / len(gold)
    return {
        'score': 2 * precision * recall / (precision + recall + 1e-10),
        # Upstream misspells this key as anstention_rate; retain the computed value.
        'abstention_rate': float(not pred),
    }


def _article(prediction: str, reference: str) -> Dict[str, float]:
    import cn2an

    if not reference.startswith('法条:刑法第') or not reference.endswith('条'):
        raise ValueError(f'Invalid LawBench article reference: {reference!r}')
    gold = {int(item) for item in reference.replace('法条:刑法第', '').replace('条', '').split('、')}
    pred = set()
    for chunk in prediction.split('、'):
        chunk = re.sub(r'第(.*?)款', '', chunk.replace('万元', '元'))
        chunk = re.sub(r'第(.*?)条', lambda match: match.group(1), chunk)
        numbers = re.findall(r'\d+', cn2an.transform(chunk, 'cn2an'))
        if numbers:
            pred.add(int(numbers[0]))
    return {'score': _set_f1(pred, gold), 'abstention_rate': float(not pred)}


def _imprisonment(prediction: str, reference: str) -> Dict[str, float]:
    import cn2an

    if '死刑' in reference or '无期' in reference:
        return {'abstention_rate': 0.0}
    if not reference.startswith('刑期:') or not reference.endswith('个月'):
        raise ValueError(f'Invalid LawBench imprisonment reference: {reference!r}')
    gold = int(reference.replace('刑期:', '').replace('个月', ''))
    prediction = cn2an.transform(prediction, 'cn2an')
    months = re.findall(r'(\d+)个月', prediction) + re.findall(r'(\d+)月', prediction)
    years = re.findall(r'(\d+)年', prediction)
    if not months and not years:
        return {'score': 0.0, 'abstention_rate': 1.0}
    pred = int(months[0]) if months else int(years[0]) * 12
    return {'score': 1 - abs(math.log(gold + 1) - math.log(pred + 1)) / math.log(216), 'abstention_rate': 0.0}


def score_response(task: str, prediction: str, reference: str) -> Dict[str, float]:
    """Apply the official row rule; omit score for officially skipped references.

    Abstention is returned only for tasks defining it. An omitted score still carries
    zero abstention so skipped references remain in the official abstention denominator.
    Task 2-1 requires corpus-level ChERRANT scoring instead of a row average.
    """
    if task in ('1-1', '2-7', '3-2', '3-8'):
        gold = reference.replace('答案:', '') if task == '1-1' else reference
        return {'score': _rouge_l(prediction, gold)}
    if task in ('1-2', '3-6', '2-8'):
        prefix = {'1-2': '正确答案：', '3-6': '正确答案:', '2-8': '[正确答案]'}[task]
        labels = 'ABCDE' if task == '2-8' else 'ABCD'
        if not reference.startswith(prefix) or reference[len(prefix) : len(prefix) + 1] not in tuple(labels):
            raise ValueError(f'Invalid LawBench choice reference: {reference!r}')
        return _choice(prediction, reference[len(prefix)], labels)
    if task == '2-2':
        gold = reference[7:-1]
        if gold == '赔偿':
            return {'abstention_rate': 0.0}
        if not reference.startswith('争议焦点类别：') or gold not in DISPUTE_LABELS:
            raise ValueError(f'Invalid LawBench dispute reference: {reference!r}')
        return _choice(prediction, gold, DISPUTE_LABELS)
    if task == '2-3':
        if not reference.startswith('类别:') or not reference.endswith('。'):
            raise ValueError(f'Invalid LawBench marital reference: {reference!r}')
        return _multi_label(prediction, reference[3:-1].split('、'), MARITAL_LABELS)
    if task == '2-4':
        return _choice(prediction, reference, TOPIC_LABELS)
    if task == '2-5':
        return {'score': _character_f1(prediction, reference.replace('回答:', ''))}
    if task == '2-6':
        return _entities(prediction, reference)
    if task == '2-9':
        return _multi_label(prediction, reference.split(';'), EVENT_LABELS)
    if task == '2-10':
        pred, gold = prediction.split(';'), reference.split(';')
        matched = sum(_character_f1(p, g) for p, g in zip(pred, gold))
        precision, recall = matched / len(pred), matched / len(gold)
        return {'score': 2 * precision * recall / (precision + recall + 1e-10)}
    if task == '3-1':
        return _article(prediction, reference)
    if task == '3-3':
        if not reference.startswith('罪名:'):
            raise ValueError(f'Invalid LawBench charge reference: {reference!r}')
        return _multi_label(prediction, reference.replace('罪名:', '').split(';'), CHARGE_LABELS)
    if task in ('3-4', '3-5'):
        return _imprisonment(prediction, reference)
    if task == '3-7':
        if not reference.startswith('上文涉及到的犯罪金额:') or not reference.endswith('元。'):
            raise ValueError(f'Invalid LawBench amount reference: {reference!r}')
        gold = float(reference.replace('上文涉及到的犯罪金额:', '').replace('元。', ''))
        pred = [float(number) for number in re.findall(r'\d+\.?\d*', prediction)]
        return {'score': float(gold in pred), 'abstention_rate': float(not pred)}
    raise ValueError(f'LawBench task {task!r} has no row scorer; task 2-1 requires corpus scoring.')
