# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, List

import jieba
from rouge_chinese import Rouge

from evalscope.metrics.utils.bundled_rouge_score import rouge_scorer
from evalscope.utils.logger import get_logger

logger = get_logger()


class DummyTokenizer:
    def tokenize(self, text: str):
        return text.split()


def is_contains_chinese(string: str) -> bool:
    return any('\u4e00' <= c <= '\u9fa5' for c in string)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_rouge_score_one_sample_zh(
    predict: List[str], reference: List[str], strict: bool = False
) -> Dict[str, float]:
    if isinstance(predict, str) or isinstance(reference, str):
        raise ValueError(f'Expected list of strings, but got {type(predict)} and {type(reference)}')

    zh_scorer = Rouge()
    pair_scores = []
    for p, r in zip(predict, reference, strict=strict):
        p = ' '.join(jieba.cut(p)) if is_contains_chinese(p) else p
        r = ' '.join(jieba.cut(r)) if is_contains_chinese(r) else r

        try:
            score = zh_scorer.get_scores(p, r, ignore_empty=True)[0]
        except Exception as e:
            logger.warning(f'rouge score error: {p} {r} {e}')
            continue
        pair_scores.append(score)

    # Average per metric over the scored pairs instead of letting the last pair overwrite
    # the rest.  Keys stay present (0.0) when no pair scored so consumers indexing into the
    # result do not fail.
    result = dict()
    for prefix, key in (('Rouge-1', 'rouge-1'), ('Rouge-2', 'rouge-2'), ('Rouge-L', 'rouge-l')):
        for stat in ('r', 'p', 'f'):
            values = [score[key][stat] for score in pair_scores]
            result[f'{prefix}-{stat.upper()}'] = _mean(values)
    return result


def compute_rouge_score_one_sample(predict: List[str], reference: List[str], strict: bool = False) -> Dict[str, float]:
    if isinstance(predict, str) or isinstance(reference, str):
        raise ValueError(f'Expected list of strings, but got {type(predict)} and {type(reference)}')

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=DummyTokenizer())
    pair_scores = []
    for p, r in zip(predict, reference, strict=strict):
        try:
            score = scorer.score(target=r, prediction=p)
        except Exception as e:
            logger.warning(f'rouge score error: {p} {r} {e}')
            continue
        pair_scores.append(score)

    result = dict()
    for suffix, key in (('rouge-1', 'rouge1'), ('rouge-2', 'rouge2'), ('rouge-l', 'rougeL')):
        for stat, attr in (('r', 'recall'), ('p', 'precision'), ('f', 'fmeasure')):
            values = [getattr(score[key], attr) for score in pair_scores]
            result[f'{suffix}-{stat}'] = _mean(values)
    return result
