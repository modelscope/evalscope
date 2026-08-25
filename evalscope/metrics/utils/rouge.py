# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import defaultdict
from statistics import mean
from typing import List

import jieba
from rouge_chinese import Rouge
from tqdm import tqdm

from evalscope.constants import MetricsConstant
from evalscope.metrics.utils.bundled_rouge_score import rouge_scorer
from evalscope.utils.logger import get_logger

logger = get_logger()


class DummyTokenizer:

    def tokenize(self, text: str):
        return text.split()


def is_contains_chinese(string: str) -> bool:
    return any('\u4e00' <= c <= '\u9fa5' for c in string)


def compute_rouge_score_one_sample_zh(predict: List[str], reference: List[str], strict: bool = False):
    if isinstance(predict, str) or isinstance(reference, str):
        raise ValueError(f'Expected list of strings, but got {type(predict)} and {type(reference)}')

    result = dict()
    zh_scorer = Rouge()
    for p, r in zip(predict, reference, strict=strict):
        p = ' '.join(jieba.cut(p)) if is_contains_chinese(p) else p
        r = ' '.join(jieba.cut(r)) if is_contains_chinese(r) else r

        try:
            score = zh_scorer.get_scores(p, r, ignore_empty=True)[0]
        except Exception as e:
            logger.warning(f'rouge score error: {p} {r} {e}')
            continue
        result['Rouge-1-R'] = score['rouge-1']['r']
        result['Rouge-1-P'] = score['rouge-1']['p']
        result['Rouge-1-F'] = score['rouge-1']['f']
        result['Rouge-2-R'] = score['rouge-2']['r']
        result['Rouge-2-P'] = score['rouge-2']['p']
        result['Rouge-2-F'] = score['rouge-2']['f']
        result['Rouge-L-R'] = score['rouge-l']['r']
        result['Rouge-L-P'] = score['rouge-l']['p']
        result['Rouge-L-F'] = score['rouge-l']['f']

    return result


def compute_rouge_score_one_sample(predict: List[str], reference: List[str], strict: bool = False):
    if isinstance(predict, str) or isinstance(reference, str):
        raise ValueError(f'Expected list of strings, but got {type(predict)} and {type(reference)}')

    result = dict()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=DummyTokenizer())
    for p, r in zip(predict, reference, strict=strict):
        try:
            score = scorer.score(target=r, prediction=p)
        except Exception as e:
            logger.warning(f'rouge score error: {p} {r} {e}')
            continue
        result['rouge-1-r'] = score['rouge1'].recall
        result['rouge-1-p'] = score['rouge1'].precision
        result['rouge-1-f'] = score['rouge1'].fmeasure
        result['rouge-2-r'] = score['rouge2'].recall
        result['rouge-2-p'] = score['rouge2'].precision
        result['rouge-2-f'] = score['rouge2'].fmeasure
        result['rouge-l-r'] = score['rougeL'].recall
        result['rouge-l-p'] = score['rougeL'].precision
        result['rouge-l-f'] = score['rougeL'].fmeasure

    return result
