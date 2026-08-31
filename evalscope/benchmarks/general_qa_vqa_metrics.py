# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, Tuple

METRIC_SCORE_KEYS: Dict[str, Tuple[str, ...]] = {
    'Rouge': (
        'Rouge-1-R',
        'Rouge-1-P',
        'Rouge-1-F',
        'Rouge-2-R',
        'Rouge-2-P',
        'Rouge-2-F',
        'Rouge-L-R',
        'Rouge-L-P',
        'Rouge-L-F',
    ),
    'BLEU': ('bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'),
}


class MetricScoringError(Exception):
    """Raised when a metric returns an incomplete General QA/VQA score schema."""


def calculate_metric_score(metric: str, prediction: str, reference: str) -> Dict[str, float]:
    """Calculate one General QA/VQA metric and validate its score schema."""
    if metric == 'Rouge':
        from evalscope.metrics.utils.rouge import compute_rouge_score_one_sample_zh

        score = compute_rouge_score_one_sample_zh([prediction], [reference])
    elif metric == 'BLEU':
        from evalscope.metrics import bleu_ngram_one_sample

        score = bleu_ngram_one_sample(prediction, reference)
    else:
        raise ValueError(f'Unsupported General QA/VQA metric: {metric}')

    missing_keys = set(METRIC_SCORE_KEYS[metric]) - score.keys()
    if missing_keys:
        raise MetricScoringError(f'{metric} returned incomplete score schema: missing {sorted(missing_keys)}')
    return score
