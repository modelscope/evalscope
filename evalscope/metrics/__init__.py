# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.metrics.metrics import bleu_ngram_one_sample, exact_match, weighted_mean
from evalscope.metrics.rouge_metric import compute_rouge_score_one_sample_zh

WeightedAverageAccuracy = {'name': 'WeightedAverageAccuracy', 'object': weighted_mean}
WeightedAverageBLEU = {'name': 'WeightedAverageBLEU', 'object': weighted_mean}
Pass1 = {'name': 'Pass@1', 'object': weighted_mean}
