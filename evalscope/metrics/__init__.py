# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.metrics.metrics import (bleu_ngram_one_sample, exact_match, macro_mean, mean, micro_mean,
                                       simple_f1_score, weighted_mean)
from evalscope.metrics.named_metrics import *
from evalscope.metrics.rouge_metric import compute_rouge_score_one_sample_zh
