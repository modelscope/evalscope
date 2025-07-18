# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .completion_parsers import ResponseParser, lmsys_parser, ranking_parser
    from .llm_judge import DEFAULT_NUMERIC_SCORE_TEMPLATE, DEFAULT_PROMPT_TEMPLATE, LLMJudge
    from .math_parser import extract_answer, math_equal, strip_answer_string
    from .metrics import (bleu_ngram_one_sample, exact_match, macro_mean, mean, micro_mean, simple_f1_score,
                          weighted_mean)
    from .named_metrics import Metric, metric_registry
    from .rouge_metric import compute_rouge_score, compute_rouge_score_one_sample, compute_rouge_score_one_sample_zh

else:
    _import_structure = {
        'metrics': [
            'bleu_ngram_one_sample',
            'exact_match',
            'macro_mean',
            'mean',
            'micro_mean',
            'simple_f1_score',
            'weighted_mean',
        ],
        'named_metrics': [
            'Metric',
            'metric_registry',
        ],
        'rouge_metric': [
            'compute_rouge_score_one_sample_zh',
            'compute_rouge_score',
            'compute_rouge_score_one_sample',
        ],
        'llm_judge': [
            'LLMJudge',
            'DEFAULT_PROMPT_TEMPLATE',
            'DEFAULT_NUMERIC_SCORE_TEMPLATE',
        ],
        'math_parser': [
            'extract_answer',
            'math_equal',
            'strip_answer_string',
        ],
        'completion_parsers': [
            'ResponseParser',
            'lmsys_parser',
            'ranking_parser',
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
