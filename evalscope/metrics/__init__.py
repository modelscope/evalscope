# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .aggregators.aggregators import ClippedMean, Mean, MeanPassAtK, MeanPassHatK, MeanVoteAtK
    from .judge.llm_judge import DEFAULT_NUMERIC_SCORE_TEMPLATE, DEFAULT_PROMPT_TEMPLATE, LLMJudge
    from .math.parser import extract_answer, math_equal, strip_answer_string
    from .nlp.metrics import ExactMatch
    from .utils.functions import bleu_ngram_one_sample, exact_match, macro_mean, mean, micro_mean, simple_f1_score
    from .utils.rouge import compute_rouge_score, compute_rouge_score_one_sample, compute_rouge_score_one_sample_zh
    from .utils.text_normalizer import BasicTextNormalizer, ChineseTextNormalizer, EnglishTextNormalizer

else:
    _import_structure = {
        'utils.functions': [
            'bleu_ngram_one_sample',
            'exact_match',
            'macro_mean',
            'mean',
            'micro_mean',
            'simple_f1_score',
        ],
        'aggregators.aggregators': [
            'Mean',
            'ClippedMean',
            'MeanPassAtK',
            'MeanVoteAtK',
            'MeanPassHatK',
        ],
        'nlp.metrics': [
            'ExactMatch',
        ],
        'utils.rouge': [
            'compute_rouge_score_one_sample_zh',
            'compute_rouge_score',
            'compute_rouge_score_one_sample',
        ],
        'judge.llm_judge': [
            'LLMJudge',
            'DEFAULT_PROMPT_TEMPLATE',
            'DEFAULT_NUMERIC_SCORE_TEMPLATE',
        ],
        'math.parser': [
            'extract_answer',
            'math_equal',
            'strip_answer_string',
        ],
        'utils.text_normalizer': [
            'BasicTextNormalizer',
            'EnglishTextNormalizer',
            'ChineseTextNormalizer',
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
