# Copyright (c) Alibaba, Inc. and its affiliates.
# Following official AGIEval evaluation: https://github.com/ruixiangcui/AGIEval

# Dataset classification following official AGIEval src/dataset_loader.py
ENGLISH_QA = [
    'aqua-rat', 'logiqa-en', 'lsat-ar', 'lsat-lr', 'lsat-rc', 'sat-math', 'sat-en', 'sat-en-without-passage',
    'gaokao-english'
]
CHINESE_QA = [
    'logiqa-zh', 'gaokao-chinese', 'gaokao-geography', 'gaokao-history', 'gaokao-biology', 'gaokao-chemistry',
    'gaokao-physics', 'gaokao-mathqa', 'jec-qa-kd', 'jec-qa-ca'
]
ENGLISH_CLOZE = ['math']
CHINESE_CLOZE = ['gaokao-mathcloze']
MULTI_CHOICE = ['jec-qa-kd', 'jec-qa-ca', 'gaokao-physics']

ALL_SUBSETS = ENGLISH_QA + CHINESE_QA + ENGLISH_CLOZE + CHINESE_CLOZE


def is_english_qa(subset: str) -> bool:
    return subset in ENGLISH_QA


def is_chinese_qa(subset: str) -> bool:
    return subset in CHINESE_QA


def is_multi_choice(subset: str) -> bool:
    return subset in MULTI_CHOICE


def is_qa(subset: str) -> bool:
    return is_english_qa(subset) or is_chinese_qa(subset)


def is_cloze(subset: str) -> bool:
    return subset in ENGLISH_CLOZE or subset in CHINESE_CLOZE
