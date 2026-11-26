# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import (
    LANGUAGE_NAMES,
    LANGUAGES,
    PAPER_URL,
    CorpusChrfMixin,
    IndicGenBenchAdapter,
    first_line,  # noqa: F401
    language_name,
)

EN_TO_XX = 'en-xx'
XX_TO_EN = 'xx-en'

# One subset per language and direction, named `<source>-<target>`: `en-hi` is English to Hindi.
SUBSET_LIST = [f'en-{lang}' for lang in LANGUAGES] + [f'{lang}-en' for lang in LANGUAGES]

# Table 15 of the paper; the one-shot example uses the same format, followed by its reference.
PROMPT_TEMPLATE = 'Translate the following:\n\nTo {language}: {source}\nOutput:'


def subset_languages(subset: str) -> Tuple[str, str]:
    """Return the (source, target) language codes of a subset such as `en-hi` or `hi-en`."""
    source, _, target = subset.partition('-')
    if 'en' not in (source, target) or source == target or (source if target == 'en' else target) not in LANGUAGE_NAMES:
        raise ValueError(f'Unknown FLORES-IN subset `{subset}`. Expected one of: {", ".join(SUBSET_LIST)}.')
    return source, target


def subset_file(subset: str, split: str) -> str:
    """Name of the released file for a subset and split, e.g. `flores_en_hi_test.json`."""
    source, target = subset_languages(subset)
    return f'flores_{source}_{target}_{split}.json'


def subset_direction(subset: str) -> str:
    """`en-xx` for English to an Indic language, `xx-en` for the reverse."""
    return EN_TO_XX if subset_languages(subset)[0] == 'en' else XX_TO_EN


@register_benchmark(
    BenchmarkMeta(
        name='flores_in',
        evaluation_version='v1.0',
        pretty_name='FLORES-IN',
        tags=[Tags.MULTI_LINGUAL, Tags.MT],
        description="""
## Overview

FLORES-IN is the machine translation task of IndicGenBench (Google Research, ACL 2024). It extends the
FLORES-200 `devtest` sentences with human translations into 29 Indic languages, many of them low-resource,
and evaluates translation in both directions between English and each language.

## Task Description

- **Task Type**: Machine Translation
- **Input**: A sentence in English (`en-xx` subsets) or in an Indic language (`xx-en` subsets)
- **Output**: Its translation into the other language
- **Languages**: Assamese, Awadhi, Bengali, Bhojpuri, Bodo, Chhattisgarhi, Garhwali, Gujarati, Haryanvi, Hindi,
  Kannada, Konkani, Maithili, Malvi, Malayalam, Manipuri, Marathi, Marwari, Nepali, Odia, Pashto, Punjabi,
  Rajasthani, Sanskrit, Santali, Tamil, Telugu, Tibetan and Urdu, in 13 scripts

## Key Features

- 58 subsets, one per language and direction, named `<source>-<target>` (e.g. `en-hi`, `hi-en`)
- Each subset evaluates all 1,012 sentences of its released `test` file (FLORES `devtest`), 58,696 in total
- The `dev` file of the same subset (997 sentences, FLORES `dev`) provides the in-context examples
- Results are grouped by direction: `en-xx` (English to Indic) and `xx-en` (Indic to English)

## Evaluation Notes

- Default configuration uses **1-shot** evaluation, the paper's main setting. The example is the first `dev`
  sentence of the same subset, so every model sees the same prompt; set `few_shot_num=0` for zero-shot
- Prompt (Table 15 of the paper), with the English language names of its Table 7:
  `Translate the following:\\n\\nTo {language}: {source}\\nOutput:`. The example uses the same format followed by
  ` {reference}`, then a blank line before the test sentence
- The first non-empty line of the output is scored; use greedy decoding (`temperature=0`) as in the paper
- Metric: **chrF** from SacreBLEU with default settings (character order 6, word order 0, beta 2), computed at
  the corpus level per subset. Each direction's score is the macro-average of its 29 subsets, reported as the
  `macro_score` of the `en-xx` and `xx-en` categories; all subsets have the same size, so the category `score`
  is the same number
- Each sample also records its sentence-level chrF for inspection; only the corpus score is reported
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), from FLORES
""",
        paper_url=PAPER_URL,
        dataset_id='google/IndicGenBench_flores_in',
        hf_dataset_id='google/IndicGenBench_flores_in',
        metric_list=['chrf'],
        primary_metric=MetricSelector(name='chrf', aggregation='official'),
        subset_list=SUBSET_LIST,
        few_shot_num=1,
        train_split='dev',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class FloresINAdapter(CorpusChrfMixin, IndicGenBenchAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {subset: subset_direction(subset) for subset in SUBSET_LIST}

    def subset_file(self, subset: str, split: str) -> str:
        return subset_file(subset, split)

    def check_record(self, subset: str, record: Dict[str, Any]) -> Optional[str]:
        source, target = subset_languages(subset)
        direction, lang = ('enxx', target) if source == 'en' else ('xxen', source)
        if record['translation_direction'] != direction or record['lang'] != lang:
            return f'holds a {record["lang"]}/{record["translation_direction"]} row, expected {lang}/{direction}.'
        return None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        lang = record['lang']
        source, target = ('en', lang) if record['translation_direction'] == 'enxx' else (lang, 'en')
        # About 1,200 sources and 170 references carry stray leading or trailing spaces; strip them so the prompt
        # reads `To Hindi: <source>` and `Output: <reference>`. chrF ignores whitespace, so scores are unaffected.
        source_text = record['source'].strip()
        return Sample(
            input=source_text,
            target=record['target'].strip(),
            metadata={
                'source': source_text,
                'source_language': source,
                'target_language': target,
                'target_language_name': language_name(target),
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        return self.prompt_template.format(
            language=sample.metadata['target_language_name'],
            source=sample.metadata['source'],
        )
