# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import LANGUAGE_NAMES, LANGUAGES, PAPER_URL, IndicGenBenchAdapter, language_name

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

FLORES-IN evaluates translation between English and 29 languages from India and nearby regions.

## Task Description

- **Task Type**: Machine Translation
- **Input**: A sentence in English (`en-xx` subsets) or in the paired language (`xx-en` subsets)
- **Output**: Its translation into the other language
- **Domain**: General-domain translation between English and 29 languages from India and nearby regions

## Key Features

- 58,696 test sentences across 58 subsets, one for each language and direction.
- Uses the paper's prompt and one example from the same subset's `dev` split by default.

## Evaluation Notes

- Scores the first non-empty line using corpus SacreBLEU chrF (character order 6, word order 0, beta 2).
- Each direction's `macro_score` gives equal weight to its 29 languages.
- Licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
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
class FloresINAdapter(IndicGenBenchAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {subset: subset_direction(subset) for subset in SUBSET_LIST}

    def subset_file(self, subset: str, split: str) -> str:
        source, target = subset_languages(subset)
        return f'flores_{source}_{target}_{split}.json'

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
