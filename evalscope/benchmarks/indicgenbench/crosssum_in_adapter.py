# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import LANGUAGES, PAPER_URL, IndicGenBenchAdapter, language_name

ENGLISH = 'english'
INDIC = 'indic'

# One subset per summary language: English and the 29 Indic languages. The article is always in English.
SUBSET_LIST = ['en'] + LANGUAGES

# Table 15 of the paper; the one-shot example uses the same format, followed by its reference summary.
PROMPT_TEMPLATE = (
    'I will first show a news article in English and then provide a summary of it in the {language} language.'
    '\n\nSummarize the following article: {article}\nSummary:'
)


@register_benchmark(
    BenchmarkMeta(
        name='crosssum_in',
        evaluation_version='v1.0',
        pretty_name='CrossSum-IN',
        tags=[Tags.MULTI_LINGUAL],
        description="""
## Overview

CrossSum-IN evaluates summarization of English BBC news in English and 29 languages from India and nearby regions.

## Task Description

- **Task Type**: Cross-lingual Summarization
- **Input**: A news article in English
- **Output**: A one-sentence summary in the subset's language
- **Domain**: BBC news

## Key Features

- 12,463 test articles across 30 summary-language subsets.
- Uses the paper's prompt and one example from the same language's `train` split by default.

## Evaluation Notes

- Scores the first non-empty line using corpus SacreBLEU chrF (character order 6, word order 0, beta 2).
- The `indic` category's `macro_score` gives equal weight to each language. Its `score` weights by sample count.
- Articles are untruncated. The longest examples require a long context window.
- Licensed for non-commercial use under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
""",
        paper_url=PAPER_URL,
        dataset_id='google/IndicGenBench_crosssum_in',
        hf_dataset_id='google/IndicGenBench_crosssum_in',
        metric_list=['chrf'],
        primary_metric=MetricSelector(name='chrf', aggregation='official'),
        subset_list=SUBSET_LIST,
        few_shot_num=1,
        train_split='train',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class CrossSumINAdapter(IndicGenBenchAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {subset: ENGLISH if subset == 'en' else INDIC for subset in SUBSET_LIST}

    def subset_file(self, subset: str, split: str) -> str:
        if subset not in SUBSET_LIST:
            raise ValueError(f'Unknown CrossSum-IN subset `{subset}`. Expected one of: {", ".join(SUBSET_LIST)}.')
        return f'crosssum_english-{subset}_{split}.json'

    def check_record(self, subset: str, record: Dict[str, Any]) -> Optional[str]:
        if record['lang'] != subset:
            return f'holds a `{record["lang"]}` row, expected `{subset}`.'
        if not record['text'].strip() or not record['summary'].strip():
            return 'holds a row with an empty article or summary.'
        return None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        article = record['text'].strip()
        return Sample(
            input=article,
            target=record['summary'].strip(),
            metadata={
                'article': article,
                'language': record['lang'],
                'language_name': language_name(record['lang']),
                'source_url': record['source_url'],
                'target_url': record['target_url'],
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        return self.prompt_template.format(
            language=sample.metadata['language_name'],
            article=sample.metadata['article'],
        )
