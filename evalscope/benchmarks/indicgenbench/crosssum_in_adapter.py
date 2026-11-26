# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import LANGUAGES, PAPER_URL, CorpusChrfMixin, IndicGenBenchAdapter, language_name

ENGLISH = 'english'
INDIC = 'indic'

# One subset per summary language: English and the 29 Indic languages. The article is always in English.
SUBSET_LIST = ['en'] + LANGUAGES

# Table 15 of the paper; the one-shot example uses the same format, followed by its reference summary.
PROMPT_TEMPLATE = (
    'I will first show a news article in English and then provide a summary of it in the {language} language.'
    '\n\nSummarize the following article: {article}\nSummary:'
)


def subset_file(subset: str, split: str) -> str:
    """Name of the released file for a subset and split, e.g. `crosssum_english-hi_test.json`."""
    if subset not in SUBSET_LIST:
        raise ValueError(f'Unknown CrossSum-IN subset `{subset}`. Expected one of: {", ".join(SUBSET_LIST)}.')
    return f'crosssum_english-{subset}_{split}.json'


@register_benchmark(
    BenchmarkMeta(
        name='crosssum_in',
        evaluation_version='v1.0',
        pretty_name='CrossSum-IN',
        tags=[Tags.MULTI_LINGUAL],
        description="""
## Overview

CrossSum-IN is the cross-lingual summarization task of IndicGenBench (Google Research, ACL 2024). Built on
CrossSum, it pairs English BBC news articles with human-written summaries in 29 Indic languages, many of them
low-resource, and in English.

## Task Description

- **Task Type**: Cross-lingual Summarization
- **Input**: A news article in English
- **Output**: A one-sentence summary in the subset's language
- **Languages**: Assamese, Awadhi, Bengali, Bhojpuri, Bodo, Chhattisgarhi, Garhwali, Gujarati, Haryanvi, Hindi,
  Kannada, Konkani, Maithili, Malvi, Malayalam, Manipuri, Marathi, Marwari, Nepali, Odia, Pashto, Punjabi,
  Rajasthani, Sanskrit, Santali, Tamil, Telugu, Tibetan and Urdu, plus English

## Key Features

- 30 subsets named by summary language (`en`, `hi`, `sat`, ...), each evaluated on its released `test` file
  (104 to 632 articles per language, 12,463 in total)
- The `train` file of the same language (100 articles, 99 for `gbm`) provides the in-context examples
- Results are grouped into `indic` (the 29 Indic languages) and `english`, which the paper reports separately

## Evaluation Notes

- Default configuration uses **1-shot** evaluation, the paper's main setting. The example is the first `train`
  article of the same language, so every model sees the same prompt; set `few_shot_num=0` for zero-shot
- Prompt (Table 15 of the paper), with the English language names of its Table 7:
  `I will first show a news article in English and then provide a summary of it in the {language} language.\\n\\n`
  `Summarize the following article: {article}\\nSummary:`. The example uses the same format followed by
  ` {summary}`, then a blank line before the test article
- Articles are not truncated, as in the paper: they average about 4,200 characters, 5% exceed 8,200, and 28
  (0.2%) exceed 32,768 characters (roughly 8k tokens), so a model with a short context window cannot take those
- The first non-empty line of the output is scored; use greedy decoding (`temperature=0`) as in the paper
- Metric: **chrF** from SacreBLEU with default settings (character order 6, word order 0, beta 2), computed at
  the corpus level per language. The Indic score is the macro-average of the 29 Indic languages, reported as the
  `macro_score` of the `indic` category (languages differ in size, so the category `score`, weighted by sample
  count, is a different number)
- Each sample also records its sentence-level chrF for inspection; only the corpus score is reported
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), from CrossSum: non-commercial
  use only
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
class CrossSumINAdapter(CorpusChrfMixin, IndicGenBenchAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {subset: ENGLISH if subset == 'en' else INDIC for subset in SUBSET_LIST}

    def subset_file(self, subset: str, split: str) -> str:
        return subset_file(subset, split)

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
