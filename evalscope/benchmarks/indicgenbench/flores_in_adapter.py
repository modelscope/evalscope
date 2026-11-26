# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, Sample, build_dataset_from_records, download_dataset_file
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# The 29 Indic languages of FLORES-IN, with the English names of the IndicGenBench paper (Table 7).
LANGUAGE_NAMES = {
    'as': 'Assamese',
    'awa': 'Awadhi',
    'bgc': 'Haryanvi',
    'bho': 'Bhojpuri',
    'bn': 'Bengali',
    'bo': 'Tibetan',
    'brx': 'Bodo',
    'gbm': 'Garhwali',
    'gom': 'Konkani',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'hne': 'Chhattisgarhi',
    'hoj': 'Rajasthani',
    'kn': 'Kannada',
    'mai': 'Maithili',
    'ml': 'Malayalam',
    'mni': 'Manipuri',
    'mr': 'Marathi',
    'mup': 'Malvi',
    'mwr': 'Marwari',
    'ne': 'Nepali',
    'or': 'Odia',
    'pa': 'Punjabi',
    'ps': 'Pashto',
    'sa': 'Sanskrit',
    'sat': 'Santali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ur': 'Urdu',
}
LANGUAGES = list(LANGUAGE_NAMES)

EN_TO_XX = 'en-xx'
XX_TO_EN = 'xx-en'

# One subset per language and direction, named `<source>-<target>`: `en-hi` is English to Hindi.
SUBSET_LIST = [f'en-{lang}' for lang in LANGUAGES] + [f'{lang}-en' for lang in LANGUAGES]

# Table 15 of the paper; the one-shot example uses the same format, followed by its reference.
PROMPT_TEMPLATE = 'Translate the following:\n\nTo {language}: {source}\nOutput:'

# chrF as defined by SacreBLEU's defaults: character n-grams up to 6, no word n-grams (chrF, not chrF++), beta 2.
CHRF_SETTINGS = {'char_order': 6, 'word_order': 0, 'beta': 2}


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


def first_line(text: str) -> str:
    """The first non-empty line of a completion; the prompt asks for a single translation after `Output:`."""
    return text.strip().split('\n', 1)[0].strip()


def corpus_chrf(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level chrF in [0, 1], computed by SacreBLEU from character n-gram counts pooled over all sentences."""
    from sacrebleu.metrics import CHRF

    return CHRF(**CHRF_SETTINGS).corpus_score(hypotheses, [references]).score / 100.0


def sentence_chrf(hypothesis: str, reference: str) -> float:
    """Sentence-level chrF in [0, 1], recorded per sample for inspection; subsets report `corpus_chrf`."""
    from sacrebleu.metrics import CHRF

    return CHRF(**CHRF_SETTINGS).sentence_score(hypothesis, [reference]).score / 100.0


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
        paper_url='https://aclanthology.org/2024.acl-long.595/',
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
class FloresINAdapter(DefaultDataAdapter):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {subset: subset_direction(subset) for subset in SUBSET_LIST}

    # ###############
    # DATA LOADING
    # ###############

    def load(self) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        """Read each subset's released JSON file; rows are under its top-level `examples` key."""
        for subset in self.subset_list:
            subset_languages(subset)
        test_dataset = DatasetDict(
            {
                subset: self._build_subset(subset, self.eval_split, self.limit, self.repeats, self.shuffle)
                for subset in self.subset_list
            }
        )
        fewshot_dataset = None
        if self._should_load_fewshot():
            fewshot_dataset = DatasetDict(
                {
                    subset: self._build_subset(subset, self.train_split, self.few_shot_num, 1, self.few_shot_random)
                    for subset in self.subset_list
                }
            )
        return test_dataset, fewshot_dataset

    def _build_subset(self, subset: str, split: str, limit: Any, repeats: int, shuffle: bool):
        records = self._read_records(subset, split)
        return build_dataset_from_records(
            records=records,
            sample_fields=self.record_to_sample,
            name=subset,
            location=self.dataset_id,
            limit=limit,
            repeats=repeats,
            shuffle=shuffle,
            seed=self.seed,
            filter_func=self.sample_filter,
        )

    def _read_records(self, subset: str, split: str) -> List[Dict[str, Any]]:
        path = download_dataset_file(
            data_id_or_path=self.dataset_id,
            file_path=subset_file(subset, split),
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        )
        with open(path, encoding='utf-8') as f:
            records = json.load(f)['examples']

        source, target = subset_languages(subset)
        direction, lang = ('enxx', target) if source == 'en' else ('xxen', source)
        for record in records:
            if record['translation_direction'] != direction or record['lang'] != lang:
                raise ValueError(
                    f'{subset_file(subset, split)} holds a {record["lang"]}/{record["translation_direction"]} row, '
                    f'expected {lang}/{direction}.'
                )
        return records

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
                'target_language_name': 'English' if target == 'en' else LANGUAGE_NAMES[target],
            },
        )

    # ###############
    # PROMPTS
    # ###############

    def format_prompt_template(self, sample: Sample) -> str:
        return self.prompt_template.format(
            language=sample.metadata['target_language_name'],
            source=sample.metadata['source'],
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'{self.format_prompt_template(sample)} {sample.target}'

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        return f'{fewshot}\n\n{self.format_prompt_template(sample)}'

    # ###############
    # SCORING
    # ###############

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return first_line(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        return Score(
            value={'chrf': sentence_chrf(filtered_prediction, reference)},
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            main_score_name='chrf',
            metadata={'reference': reference},
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Corpus-level chrF of the subset, from the scored translation and reference of every sample."""
        scored = [ss for ss in sample_scores if 'chrf' in ss.score.value]
        if not scored:
            return []
        hypotheses = [ss.score.extracted_prediction or '' for ss in scored]
        references = [ss.score.metadata['reference'] for ss in scored]
        return [
            AggScore(
                score=corpus_chrf(hypotheses, references),
                metric_name='chrf',
                aggregation='official',
                num=len(scored),
                ids=[ss.sample_id for ss in scored],
                metadata={'type': 'sacrebleu-corpus-chrf', **CHRF_SETTINGS},
            )
        ]
