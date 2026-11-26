# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Type, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import DataLoader, Dataset, MemoryDataset, Sample
from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.aggregators.aggregators import Mean
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

# Language code -> (name, resource level), as listed on each dataset card. The two cards rate some languages differently
# (Czech is high-resource in Global-MMLU and mid-resource in Global-MMLU-Lite), so each benchmark keeps its own table.
GLOBAL_MMLU_LANGUAGES = {
    'am': ('Amharic', 'low'),
    'ar': ('Arabic', 'high'),
    'bn': ('Bengali', 'mid'),
    'cs': ('Czech', 'high'),
    'de': ('German', 'high'),
    'el': ('Greek', 'mid'),
    'en': ('English', 'high'),
    'es': ('Spanish', 'high'),
    'fa': ('Persian', 'high'),
    'fil': ('Filipino', 'mid'),
    'fr': ('French', 'high'),
    'ha': ('Hausa', 'low'),
    'he': ('Hebrew', 'mid'),
    'hi': ('Hindi', 'high'),
    'id': ('Indonesian', 'mid'),
    'ig': ('Igbo', 'low'),
    'it': ('Italian', 'high'),
    'ja': ('Japanese', 'high'),
    'ko': ('Korean', 'mid'),
    'ky': ('Kyrgyz', 'low'),
    'lt': ('Lithuanian', 'mid'),
    'mg': ('Malagasy', 'low'),
    'ms': ('Malay', 'mid'),
    'ne': ('Nepali', 'low'),
    'nl': ('Dutch', 'high'),
    'ny': ('Chichewa', 'low'),
    'pl': ('Polish', 'high'),
    'pt': ('Portuguese', 'high'),
    'ro': ('Romanian', 'mid'),
    'ru': ('Russian', 'high'),
    'si': ('Sinhala', 'low'),
    'sn': ('Shona', 'low'),
    'so': ('Somali', 'low'),
    'sr': ('Serbian', 'high'),
    'sv': ('Swedish', 'high'),
    'sw': ('Swahili', 'low'),
    'te': ('Telugu', 'low'),
    'tr': ('Turkish', 'high'),
    'uk': ('Ukrainian', 'mid'),
    'vi': ('Vietnamese', 'high'),
    'yo': ('Yoruba', 'low'),
    'zh': ('Chinese', 'high'),
}

GLOBAL_MMLU_LITE_LANGUAGES = {
    'ar': ('Arabic', 'high'),
    'bn': ('Bengali', 'mid'),
    'cs': ('Czech', 'mid'),
    'cy': ('Welsh', 'low'),
    'de': ('German', 'high'),
    'en': ('English', 'high'),
    'es': ('Spanish', 'high'),
    'fr': ('French', 'high'),
    'hi': ('Hindi', 'high'),
    'hu': ('Hungarian', 'mid'),
    'id': ('Indonesian', 'mid'),
    'it': ('Italian', 'high'),
    'ja': ('Japanese', 'high'),
    'ko': ('Korean', 'mid'),
    'my': ('Burmese', 'low'),
    'or': ('Oriya', 'low'),
    'pt': ('Portuguese', 'high'),
    'sk': ('Slovak', 'mid'),
    'sq': ('Albanian', 'low'),
    'sw': ('Swahili', 'low'),
    'tg': ('Tajik', 'low'),
    'yo': ('Yoruba', 'low'),
    'zh': ('Chinese', 'high'),
}

# Culturally sensitive / culturally agnostic, from the dataset's annotator votes; '-' marks unannotated questions.
CULTURAL_LABELS = {'CS': 'cs', 'CA': 'ca'}


class GlobalMMLUBaseAdapter(MultiChoiceAdapter):
    """Shared Global-MMLU logic: subject-matched 5-shot examples and CS/CA accuracy next to the overall accuracy."""

    LANGUAGES: Dict[str, tuple] = {}
    # Languages released without a `dev` split; they are evaluated zero-shot.
    LANGUAGES_WITHOUT_DEV: frozenset = frozenset()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {code: f'{level}-resource' for code, (_, level) in self.LANGUAGES.items()}

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        language = self.current_subset_name
        question = (record['question'] or '').strip()
        if not question:
            # Global-MMLU-Lite's Burmese test row 202 was released without its question; it cannot be asked.
            return []
        return Sample(
            input=question,
            # 39 of the 589,764 Global-MMLU test rows lost an option in translation (None or empty); like the
            # lm-evaluation-harness tasks, keep the row and show the option empty.
            choices=[record[f'option_{letter}'] or '' for letter in 'abcd'],
            target=record['answer'].strip(),
            metadata={
                'sample_id': record['sample_id'],
                # The English subject from `sample_id` (e.g. `astronomy/test/58`): Global-MMLU-Lite's Hungarian rows
                # give `subject` in Hungarian, which would not match the `dev` examples of other rows.
                'subject': record['sample_id'].split('/')[0],
                'subject_category': record['subject_category'],
                'cultural_sensitivity_label': record['cultural_sensitivity_label'],
                'language': self.LANGUAGES.get(language, (language, None))[0],
            },
        )

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Load the whole `dev` split: each question takes the examples of its own subject from it."""
        if subset in self.LANGUAGES_WITHOUT_DEV:
            logger.warning(
                f'{self.pretty_name} `{subset}` has no `dev` split, so it is evaluated zero-shot '
                f'instead of {self.few_shot_num}-shot.'
            )
            return MemoryDataset([], name=subset)
        loader = data_loader(
            data_id_or_path=self.dataset_id,
            split=self.train_split,
            subset=subset,
            sample_fields=self.record_to_sample,
            limit=None,
            shuffle=False,
            shuffle_choices=False,
            seed=self.seed,
            data_source=self.dataset_hub,
            version=self.dataset_revision,
            force_redownload=self.force_redownload,
            dataset_dir=self.dataset_dir,
        )
        return loader.load()

    def process_sample_input(self, sample: Sample, subset: str) -> str:
        if self.few_shot_num <= 0 or self.fewshot_dataset is None or subset in self.LANGUAGES_WITHOUT_DEV:
            return self.format_prompt_template(sample=sample)
        subject = sample.metadata['subject']
        examples = [s for s in self.fewshot_dataset[subset] if s.metadata['subject'] == subject][: self.few_shot_num]
        if len(examples) < self.few_shot_num:
            raise ValueError(
                f'{self.pretty_name} `{subset}` has {len(examples)} `dev` questions for subject `{subject}`, '
                f'fewer than few_shot_num={self.few_shot_num}.'
            )
        fewshot = '\n\n'.join(self.sample_to_fewshot(example) for example in examples)
        return self.format_fewshot_template(fewshot=fewshot, sample=sample)

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        aggregates = super().aggregate_scores(sample_scores)
        for aggregate in aggregates:
            aggregate.dimensions = {**aggregate.dimensions, 'cultural_sensitivity': 'all'}
        for label, dimension in CULTURAL_LABELS.items():
            group = [
                ss for ss in sample_scores if (ss.sample_metadata or {}).get('cultural_sensitivity_label') == label
            ]
            for aggregate in Mean()(group):
                aggregate.dimensions = {'cultural_sensitivity': dimension}
                aggregates.append(aggregate)
        return aggregates


COMMON_NOTES = """
- Default configuration uses **5-shot** evaluation, as in the paper. The examples for a question are the five `dev`
  questions of its own subject (each subject has exactly five), in the same language
- Every language reports its accuracy on all questions, and separately on the culturally sensitive (`cs`) and
  culturally agnostic (`ca`) questions, as the `cultural_sensitivity` dimension of the accuracy metric
- Languages are grouped by the resource level the dataset card gives them (`high-resource`, `mid-resource`,
  `low-resource`)
- License: [Apache 2.0](https://opensource.org/license/apache-2-0)
"""


@register_benchmark(
    BenchmarkMeta(
        name='global_mmlu',
        evaluation_version='v1.0',
        pretty_name='Global-MMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=f"""
## Overview

Global-MMLU (Cohere Labs, 2024) is MMLU in 42 languages. It combines machine translations with professional
translations and community post-edits, and labels part of the questions as culturally sensitive (CS) or culturally
agnostic (CA), so a model's multilingual knowledge can be told apart from its Western-centric knowledge.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: An MMLU question with four options in one of 42 languages
- **Output**: The letter of the correct option (A-D)
- **Domain**: The 57 MMLU subjects

## Key Features

- 42 subsets, one per language, each with all 14,042 MMLU test questions (589,764 in total) and the 285 `dev`
  questions (five per subject)
- 2,850 questions per language carry a cultural sensitivity label: 792 CS and 2,058 CA; the rest are unannotated
- 39 test questions (0.007%, mostly Amharic, Hindi and Vietnamese) lost an option in translation; they are kept
  with that option empty, as in lm-evaluation-harness, so every language keeps all 14,042 questions. In 12 of
  them the lost option is the correct one
- For a smaller, fully human-translated set in 23 languages, use `global_mmlu_lite`

## Evaluation Notes
{COMMON_NOTES}""",
        paper_url='https://arxiv.org/abs/2412.03304',
        dataset_id='CohereLabs/Global-MMLU',
        hf_dataset_id='CohereLabs/Global-MMLU',
        metric_list=['acc'],
        primary_metric=MetricSelector(name='accuracy', aggregation='mean', dimensions={'cultural_sensitivity': 'all'}),
        subset_list=list(GLOBAL_MMLU_LANGUAGES),
        few_shot_num=5,
        allowed_few_shot_nums=(0, 1, 2, 3, 4, 5),
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class GlobalMMLUAdapter(GlobalMMLUBaseAdapter):
    LANGUAGES = GLOBAL_MMLU_LANGUAGES


@register_benchmark(
    BenchmarkMeta(
        name='global_mmlu_lite',
        evaluation_version='v1.0',
        pretty_name='Global-MMLU-Lite',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=f"""
## Overview

Global-MMLU-Lite (Cohere Labs) is the balanced, fully human-translated or post-edited subset of Global-MMLU: 200
culturally sensitive (CS) and 200 culturally agnostic (CA) MMLU questions in each of 23 languages.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: An MMLU question with four options in one of 23 languages
- **Output**: The letter of the correct option (A-D)
- **Domain**: 43 of the 57 MMLU subjects

## Key Features

- 23 subsets, one per language, each with 400 test questions (9,199 evaluated in total, see below) and 215 `dev`
  questions (285 for Czech), five per subject
- The language set differs from Global-MMLU: Welsh, Hungarian, Burmese, Oriya, Slovak, Albanian and Tajik are only in
  the Lite set. The paper's Lite results cover the 15 languages of the first release
- Every question is labeled CS or CA
- Burmese (`my`) test row 202 was released without its question and is skipped, so Burmese has 399 questions
- The Hungarian rows name their subject in Hungarian; few-shot examples are matched on the English subject in
  `sample_id` instead, which every row carries
- Welsh (`cy`) is released without a `dev` split, so it is always evaluated zero-shot

## Evaluation Notes
{COMMON_NOTES}""",
        paper_url='https://arxiv.org/abs/2412.03304',
        dataset_id='CohereLabs/Global-MMLU-Lite',
        hf_dataset_id='CohereLabs/Global-MMLU-Lite',
        metric_list=['acc'],
        primary_metric=MetricSelector(name='accuracy', aggregation='mean', dimensions={'cultural_sensitivity': 'all'}),
        subset_list=list(GLOBAL_MMLU_LITE_LANGUAGES),
        few_shot_num=5,
        allowed_few_shot_nums=(0, 1, 2, 3, 4, 5),
        train_split='dev',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class GlobalMMLULiteAdapter(GlobalMMLUBaseAdapter):
    LANGUAGES = GLOBAL_MMLU_LITE_LANGUAGES
    LANGUAGES_WITHOUT_DEV = frozenset({'cy'})
