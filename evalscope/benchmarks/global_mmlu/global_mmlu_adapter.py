# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Optional, Type, Union

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
OVERALL_ACCURACY = MetricSelector(name='accuracy', aggregation='mean', dimensions={'cultural_sensitivity': 'all'})


class GlobalMMLUBaseAdapter(MultiChoiceAdapter):
    """Shared Global-MMLU logic: subject-matched 5-shot examples and CS/CA accuracy next to the overall accuracy."""

    LANGUAGES: Dict[str, tuple[str, str]] = {}
    # Languages released without a `dev` split; they are evaluated zero-shot.
    LANGUAGES_WITHOUT_DEV: frozenset[str] = frozenset()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category_map = {code: f'{level}-resource' for code, (_, level) in self.LANGUAGES.items()}

    @property
    def primary_metric(self) -> Optional[MetricSelector]:
        selector = super().primary_metric
        if self.aggregation == 'bayes_at_n' and selector == OVERALL_ACCURACY:
            return OVERALL_ACCURACY.model_copy(update={'aggregation': 'bayes_at_n'})
        return selector

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        language = self.current_subset_name
        question = (record['question'] or '').strip()
        if not question:
            # Global-MMLU-Lite's Burmese test row 202 was released without its question; it cannot be asked.
            return []
        return Sample(
            input=question,
            # Keep rows with missing options, leaving the affected choice blank.
            choices=[(record[f'option_{letter}'] or '').strip() for letter in 'abcd'],
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

    def _fewshot_load_limit(self) -> Optional[int]:
        return None

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Load the whole `dev` split: each question takes the examples of its own subject from it."""
        if subset in self.LANGUAGES_WITHOUT_DEV:
            logger.warning(
                f'{self.pretty_name} `{subset}` has no `dev` split, so it is evaluated zero-shot '
                f'instead of {self.few_shot_num}-shot.'
            )
            return MemoryDataset([], name=subset)
        return super().load_fewshot_subset(subset, data_loader)

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
        if self.aggregation == 'bayes_at_n':
            return aggregates
        for label, dimension in CULTURAL_LABELS.items():
            group = [
                ss for ss in sample_scores if (ss.sample_metadata or {}).get('cultural_sensitivity_label') == label
            ]
            for aggregate in Mean()(group):
                aggregate.dimensions = {'cultural_sensitivity': dimension}
                aggregates.append(aggregate)
        return aggregates


COMMON_NOTES = """
- Uses five same-subject examples from the language's `dev` split where available.
- Scores generated answer letters by exact match. The upstream harness scores choice likelihoods and its Lite tasks
  select examples by subject category.
- Reports overall, culturally sensitive, and culturally agnostic accuracy per language. Bayes@N reports overall only.
- Licensed under [Apache 2.0](https://opensource.org/license/apache-2-0).
"""


@register_benchmark(
    BenchmarkMeta(
        name='global_mmlu',
        evaluation_version='v1.0',
        pretty_name='Global-MMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=f"""
## Overview

Global-MMLU evaluates multilingual knowledge and performance on culturally sensitive and agnostic questions.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: An MMLU question with four options in one of 42 languages
- **Output**: The letter of the correct option (A-D)
- **Domain**: The 57 MMLU subjects

## Key Features

- 589,764 test questions across 42 languages. Each language has 2,850 culturally labeled questions.
- Keeps the 39 rows with a missing option, including 12 where the correct choice is blank.

## Evaluation Notes
{COMMON_NOTES}""",
        paper_url='https://arxiv.org/abs/2412.03304',
        dataset_id='CohereLabs/Global-MMLU',
        hf_dataset_id='CohereLabs/Global-MMLU',
        metric_list=['acc'],
        primary_metric=OVERALL_ACCURACY,
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

Global-MMLU-Lite balances culturally sensitive and agnostic questions across 23 human-reviewed languages.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: An MMLU question with four options in one of 23 languages
- **Output**: The letter of the correct option (A-D)
- **Domain**: 43 of the 57 MMLU subjects

## Key Features

- Version 3.0 evaluates 9,199 questions after skipping one blank Burmese question. The paper covers the first 15 languages.
- Welsh uses zero-shot because its release has no `dev` split.

## Evaluation Notes
{COMMON_NOTES}""",
        paper_url='https://arxiv.org/abs/2412.03304',
        dataset_id='CohereLabs/Global-MMLU-Lite',
        hf_dataset_id='CohereLabs/Global-MMLU-Lite',
        metric_list=['acc'],
        primary_metric=OVERALL_ACCURACY,
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
