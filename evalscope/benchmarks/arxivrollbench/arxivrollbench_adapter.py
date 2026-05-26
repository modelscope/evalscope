# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import RemoteDataLoader, Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

PROMPT_TEMPLATE = """Answer the following ArxivRollBench multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
""".strip()

DOMAINS = [
    'cs',
    'q_fin',
    'math',
    'physics',
    'stat',
    'q_bio',
    'econ',
    'eess',
]
RELEASES = ['2024b', '2025a', '2026a']
TASK_TYPES = ['s', 'c', 'p']
SUBSET_LIST = [
    f'{release}_{domain}_{task_type}' for release in RELEASES for domain in DOMAINS for task_type in TASK_TYPES
]

DESCRIPTION = """
## Overview

ArxivRollBench is a rolling benchmark built from recent arXiv papers. It evaluates whether large language models can reason over fresh scientific text through three task formats: sequencing, cloze, and next-fragment prediction.

## Task Description

- **Task Type**: Multiple-choice scientific text reasoning
- **Input**: Recent arXiv text fragments with four answer choices
- **Output**: Single correct answer letter (A, B, C, or D)
- **Domains**: Computer Science, Quantitative Finance, Mathematics, Physics, Statistics, Quantitative Biology, Economics, and Electrical Engineering/System Science
- **Releases**: 2024b, 2025a, and 2026a rolling snapshots

## Key Features

- Time-aware benchmark snapshots reduce contamination-related overestimation
- Covers multiple arXiv domains and scientific writing styles
- Includes sequencing, cloze, and prediction formats under the SCP framework
- Compact `-50` split is suitable for cost-controlled API evaluation
- Full split is available as `arxivrollbench_full`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- The default `arxivrollbench` benchmark uses compact `-50` datasets
- Use `arxivrollbench_full` for the complete public splits
- Each subset is loaded from the public ModelScope mirror under the `liangzid` namespace
- Answers are normalized to A-D and evaluated with accuracy
""".strip()


def _parse_subset(subset: str) -> tuple[str, str, str]:
    match = re.fullmatch(r'(2024b|2025a|2026a)_(.+)_([scp])', subset)
    if not match:
        raise ValueError(f'Invalid ArxivRollBench subset: {subset}')
    return match.group(1), match.group(2), match.group(3)


def _selection_to_letter(label: str) -> str:
    match = re.search(r'\bselection\s*([1-4])\b', str(label), re.IGNORECASE)
    if match:
        return chr(ord('A') + int(match.group(1)) - 1)
    return str(label).strip().upper()


class ArxivRollBenchBaseAdapter(MultiChoiceAdapter):
    compact = True

    def load_subset(self, subset: str, data_loader=RemoteDataLoader):
        _parse_subset(subset)
        loader = data_loader(
            data_id_or_path=self.dataset_id,
            split='train',
            subset=subset,
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            shuffle_choices=self.shuffle_choices,
            data_source=self.dataset_hub,
            force_redownload=self.force_redownload,
            dataset_dir=self.dataset_dir,
        )
        return loader.load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [str(record.get(key, '')) for key in ['A', 'B', 'C', 'D']]
        label = record.get('label', '')

        if 'context' in record:
            question = (
                'Given the context, select the text that is the next sequence.'
                f"\n\nContext:\n{record.get('context', '')}"
            )
            task_type = 'p'
            target = str(label).strip().upper()
        else:
            question = (
                'Select the option that correctly completes the sequencing '
                f"or cloze task.\n\n{record.get('shuffled_text', '')}"
            )
            task_type = 's/c'
            target = _selection_to_letter(label)

        return Sample(
            input=question,
            choices=choices,
            target=target,
            metadata={
                'original_label': label,
                'task_type': task_type,
            },
        )


@register_benchmark(
    BenchmarkMeta(
        name='arxivrollbench',
        pretty_name='ArxivRollBench',
        tags=[Tags.MULTIPLE_CHOICE, Tags.REASONING, Tags.KNOWLEDGE],
        description=DESCRIPTION,
        dataset_id='liangzid/arxivrollbench',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template=PROMPT_TEMPLATE,
        paper_url='https://ojs.aaai.org/index.php/AAAI/article/view/41098',
    )
)
class ArxivRollBenchAdapter(ArxivRollBenchBaseAdapter):
    compact = True


@register_benchmark(
    BenchmarkMeta(
        name='arxivrollbench_full',
        pretty_name='ArxivRollBench-Full',
        tags=[Tags.MULTIPLE_CHOICE, Tags.REASONING, Tags.KNOWLEDGE],
        description=DESCRIPTION,
        dataset_id='liangzid/arxivrollbench-full',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template=PROMPT_TEMPLATE,
        paper_url='https://ojs.aaai.org/index.php/AAAI/article/view/41098',
    )
)
class ArxivRollBenchFullAdapter(ArxivRollBenchBaseAdapter):
    compact = False
