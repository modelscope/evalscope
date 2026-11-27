# Copyright (c) Alibaba, Inc. and its affiliates.
"""CVTG-2K adapter: complex visual text generation benchmark (TextCrafter, arXiv:2503.23461)."""

import json
from typing import Any, Dict, Type
from zipfile import ZipFile

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import DataLoader, Dataset, Sample, build_dataset_from_records, download_dataset_file
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.text2image.corpus_aggregation import CorpusAggregationMixin
from evalscope.constants import HubType, Tags

SUBSET_LIST = ['CVTG', 'CVTG-Style']


@register_benchmark(
    BenchmarkMeta(
        name='cvtg_2k',
        pretty_name='CVTG-2K',
        paper_url='https://arxiv.org/abs/2503.23461',
        dataset_id='evalscope/CVTG-2K',
        dataset_revision='dcfd5c1b341ad9c5ea4601576865770957be36e0',
        hf_dataset_id='dnkdnk/CVTG-2K',
        hf_dataset_revision='7823ea5a45f8acc53d281da83ef1a1df23812608',
        description="""
## Overview

CVTG-2K evaluates whether text-to-image models accurately render multiple text regions in one image.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: Prompt specifying 2-5 quoted text regions
- **Output**: Image containing the requested text
- **Domain**: Multi-region text rendering

## Key Features

- 2,000 prompts from TextCrafter: CVTG (1,002) and CVTG-Style (998).
- CVTG-Style additionally specifies text size, color and font.
- Loads the ModelScope mirror by default; the official Hugging Face source is also supported.

## Evaluation Notes

- Install `evalscope[cvtg_2k]` for PaddleOCR scoring. Use Python 3.11 or 3.12 on a platform
  supported by PaddlePaddle 2.6.
- Word Accuracy and word-level normalized edit similarity (NED, higher is better) follow the official evaluator,
  aggregating across all ground-truth words rather than averaging image scores.
- These metrics assess text content, not font, color, layout or image aesthetics.
- Scoring failures are excluded; valid images with no recognized text count as missing text.
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=SUBSET_LIST,
        metric_list=['CVTGWordAccuracy', 'CVTGNED'],
        primary_metric=MetricSelector(name='cvtg_word_accuracy', aggregation='mean', dimensions={'scope': 'overall'}),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        evaluation_version='v1.0',
    )
)
class CVTG2KAdapter(CorpusAggregationMixin, Text2ImageAdapter):
    corpus_metrics = ('cvtg_word_accuracy', 'cvtg_ned')

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Parse the official HF archive; ModelScope uses the standard configured JSONL loader."""
        if self.dataset_hub != HubType.HUGGINGFACE or self.dataset_id != 'dnkdnk/CVTG-2K':
            return super().load_subset(subset, data_loader)
        if self.eval_split != 'test' or subset not in SUBSET_LIST:
            raise ValueError('CVTG-2K provides the test split for CVTG and CVTG-Style only.')
        archive_path = download_dataset_file(
            self.dataset_id,
            'CVTG-2K.zip',
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            cache_dir=self.dataset_dir,
            force_redownload=self.force_redownload,
        )
        records = []
        with ZipFile(archive_path) as archive:
            for region_count in range(2, 6):
                source = f'CVTG-2K/{subset}/{region_count}.json'
                records.extend(
                    {**record, 'benchmark_type': subset, 'region_count': region_count}
                    for record in json.loads(archive.read(source))['data_list']
                )
        return build_dataset_from_records(
            records,
            self.record_to_sample,
            name=subset,
            location=archive_path,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
            filter_func=self.sample_filter,
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """CVTG-2K records carry the prompt with quoted text regions; keep the raw fields."""
        return super().record_to_sample(
            {
                **record,
                'category': record.get('benchmark_type', record.get('category', '')),
            }
        )
