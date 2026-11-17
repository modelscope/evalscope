# Copyright (c) Alibaba, Inc. and its affiliates.
"""CVTG-2K adapter: complex visual text generation benchmark (TextCrafter, arXiv:2503.23461)."""

import json
from typing import Any, Dict, Type
from zipfile import ZipFile

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import DataLoader, Dataset, Sample, build_dataset_from_records, download_dataset_file
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.text2image.corpus_aggregation import CorpusAggregationMixin
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

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

CVTG-2K is a benchmark of 2,000 prompts for evaluating complex visual text generation
(TextCrafter, arXiv:2503.23461). Prompts contain 2-5 quoted text regions that the generated
image must render; prompts are organized by region count under the CVTG and CVTG-Style splits
(the latter annotates style attributes such as size, color and font).

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation (text rendering)
- **Input**: Text prompt with quoted text regions
- **Output**: Generated image scored with Word Accuracy and word-level NED against the quoted text
- **Domain**: Complex visual text generation for text-to-image models

## Key Features

- 2,000 prompts (CVTG and CVTG-Style splits, 2-5 text regions each) from the official
  TextCrafter release
- Scores Word Accuracy and word-level NED against the single-quoted fragments of each prompt,
  mirroring the official TextCrafter evaluator including its corpus-level aggregation
- Shares one PaddleOCR engine across both metrics per language

## Evaluation Notes

- Ground-truth words are the single-quoted fragments of each prompt, matching the official
  TextCrafter evaluator; repeated ground-truth words are not consumed
- Metrics: cvtg_word_accuracy (membership word match) and cvtg_ned (word-level best-match
  Levenshtein similarity, higher is better), both aggregated corpus-level (total numerator /
  total denominator) as in the official evaluator
- Loads the complete 2,000-prompt ModelScope mirror by default. The CVTG and CVTG-Style
  configurations contain 1,002 and 998 test examples, respectively; the original source is
  [dnkdnk/CVTG-2K](https://huggingface.co/datasets/dnkdnk/CVTG-2K).
- Install PaddleOCR 2.x and PaddlePaddle for OCR scoring (for example, paddleocr==2.10.0
  and paddlepaddle==2.6.2). Missing dependencies or OCR failures exclude the affected
  metric instead of assigning a zero. A valid image with no recognized text counts as missing text.
- Images can be generated with a local diffusion model or an OpenAI-compatible image
  generation service; set model_task='image_generation' and eval_type='text2image'.
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
    corpus_metrics = {
        'CVTGWordAccuracy': 'cvtg_word_accuracy',
        'CVTGNED': 'cvtg_ned',
    }

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
        metadata = dict(record)
        prompt = record.get('prompt', '')
        metadata.update({'prompt': prompt, 'category': record.get('benchmark_type', record.get('category', ''))})
        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target='',
            metadata=metadata,
        )
