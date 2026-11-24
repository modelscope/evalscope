# Copyright (c) Alibaba, Inc. and its affiliates.
"""DPG-Bench adapter: dense prompt graph benchmark (ELLA, TencentQQGYLab, arXiv:2403.12915)."""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Type

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import DataLoader, Dataset, LocalDataLoader, Sample, download_dataset_file
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.text2image.corpus_aggregation import CorpusAggregationMixin
from evalscope.constants import HubType, Tags


def convert_dpg_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Group the official ``dpg_bench.csv`` rows into prompt-level records.

    Each record carries the prompt and its atomic question metadata (proposition id,
    question text, dependency ids, L1/L2 categories), following the official
    question_dict structure built by the reference evaluator.
    """
    by_item: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        item_id = row['item_id']
        if item_id not in by_item:
            by_item[item_id] = {'item_id': item_id, 'prompt': row['text'], 'questions': []}
            order.append(item_id)
        dependency = [int(d) for d in str(row.get('dependency', '')).split(',') if d.strip().isdigit()]
        by_item[item_id]['questions'].append(
            {
                'proposition_id': int(row['proposition_id']),
                'question': row['question_natural_language'],
                'dependency': dependency,
                'category_broad': row['category_broad'],
                'category_detailed': row['category_detailed'],
            }
        )
    return [by_item[item_id] for item_id in order]


@register_benchmark(
    BenchmarkMeta(
        name='dpg_bench',
        pretty_name='DPG-Bench',
        paper_url='https://arxiv.org/abs/2403.12915',
        dataset_id='evalscope/DPG-Bench',
        hf_dataset_id='End2End-Diffusion/diffusionbench-evals',
        description="""
## Overview

DPG-Bench (ELLA) evaluates semantic alignment on 1,065 dense prompts describing multiple
objects, attributes and relations, built from COCO, PartiPrompts, DSG-1k and Object365.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: Dense prompt describing a multi-object scene
- **Output**: Image scored through atomic yes/no questions decomposed from the prompt
- **Domain**: Dense prompt semantic alignment

## Key Features

- 1,065 dense prompts decomposed into ~14k atomic questions with dependency structure,
  following the DSG scoring pipeline of the official evaluator.
- Loads the ModelScope mirror by default; the official metadata CSV from the
  diffusionbench-evals packaging is also supported.

## Evaluation Notes

- A vision API answers each atomic question. An answer scores 1.0 only when it is exactly
  'yes'; a question whose parent is adjusted to 0 is forced to 0. The overall score is the
  mean of adjusted per-image ratios; the five L1 category scores (Global / Entity /
  Attribute / Relation / Other) aggregate RAW answers corpus-level, as in the official
  evaluator.
- Defaults to qwen3.8-flash on DashScope: set DASHSCOPE_API_KEY; no local VLM weights are
  needed. The official evaluator uses mPLUG, so results from different judges are not
  directly comparable. API failures are excluded rather than scored zero.
- A prompt-level category without questions emits no report row; use repeats=4 for the
  official repeated-image protocol.
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['default'],
        metric_list=['DPGScore'],
        primary_metric=MetricSelector(name='dpg_score', aggregation='mean', dimensions={'scope': 'overall'}),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        evaluation_version='v1.0',
    )
)
class DPGBenchAdapter(CorpusAggregationMixin, Text2ImageAdapter):
    corpus_metrics = {'dpg_score': 'components'}

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Load the official metadata CSV and convert it into prompt-level records.

        A local prompt-level JSONL path loads directly; the ModelScope mirror loads through
        the standard flow; any other hub resolves the official metadata CSV and converts it.
        """
        if os.path.isfile(self.dataset_id):
            data_path = self.dataset_id
        elif self.dataset_hub == HubType.MODELSCOPE and self.dataset_id == 'evalscope/DPG-Bench':
            return super().load_subset(subset, data_loader)
        else:
            if self.eval_split != 'test' or subset != 'default':
                raise ValueError('DPG-Bench provides the test split only.')
            source = download_dataset_file(
                self.dataset_id,
                'dpg_bench.csv',
                data_source=self.dataset_hub,
                revision=self.dataset_revision,
                cache_dir=self.dataset_dir,
                force_redownload=self.force_redownload,
            )
            records = convert_dpg_csv(source)
            converted = Path(source).with_suffix('.prompts.jsonl')
            with open(converted, 'w', encoding='utf-8', newline='\n') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            data_path = str(converted)
        return LocalDataLoader(
            data_id_or_path=data_path,
            split=self.eval_split,
            subset=subset,
            sample_fields=self.record_to_sample,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
            filter_func=self.sample_filter,
        ).load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Records carry the prompt and its atomic question metadata.

        The question list is serialized into the sample target so the scorer receives it
        through the image-pair channel.
        """
        sample = super().record_to_sample(record)
        sample.target = json.dumps(record.get('questions', []), ensure_ascii=False)
        return sample
