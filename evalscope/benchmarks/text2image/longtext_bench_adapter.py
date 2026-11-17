# Copyright (c) Alibaba, Inc. and its affiliates.
"""LongText-Bench adapter: long text rendering benchmark (X-Omni, arXiv:2507.22058)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Type

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import DataLoader, Dataset, Sample, build_dataset_from_records, download_dataset_file
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.text2image.corpus_aggregation import CorpusAggregationMixin
from evalscope.constants import HubType, Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='longtext_bench',
        pretty_name='LongText-Bench',
        paper_url='https://arxiv.org/abs/2507.22058',
        dataset_id='evalscope/LongText-Bench',
        dataset_revision='0d375b2e1ad13fa36bec2c2fb28ed4f7dc1e060a',
        hf_dataset_id='X-Omni/LongText-Bench',
        hf_dataset_revision='429e30e9c1e0dea535038ae4238ea6ec42823098',
        description="""
## Overview

LongText-Bench (X-Omni, arXiv:2507.22058) evaluates long text rendering with 160 prompts each
for English and Chinese, across 8 text-rich scenes (signboards, labeled objects, printed
materials, webpages, slides, posters, subtitles, dialogues).

## Task Description

- **Task Type**: Text-to-Image Generation Evaluation (text rendering)
- **Input**: Text prompt describing a text-rich scene
- **Output**: Generated image transcribed by a VLM and matched against the ground-truth text
- **Domain**: Long text rendering for text-to-image models

## Key Features

- 160 English + 160 Chinese prompts across 8 text-rich scene categories from the official
  X-Omni release
- Text Score compares the VLM transcription with the ground-truth fragments, word-level for
  English and character-level for Chinese, corpus-aggregated as in the official evaluator
- Language is detected per sample from the ground-truth text, so one run covers both languages

## Evaluation Notes

- The ground truth is the dataset ``text`` field (a list of rendered fragments, joined with
  spaces like the official evaluator), passed to the scorer as the sample target
- Text Score is a Counter-intersection match ratio: word-level for English, character-level
  for Chinese (whitespace removed). Each language score is total matched / total ground
  truth tokens; the overall score averages the English and Chinese scores, as in the
  official leaderboard. Evaluating one language returns that language's score.
- Loads the ModelScope mirror by default, with 160 test examples in each of the en and zh
  configurations. Original data: [X-Omni/LongText-Bench](https://huggingface.co/datasets/X-Omni/LongText-Bench).
- Transcription uses an OpenAI-compatible vision API, defaulting to qwen3.8-flash on
  DashScope with thinking disabled. Set DASHSCOPE_API_KEY; no local VLM weights or accelerator
  are required. Metric args model, api_url, api_key and generation_config configure a different
  transcription service independently of the image generation model.
- Matching and language aggregation follow the official evaluator. The official transcription
  model is Qwen2.5-VL-7B; using qwen3.8-flash or another VLM can change the score, so those
  results must be identified by transcription model and are not directly comparable to the official results.
- Transcription failures and API token-limit truncation exclude the affected metric instead
  of assigning zero; a valid empty transcription scores zero.
- Set model_task='image_generation' and eval_type='text2image'. For the official repeated-image
  protocol, use repeats=4; limit is intended only for smoke runs.
""",
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['en', 'zh'],
        metric_list=['LongTextScore'],
        primary_metric=MetricSelector(name='text_score', aggregation='mean', dimensions={'scope': 'overall'}),
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        evaluation_version='v1.0',
    )
)
class LongTextBenchAdapter(CorpusAggregationMixin, Text2ImageAdapter):
    corpus_metrics = {
        'LongTextScore': 'text_score',
    }

    def _corpus_report_score(self, counts: List[Dict[str, Any]]) -> float:
        """Average language scores, matching the official LongText-Bench leaderboard."""
        return sum(item['numerator'] / item['denominator'] for item in counts) / len(counts)

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Load each official HF language file through the shared download and conversion flow."""
        if self.dataset_hub != HubType.HUGGINGFACE or self.dataset_id != 'X-Omni/LongText-Bench':
            return super().load_subset(subset, data_loader)
        if self.eval_split != 'test' or subset not in {'en', 'zh'}:
            raise ValueError('LongText-Bench provides the test split for en and zh only.')
        filename = 'text_prompts_zh.jsonl' if subset == 'zh' else 'text_prompts.jsonl'
        source = download_dataset_file(
            self.dataset_id,
            filename,
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            cache_dir=self.dataset_dir,
            force_redownload=self.force_redownload,
        )
        records = [json.loads(line) for line in Path(source).read_text(encoding='utf-8').splitlines() if line.strip()]
        return build_dataset_from_records(
            records,
            self.record_to_sample,
            name=subset,
            location=source,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=self.seed,
            filter_func=self.sample_filter,
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Records carry prompt_id/prompt/text; the ``text`` field is the rendering ground truth."""
        metadata = dict(record)
        prompt = record.get('prompt', '')
        metadata.update(
            {
                'prompt': prompt,
                'category': record.get('category', ''),
            }
        )
        # The dataset stores the ground truth as a list of rendered text fragments
        # (e.g. ['FRESH BREWS...', 'Locally Roasted...']); the official evaluator joins
        # them with spaces before matching.
        gt_text = record.get('text', '')
        if isinstance(gt_text, list):
            gt_text = ' '.join(str(part) for part in gt_text)
        return Sample(
            input=[ChatMessageUser(content=prompt)],
            # The target doubles as the ground-truth text: the framework's image-pair metric
            # channel passes the sample target to the scorer, which LongTextScore consumes.
            target=gt_text,
            metadata=metadata,
        )
