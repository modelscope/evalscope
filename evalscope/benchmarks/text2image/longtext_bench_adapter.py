# Copyright (c) Alibaba, Inc. and its affiliates.
"""LongText-Bench adapter: long text rendering benchmark (X-Omni, arXiv:2507.22058)."""

from typing import Any, Dict, List, Type

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import DataLoader, Dataset, LocalDataLoader, Sample, download_dataset_file
from evalscope.api.metric import AggScore, MetricIdentity, MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.text2image.corpus_aggregation import CorpusAggregationMixin
from evalscope.constants import HubType, Tags
from evalscope.report import Report


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

LongText-Bench evaluates whether text-to-image models accurately render long English and Chinese text.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: Prompt describing a scene with specified text
- **Output**: Image containing the requested text
- **Domain**: Long text rendering

## Key Features

- 160 English and 160 Chinese prompts from X-Omni, covering 8 scenes including posters,
  webpages and slides.
- Loads the ModelScope mirror by default; the official Hugging Face source is also supported.

## Evaluation Notes

- A vision API transcribes the image. Text Score is matched tokens divided by ground-truth
  tokens (English words or Chinese characters). A bilingual overall score requires a scored result
  for each language; otherwise the report shows available language scores.
- Defaults to qwen3.8-flash on DashScope: set DASHSCOPE_API_KEY; no local VLM weights are needed.
  The official evaluator uses Qwen2.5-VL-7B, so results from different transcription models
  are not directly comparable.
- Scoring follows the official matching rules; use repeats=4 for its repeated-image protocol.
  Text Score does not assess layout or aesthetics. API failures and truncated replies are excluded;
  valid empty transcriptions score zero.
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
    corpus_metrics = ('text_score',)

    def _corpus_report_score(self, counts: List[Dict[str, Any]]) -> float:
        """Average language scores, matching the official LongText-Bench leaderboard."""
        return sum(item['numerator'] / item['denominator'] for item in counts) / len(counts)

    def _on_generate_report(self, scores: Dict[str, List[AggScore]], model_name: str) -> Report:
        report = super()._on_generate_report(scores, model_name)
        scored_languages = {
            subset
            for subset, aggregates in scores.items()
            if any(item.metric_name == 'text_score' and item.dimensions == {'scope': 'overall'} for item in aggregates)
        }
        missing_languages = set(self.subset_list) - scored_languages
        if len(scored_languages) != 1 or not missing_languages:
            return report

        # Keep the surviving language visible without presenting it as the bilingual result.
        language = next(iter(scored_languages))
        if report.primary_metric_identity and report.primary_metric_identity.name == 'text_score':
            report.primary_metric_identity = None
            report.primary_metric_unavailable_reason = (
                f'No usable Text Score for: {", ".join(sorted(missing_languages))}.'
            )
        for metric in report.metrics:
            if metric.identity.name == 'text_score':
                dimensions = {key: value for key, value in metric.identity.dimensions.items() if key != 'scope'}
                metric.identity = MetricIdentity(
                    name=metric.identity.name,
                    aggregation=metric.identity.aggregation,
                    dimensions={**dimensions, 'subset': language},
                )
        return report

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
        return LocalDataLoader(
            data_id_or_path=source,
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
        """Records carry prompt_id/prompt/text; the ``text`` field is the rendering ground truth."""
        sample = super().record_to_sample(record)
        text = record.get('text', '')
        sample.target = ' '.join(str(part) for part in text) if isinstance(text, list) else text
        return sample
