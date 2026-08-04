# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Type

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import DataLoader, Dataset, DictDataLoader, Sample, download_dataset_file
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.mixin import CodeExecutionSandboxMixin
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from ..legacy.omnidoc_bench_adapter import PROMPT_TEMPLATE
from .sandbox_scorer import PAGE_METRICS, build_scoring_program, parse_scoring_result

OFFICIAL_IMAGE = (
    'ghcr.io/zeng-weijun/omnidocbench-eval'
    '@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617'
)
REVIEW_TIMEOUT = 900
DEFAULT_SANDBOX_CONFIG = {
    'image': OFFICIAL_IMAGE,
    'entrypoint': [],
    'command': ['sleep', 'infinity'],
    'platform': 'linux/amd64',
    'working_dir': '/workspace',
    'network_enabled': False,
    'tools_config': {
        'python_executor': {}
    },
}

DESCRIPTION = """
## Overview

OmniDocBench v1.6 evaluates end-to-end document parsing for text, formulas, tables, layout, and reading order. This adapter is intentionally restricted to the official v1.6 data and scoring contract.

## Task Description

- **Task Type**: End-to-end document parsing
- **Input**: A complete document page image
- **Output**: Markdown containing the page text, formulas, tables, and reading order
- **Domain**: Multilingual academic, financial, textbook, newspaper, magazine, and presentation documents

## Key Features

- Uses the latest `OpenDataLab/OmniDocBench` revision available from ModelScope
- Contains 1,651 pages: 1,355 base pages plus 100 equation-hard, 99 layout-hard, and 97 table-hard pages
- Uses the official v1.6 data format; other releases and the legacy TSV format are not supported
- Scores each page independently with the official v1.6 evaluator in a reusable ms-enclave Docker sandbox

## Evaluation Notes

- Uses MGAM `quick_match`, formula CDM, table TEDS/TEDS-S, edit distance, and reading-order evaluation.
- EvalScope averages page metrics and computes Overall only from the aggregated text, formula, and table components.
- Edit-distance metrics use the 0-1 scale.
- CDM, TEDS, TEDS-S, and Overall use the 0-100 scale.
- Docker with amd64 support and `evalscope[sandbox]` are required.
- The default image is pinned; custom image overrides are allowed, but incompatible images fail during scoring.
- The sandbox pool defaults to one container; increase `sandbox.pool_size` only when sufficient memory is available.
- The official image is large; ensure sufficient disk and memory before evaluation.
- Scores are not directly comparable with the legacy `omni_doc_bench` integration.
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='omni_doc_bench_v1_6',
        pretty_name='OmniDocBench-v1.6',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=DESCRIPTION,
        dataset_id='OpenDataLab/OmniDocBench',
        paper_url='https://github.com/opendatalab/OmniDocBench',
        metric_list=[*PAGE_METRICS, 'overall'],
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        review_timeout=REVIEW_TIMEOUT,
        sandbox_config=DEFAULT_SANDBOX_CONFIG,
    )
)
class OmniDocBenchV16Adapter(CodeExecutionSandboxMixin, VisionLanguageAdapter):
    """OmniDocBench adapter pinned to the official v1.6 dataset and Docker scorer."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_aggregation_name = False
        self.add_overall_metric = False

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """Load the pinned v1.6 annotation and delegate selection to the standard loader."""
        annotation_path = Path(
            download_dataset_file(
                data_id_or_path=self.dataset_id,
                file_path='OmniDocBench.json',
                data_source=self.dataset_hub,
                force_redownload=self.force_redownload,
                cache_dir=self.dataset_dir,
            )
        )
        records = json.loads(annotation_path.read_text(encoding='utf-8'))
        return DictDataLoader(
            dict_list=records,
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            shuffle_choices=self.shuffle_choices,
            seed=self.seed,
        ).load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        image_name = record['page_info']['image_path']
        image_path = Path(
            download_dataset_file(
                data_id_or_path=self.dataset_id,
                file_path=f'images/{image_name}',
                data_source=self.dataset_hub,
                force_redownload=self.force_redownload,
                cache_dir=self.dataset_dir,
            )
        )
        image_format = image_path.suffix.lower().lstrip('.') or 'png'
        image_uri = self._image_bytes_to_base64(image_path.read_bytes(), default_format=image_format)
        content: List[Content] = [
            ContentImage(image=image_uri),
            ContentText(text=self.prompt_template),
        ]
        return Sample(
            input=[ChatMessageUser(content=content)],
            target=json.dumps(record, ensure_ascii=False),
            metadata={
                'omnidocbench_version': 'v1.6',
                'image_name': image_name,
            },
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score one page with the official v1.6 evaluator in the sandbox pool."""
        if not self.use_sandbox:
            raise RuntimeError('OmniDocBench v1.6 requires ms-enclave sandbox scoring. Enable TaskConfig.sandbox.')
        try:
            annotation = json.loads(reference)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError('OmniDocBench v1.6 scoring requires a valid page annotation reference.') from error
        image_name = task_state.metadata.get('image_name')
        if not isinstance(annotation, dict) or not image_name:
            raise ValueError('OmniDocBench v1.6 scoring requires a page annotation and image_name metadata.')

        program = build_scoring_program(annotation, image_name, original_prediction)
        result = self.execute_code_in_sandbox(program, timeout=int(self.review_timeout), language='python')
        metrics = parse_scoring_result(result)
        return Score(
            value=metrics,
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            main_score_name=next(name for name in PAGE_METRICS if name in metrics),
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Average official page metrics and compute Overall from the aggregated components."""
        metric_values = defaultdict(list)
        metric_ids = defaultdict(list)
        for sample_score in sample_scores:
            for metric_name, value in sample_score.score.value.items():
                metric_values[metric_name].append(float(value))
                metric_ids[metric_name].append(sample_score.sample_id)

        aggregated = []
        means = {}
        for metric_name in PAGE_METRICS:
            values = metric_values.get(metric_name, [])
            if not values:
                continue
            means[metric_name] = sum(values) / len(values)
            aggregated.append(
                AggScore(
                    score=means[metric_name],
                    metric_name=metric_name,
                    aggregation_name='mean',
                    num=len(values),
                    ids=metric_ids[metric_name],
                    metadata={'page_denominator': len(values)},
                )
            )

        overall_components = ('text_block_Edit_dist', 'display_formula_CDM', 'table_TEDS')
        if all(component in means for component in overall_components):
            overall = ((1.0 - means['text_block_Edit_dist']) * 100.0 + means['display_formula_CDM']
                       + means['table_TEDS']) / 3.0
            aggregated.append(
                AggScore(
                    score=overall,
                    metric_name='overall',
                    aggregation_name='official',
                    num=len(sample_scores),
                    ids=[sample_score.sample_id for sample_score in sample_scores],
                    metadata={
                        'component_page_denominators': {
                            component: len(metric_values[component])
                            for component in overall_components
                        }
                    },
                )
            )
        return aggregated
