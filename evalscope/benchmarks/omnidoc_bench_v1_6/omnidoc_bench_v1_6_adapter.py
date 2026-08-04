# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import glob
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.mixin import CodeExecutionSandboxMixin
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import SandboxEngine, resolve_engine
from evalscope.benchmarks.omnidoc_bench.omnidoc_bench_adapter import PROMPT_TEMPLATE
from evalscope.constants import HubType, Tags
from .sandbox_scorer import PAGE_METRICS, build_scoring_program, parse_scoring_result

# v1.6-only pins: the ModelScope revision and annotation digest identify the supported dataset;
# the scorer commit and image digest identify the official runtime used for every page.
DATASET_REVISION = '297ee5063d6ecc36fe14f3eb4f456607cc895f4a'
ANNOTATION_SHA256 = 'a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496'
OFFICIAL_SCORER_COMMIT = '147cd5ac9472002f5751221d390bf00abdbc0d2f'
OFFICIAL_IMAGE = (
    'ghcr.io/zeng-weijun/omnidocbench-eval'
    '@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617'
)
EXPECTED_SAMPLE_COUNT = 1651
EXPECTED_SUBSET_COUNTS = {
    'v1.5': 1355,
    'equation_hard': 100,
    'layout_hard': 99,
    'table_hard': 97,
}
REVIEW_TIMEOUT = 900
REQUIRED_SANDBOX_CONFIG = {
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

## Version and Data Source

- **Benchmark**: `omni_doc_bench_v1_6`
- **Dataset**: `OpenDataLab/OmniDocBench`, pinned to ModelScope revision `297ee5063d6ecc36fe14f3eb4f456607cc895f4a`
- **Scale**: 1,651 pages, including the 1,355-page v1.5 set and 296 equation, layout, and table hard pages
- **Compatibility**: other OmniDocBench releases and the legacy TSV integration are rejected

## Evaluation

Each page is scored independently by the official v1.6 evaluator inside an ms-enclave Docker sandbox. The sandbox reuses the pinned official image and runs MGAM `quick_match`, formula CDM, table TEDS/TEDS-S, edit distance, and reading-order evaluation. EvalScope averages the official page metrics and computes Overall only after all pages are aggregated.

- Edit-distance metrics use the 0-1 scale.
- CDM, TEDS, TEDS-S, and Overall use the 0-100 scale.
- Docker with amd64 support and `evalscope[sandbox]` are required.
- The sandbox pool defaults to one container; increase `sandbox.pool_size` only when sufficient memory is available.
- The official image is large; ensure sufficient disk and memory before evaluation.
- Scores are not directly comparable with the legacy `omni_doc_bench` v1.5 integration.
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
        sandbox_config=REQUIRED_SANDBOX_CONFIG,
    )
)
class OmniDocBenchV16Adapter(CodeExecutionSandboxMixin, VisionLanguageAdapter):
    """OmniDocBench adapter pinned to the official v1.6 dataset and Docker scorer."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_aggregation_name = False
        self.add_overall_metric = False
        self._source_hub: Optional[DatasetHub] = None
        self._local_root: Optional[Path] = None
        self._validate_sandbox_settings()

    def _validate_sandbox_settings(self) -> None:
        meta_conflicts = [
            key for key, value in REQUIRED_SANDBOX_CONFIG.items() if self.sandbox_config.get(key) != value
        ]
        if meta_conflicts:
            raise ValueError(
                'OmniDocBench v1.6 uses a pinned official sandbox configuration; '
                f'the following fields cannot be overridden: {", ".join(sorted(meta_conflicts))}.'
            )

        sandbox = self._task_config.sandbox if self._task_config else None
        if sandbox is None or not sandbox.enabled:
            return
        if resolve_engine(sandbox.engine) is not SandboxEngine.DOCKER:
            raise ValueError('OmniDocBench v1.6 requires the ms-enclave Docker engine.')
        user_config = dict(sandbox.default_config or {})
        conflicts = [
            key for key, value in REQUIRED_SANDBOX_CONFIG.items() if key in user_config and user_config[key] != value
        ]
        if conflicts:
            raise ValueError(
                'OmniDocBench v1.6 uses a pinned official sandbox configuration; '
                f'the following fields cannot be overridden: {", ".join(sorted(conflicts))}.'
            )

    def load(self) -> Tuple[DatasetDict, None]:
        """Load and validate the complete v1.6 annotation before selecting images."""
        annotation_path = self._resolve_annotation_path()
        records = self._load_and_validate_annotation(annotation_path)
        selected_records = self._select_records(records)
        self._prepare_selected_remote_images(selected_records)
        samples = [self.record_to_sample(record) for record in selected_records]
        if self.repeats > 1:
            samples = [copy.deepcopy(sample) for sample in samples for _ in range(self.repeats)]
        dataset = MemoryDataset(
            samples=samples,
            name='default',
            location=str(annotation_path),
            shuffled=self.shuffle,
        )
        dataset.reindex(group_size=self.repeats if self.repeats > 0 else 1)
        return DatasetDict({'default': dataset}), None

    def _resolve_annotation_path(self) -> Path:
        dataset_path = Path(self.dataset_id).expanduser()
        if dataset_path.exists():
            if dataset_path.is_file():
                self._local_root = dataset_path.resolve().parent
                return dataset_path.resolve()
            self._local_root = dataset_path.resolve()
            return self._resolve_local_file('OmniDocBench.json')

        if self.dataset_hub != HubType.MODELSCOPE:
            raise ValueError(
                'OmniDocBench v1.6 remote loading only supports the pinned ModelScope dataset; '
                'use local_path for an offline copy.'
            )
        if self.dataset_id != 'OpenDataLab/OmniDocBench':
            raise ValueError(
                'OmniDocBench v1.6 remote loading requires dataset_id `OpenDataLab/OmniDocBench`; '
                f'got `{self.dataset_id}`.'
            )
        self._source_hub = DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=HubType.MODELSCOPE,
            revision=DATASET_REVISION,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        )
        return Path(self._source_hub.download_file('OmniDocBench.json'))

    def _load_and_validate_annotation(self, annotation_path: Path) -> List[Dict[str, Any]]:
        annotation_bytes = annotation_path.read_bytes()
        digest = hashlib.sha256(annotation_bytes).hexdigest()
        if digest != ANNOTATION_SHA256:
            raise ValueError(
                'Unsupported OmniDocBench annotation: EvalScope supports v1.6 only. '
                f'Expected SHA-256 {ANNOTATION_SHA256}, got {digest}.'
            )

        records = json.loads(annotation_bytes)
        if not isinstance(records, list) or len(records) != EXPECTED_SAMPLE_COUNT:
            actual_count = len(records) if isinstance(records, list) else type(records).__name__
            raise ValueError(
                f'Invalid OmniDocBench v1.6 sample count: expected {EXPECTED_SAMPLE_COUNT}, got {actual_count}.'
            )

        required_fields = {'layout_dets', 'page_info', 'extra'}
        subset_counts = Counter()
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not required_fields.issubset(record):
                raise ValueError(
                    'Invalid OmniDocBench v1.6 schema: every page must contain layout_dets, page_info, and extra; '
                    f'first invalid record index is {index}.'
                )
            page_info = record.get('page_info')
            page_attribute = page_info.get('page_attribute') if isinstance(page_info, dict) else None
            if (
                not isinstance(record.get('layout_dets'), list) or not isinstance(page_info, dict)
                or not isinstance(page_attribute, dict) or not isinstance(record.get('extra'), dict)
            ):
                raise ValueError(f'Invalid OmniDocBench v1.6 structure at record index {index}.')
            self._validate_image_name(page_info.get('image_path', ''))
            subset = page_attribute.get('subset', 'v1.5')
            subset_counts[subset] += 1

        if dict(subset_counts) != EXPECTED_SUBSET_COUNTS:
            raise ValueError(
                f'Invalid OmniDocBench v1.6 subset counts: expected {EXPECTED_SUBSET_COUNTS}, '
                f'got {dict(subset_counts)}.'
            )
        return records

    def _select_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected_records = list(records)
        if self.shuffle:
            random.Random(self.seed).shuffle(selected_records)

        if self.limit is None:
            return selected_records
        limit = self.limit
        if isinstance(limit, float):
            if not 0.0 <= limit <= 1.0:
                raise ValueError('Limit must be a non-negative integer or a float between 0 and 1.')
            limit = int(len(selected_records) * limit)
        elif isinstance(limit, int) and limit < 0:
            raise ValueError('Limit must be a non-negative integer or a float between 0 and 1.')
        return selected_records[:limit]

    def _prepare_selected_remote_images(self, records: List[Dict[str, Any]]) -> None:
        if self._source_hub is None or not records:
            return
        image_paths = [glob.escape(f'images/{record["page_info"]["image_path"]}') for record in records]
        self._local_root = Path(self._source_hub.download_snapshot(allow_file_pattern=image_paths))

    def _resolve_local_file(self, relative_path: str) -> Path:
        if self._local_root is None:
            raise RuntimeError('Local OmniDocBench root has not been initialized.')
        root = self._local_root.resolve()
        resolved_path = (root / relative_path).resolve()
        if os.path.commonpath([str(root), str(resolved_path)]) != str(root):
            raise ValueError(f'Invalid OmniDocBench dataset path: {relative_path}')
        if not resolved_path.is_file():
            raise FileNotFoundError(f'OmniDocBench dataset file was not found: {resolved_path}')
        return resolved_path

    @staticmethod
    def _validate_image_name(image_name: str) -> None:
        if not image_name or image_name in ('.', '..') or '/' in image_name or '\\' in image_name:
            raise ValueError(f'Invalid OmniDocBench v1.6 image path: {image_name}')

    def _resolve_image_path(self, image_name: str) -> Path:
        self._validate_image_name(image_name)
        return self._resolve_local_file(f'images/{image_name}')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        image_name = record['page_info']['image_path']
        image_path = self._resolve_image_path(image_name)
        image_format = image_path.suffix.lower().lstrip('.') or 'png'
        image_uri = self._image_bytes_to_base64(image_path.read_bytes(), default_format=image_format)
        content: List[Content] = [
            ContentImage(image=image_uri),
            ContentText(text=self.prompt_template),
        ]
        return Sample(
            input=[ChatMessageUser(content=content)],
            target='',
            metadata={
                'omnidocbench_version': 'v1.6',
                'dataset_revision': DATASET_REVISION,
                'annotation_sha256': ANNOTATION_SHA256,
                'image_name': image_name,
                'annotation': record,
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
        metadata = task_state.metadata
        annotation = metadata.get('annotation')
        image_name = metadata.get('image_name')
        if not isinstance(annotation, dict) or not image_name:
            raise ValueError('OmniDocBench v1.6 scoring requires annotation and image_name metadata.')

        program = build_scoring_program(annotation, image_name, original_prediction)
        result = self.execute_code_in_sandbox(program, timeout=int(self.review_timeout), language='python')
        metrics = parse_scoring_result(result)
        return Score(
            value=metrics,
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            main_score_name=next(iter(metrics)),
            metadata={
                'image_name': image_name,
                'official_scorer_commit': OFFICIAL_SCORER_COMMIT,
                'official_image': OFFICIAL_IMAGE,
            },
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
