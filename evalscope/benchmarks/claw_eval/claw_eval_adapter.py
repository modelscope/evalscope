import json
import os
import queue
import sys
import threading
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, Sample, build_dataset_dict_from_record_map
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, HubType, JudgeStrategy, Tags
from evalscope.utils.argument_utils import get_secret_value
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger
from .utils import (
    DEFAULT_CLAW_EVAL_DATASET_ID,
    DEFAULT_CLAW_EVAL_PACKAGE,
    ClawEvalAssets,
    ensure_claw_eval_sandbox_image,
    load_claw_eval_trace,
    load_task_manifest,
    materialize_task_root,
    prepare_claw_eval_assets,
    run_claw_eval_task,
    write_claw_eval_config,
)

logger = get_logger()

_DEFAULT_SPLITS = ['general', 'multimodal', 'multi_turn']

_EXTRA_PARAMS = {
    'task_ids': {
        'type': 'list',
        'description': 'Optional exact Claw-Eval task ids to run after split filtering.',
        'value': [],
    },
}

_DESCRIPTION = """
## Overview

Claw-Eval evaluates assistant agents on realistic personal-assistant workflows that require tool use, file and fixture
access, multimodal inputs, and simulated user interactions. EvalScope runs the pinned official Claw-Eval Python runner,
Docker sandbox, and graders while exposing each Claw-Eval task as a normal EvalScope sample for caching, repeats,
parallel execution, reporting, and dashboard trace review.

## Task Description

- **Task Type**: Agentic personal-assistant tasks with tool use, sandbox files, multimodal fixtures, and optional
  simulated user turns.
- **Dataset**: `claw-eval/Claw-Eval` on ModelScope.
- **Subsets**: `general`, `multimodal`, and `multi_turn`; the current ModelScope manifest contains 300 tasks
  (161 general, 101 multimodal, and 38 multi_turn). Use `subset_list` to select subsets.
- **Output**: Official Claw-Eval scores and JSONL traces, EvalScope sample-level reviews, grouped summary metrics, and
  dashboard-rendered agent traces.

## Evaluation Notes

- Requires Python 3.11+ and the official package installed from the pinned source commit:
  `pip install "claw-eval[sandbox,mock,web] @
  git+https://github.com/claw-eval/claw-eval.git@d3f02d4938ab0832377d90535013def2b1a2fdc0"`.
- The installed package provides the Claw-Eval runner APIs. EvalScope also caches the same pinned source archive because
  `tasks/` and `Dockerfile.agent` are runtime assets, then loads the task manifest and fixtures from ModelScope.
- Full fixtures are downloaded from ModelScope (`data/fixtures.tar.gz`) and linked into the official task tree before
  execution. The archive is large; use `limit` or `extra_params.task_ids` for smoke runs.
- Each selected Claw-Eval task is one EvalScope sample. Official scoring runs once per sample; use EvalScope `repeats`
  for repeated trials per task and `eval_batch_size` for task-level worker concurrency.
- Claw-Eval runs with the official Docker sandbox image. If `claw-eval-agent:latest` is missing locally, EvalScope
  builds it automatically from the cached official `Dockerfile.agent`. The first run can be slow.
- EvalScope `use_cache` resumes completed task-level samples. Claw-Eval trace JSONL files are stored under
  `outputs/.../claw_eval/<split>/traces` and converted to EvalScope agent traces for dashboard visualization.
"""


@register_benchmark(
    BenchmarkMeta(
        name='claw_eval',
        pretty_name='Claw-Eval',
        dataset_id=DEFAULT_CLAW_EVAL_DATASET_ID,
        tags=[Tags.AGENT, Tags.MULTI_MODAL, Tags.MULTI_TURN],
        description=_DESCRIPTION,
        metric_list=['avg_score', 'pass_at_k', 'pass_hat_k', 'error_rate'],
        aggregation='mean',
        eval_split='test',
        subset_list=_DEFAULT_SPLITS,
        default_subset='general',
        prompt_template='{question}',
        extra_params=_EXTRA_PARAMS,
    )
)
class ClawEvalAdapter(AgentAdapter):
    """EvalScope wrapper for the official Claw-Eval runner."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._check_runtime()
        self._assets: Optional[ClawEvalAssets] = None
        self._assets_lock = threading.Lock()
        self._image_prepared = False
        self._port_slots: Optional[queue.Queue[int]] = None
        self._port_slots_lock = threading.Lock()

    def load(self) -> tuple[DatasetDict, None]:
        selected_splits = self._selected_splits()
        task_ids = self.extra_params.get('task_ids') or []
        if isinstance(task_ids, str):
            task_ids_filter = {task_id.strip() for task_id in task_ids.split(',') if task_id.strip()}
        else:
            task_ids_filter = {str(task_id).strip() for task_id in task_ids if str(task_id).strip()}
        records = load_task_manifest(
            dataset_id=self.dataset_id,
            data_source=self.dataset_hub or HubType.MODELSCOPE,
            splits=selected_splits,
            force_redownload=self.force_redownload,
        )

        records_by_split: Dict[str, List[Dict[str, Any]]] = {split: [] for split in selected_splits}
        for record in records:
            task_id = str(record.get('task_id') or '').strip()
            if not task_id:
                continue
            if task_ids_filter and task_id not in task_ids_filter:
                continue
            split = str(record.get('split') or '').strip()
            if split in records_by_split:
                normalized = dict(record)
                normalized['task_id'] = task_id
                normalized['split'] = split
                records_by_split[split].append(normalized)

        records_by_split = {split: split_records for split, split_records in records_by_split.items() if split_records}
        if not records_by_split:
            raise ValueError('No Claw-Eval tasks selected. Check splits, task_ids, and limit.')

        return build_dataset_dict_from_record_map(
            record_map=records_by_split,
            sample_fields=self.record_to_sample,
            location=self.dataset_id,
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
            seed=None,
        ), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        task_id = str(record.get('task_id') or '').strip()
        split = str(record.get('split') or '').strip()
        task_name = str(record.get('task_name') or record.get('name') or '')
        return Sample(
            input=[ChatMessageUser(content=f'Run Claw-Eval task {task_id}.')],
            target='',
            subset_key=split,
            metadata={
                'task_id': task_id,
                'split': split,
                'task_name': task_name,
                'difficulty': record.get('difficulty') or '',
                'dataset_id': self.dataset_id,
                'dataset_hub': self.dataset_hub,
            },
        )

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        selected_split = str(sample.metadata.get('split') or 'default')
        task_id = str(sample.metadata.get('task_id') or '').strip()
        if not task_id:
            raise ValueError('Claw-Eval sample is missing task_id metadata.')

        output_root = (Path(self.output_dir) / 'claw_eval' / selected_split).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        assets = self._prepare_assets_once()
        runtime_root = output_root / 'runtime' / f'{sample.id}_{task_id}'
        tasks_dir = materialize_task_root(
            source_tasks_dir=assets.tasks_dir,
            selected_task_ids=[task_id],
            output_root=runtime_root,
        )
        config_path = runtime_root / 'config.yaml'
        trace_root = output_root / 'traces'
        api_key = self._resolve_api_key(model)
        base_url = self._resolve_base_url(model)
        judge_args = self._task_config.judge_model_args or {}
        judge_model = judge_args.get('model_id')
        no_judge = self._task_config.judge_strategy == JudgeStrategy.RULE
        official_config = self._build_official_config(
            model=model,
            tasks_dir=tasks_dir,
            trace_root=trace_root,
            api_key=api_key,
            base_url=base_url,
            judge_args=judge_args,
            judge_model=judge_model,
            no_judge=no_judge,
        )
        write_claw_eval_config(config_path, official_config)

        port_slot = self._get_port_slots().get()
        try:
            result = run_claw_eval_task(
                task_dir=tasks_dir / task_id,
                config_path=config_path,
                trace_root=trace_root,
                repo_root=assets.repo_root,
                model_id=model.name,
                api_key=api_key,
                base_url=base_url,
                port_offset=port_slot * 50,
                judge_model=judge_model,
                no_judge=no_judge,
                proxy=os.environ.get('CLAW_EVAL_PROXY') or None,
            )
        finally:
            self._get_port_slots().put(port_slot)
            write_claw_eval_config(config_path, self._redact_official_config(official_config))
        sample.metadata['claw_eval_result'] = result

        content = json.dumps(
            {
                'task_id': result.get('task_id'),
                'trace_path': result.get('trace_path'),
                'metrics': result.get('metrics', {}),
            },
            ensure_ascii=False,
        )
        output = ModelOutput.from_content(model=model.name, content=content)
        output.metadata = result
        trace, messages = load_claw_eval_trace(result.get('trace_path'))
        return InferenceResult(output=output, trace=trace, messages=messages)

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: Any,
    ) -> Score:
        result = task_state.metadata.get('claw_eval_result') or {}
        metrics = dict(result.get('metrics') or {})
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=metrics,
            main_score_name='avg_score',
            metadata={
                'task_id': result.get('task_id'),
                'task_name': result.get('task_name'),
                'difficulty': result.get('difficulty'),
                'trace_path': result.get('trace_path'),
                'trace_root': result.get('trace_root'),
                'error': result.get('error'),
                'raw_result': result.get('raw_result'),
            },
        )

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        if not sample_scores:
            return []

        from claw_eval.models.scoring import compute_pass_at_k, compute_pass_hat_k

        grouped: Dict[Any, List[SampleScore]] = defaultdict(list)
        for sample_score in sample_scores:
            group_id = sample_score.group_id if sample_score.group_id is not None else sample_score.sample_id
            grouped[group_id].append(sample_score)

        group_avg_scores = []
        group_pass_at = []
        group_pass_hat = []
        sample_errors = 0
        for group_scores in grouped.values():
            trial_scores = [float(item.score.value.get('avg_score', 0.0)) for item in group_scores]
            group_avg_scores.append(sum(trial_scores) / len(trial_scores) if trial_scores else 0.0)
            k = len(trial_scores)
            group_pass_at.append(compute_pass_at_k(trial_scores, k=k))
            group_pass_hat.append(compute_pass_hat_k(trial_scores, k=k))
            for item in group_scores:
                if item.score.metadata and item.score.metadata.get('error'):
                    sample_errors += 1

        num_groups = len(grouped)
        num_samples = len(sample_scores)
        return [
            AggScore(
                score=sum(group_avg_scores) / num_groups,
                metric_name='avg_score',
                aggregation_name='mean',
                num=num_groups
            ),
            AggScore(
                score=sum(group_pass_at) / num_groups, metric_name='pass_at_k', aggregation_name='mean', num=num_groups
            ),
            AggScore(
                score=sum(group_pass_hat) / num_groups,
                metric_name='pass_hat_k',
                aggregation_name='mean',
                num=num_groups
            ),
            AggScore(
                score=sample_errors / num_samples, metric_name='error_rate', aggregation_name='mean', num=num_samples
            ),
        ]

    def _build_official_config(
        self,
        model: Model,
        tasks_dir: Path,
        trace_root: Path,
        api_key: Optional[str],
        base_url: Optional[str],
        judge_args: Dict[str, Any],
        judge_model: Optional[str],
        no_judge: bool,
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            'model': {
                'api_key': api_key,
                'base_url': base_url,
                'model_id': model.name,
                'temperature': getattr(model.config, 'temperature', 0.0),
            },
            'judge': {
                'api_key': get_secret_value(judge_args.get('api_key')) or api_key,
                'base_url': judge_args.get('api_url') or judge_args.get('base_url') or base_url,
                'model_id': judge_model or model.name,
                'enabled': not no_judge,
            },
            'defaults': {
                'trace_dir': str(trace_root),
                'tasks_dir': str(tasks_dir),
            },
            'sandbox': {
                'enabled': True,
            },
        }
        if getattr(model.config, 'extra_body', None):
            config['model']['extra_body'] = model.config.extra_body
        return config

    def _check_runtime(self) -> None:
        if sys.version_info < (3, 11):
            raise RuntimeError(
                'Claw-Eval official runner requires Python 3.11+. Run EvalScope with Python 3.11+ for claw_eval.'
            )
        check_import(
            'claw_eval',
            package=f'"{DEFAULT_CLAW_EVAL_PACKAGE}"',
            raise_error=True,
            feature_name=self.pretty_name,
        )

    def _cache_dir(self) -> str:
        configured = (os.environ.get('CLAW_EVAL_CACHE_DIR') or '').strip()
        return configured or str(Path(DEFAULT_EVALSCOPE_CACHE_DIR) / self.name)

    def _official_repo_path(self) -> Optional[str]:
        return (os.environ.get('CLAW_EVAL_REPO_PATH') or '').strip() or None

    def _resolve_base_url(self, model: Model) -> Optional[str]:
        if self._task_config is not None and self._task_config.api_url:
            return self._task_config.api_url
        return getattr(model.api, 'base_url', None)

    def _resolve_api_key(self, model: Model) -> Optional[str]:
        if self._task_config is not None:
            return get_secret_value(self._task_config.api_key)
        return getattr(model.api, 'api_key', None)

    def _redact_official_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        redacted = deepcopy(config)
        for section in ('model', 'judge'):
            section_config = redacted.get(section)
            if isinstance(section_config, dict) and section_config.get('api_key'):
                section_config['api_key'] = '<redacted>'
        return redacted

    def _selected_splits(self) -> List[str]:
        unsupported = [split for split in self.subset_list if split not in _DEFAULT_SPLITS]
        if unsupported:
            raise ValueError(f'Unsupported Claw-Eval subset(s): {unsupported}. Supported subsets: {_DEFAULT_SPLITS}')
        return list(self.subset_list)

    def _prepare_assets_once(self) -> ClawEvalAssets:
        with self._assets_lock:
            if self._assets is None:
                self._assets = prepare_claw_eval_assets(
                    dataset_id=self.dataset_id,
                    data_source=self.dataset_hub or HubType.MODELSCOPE,
                    cache_dir=self._cache_dir(),
                    force_redownload=self.force_redownload,
                    official_repo_path=self._official_repo_path(),
                )
            if not self._image_prepared:
                ensure_claw_eval_sandbox_image(self._assets.repo_root)
                self._image_prepared = True
            return self._assets

    def _get_port_slots(self) -> queue.Queue[int]:
        with self._port_slots_lock:
            if self._port_slots is None:
                workers = max(int(self._task_config.eval_batch_size or 1), 1)
                self._port_slots = queue.Queue(maxsize=workers)
                for slot in range(workers):
                    self._port_slots.put(slot)
            return self._port_slots
