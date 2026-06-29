from __future__ import annotations

import base64
import copy
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.agent.tools.python_exec import PYTHON_EXEC_TOOL_INFO, run_python_exec
from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.model import Model
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import ensure_docker_image_built
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, HubType, Tags
from evalscope.utils.import_utils import is_build_doc
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger
from .gdpval_scorer import GDPvalLocalScorer

logger = get_logger()

_DATASET_ID = 'openai-mirror/gdpval'
_DEFAULT_DOCKER_IMAGE = 'evalscope/gdpval:latest'
_SANDBOX_REFERENCE_DIR = '/reference_files'
_SANDBOX_DELIVERABLE_DIR = 'deliverable_files'
_HOST_ARTIFACT_ROOT = 'artifacts/gdpval'
_SUBMISSION_DIR_NAME = 'gdpval_submission'

_GDPVAL_EXTRA_PARAMS: Dict[str, Any] = {
    'dataset_hub': {
        'type': 'str',
        'description': 'Dataset hub used to load GDPval records and reference files.',
        'value': HubType.MODELSCOPE,
        'choices': [HubType.MODELSCOPE, HubType.HUGGINGFACE, HubType.LOCAL],
    },
    'dataset_revision': {
        'type': 'str',
        'description': 'Optional dataset revision. Empty uses the hub default.',
        'value': '',
    },
    'max_steps': {
        'type': 'int',
        'description': 'Maximum number of agent steps per sample.',
        'value': 250,
    },
    'command_timeout': {
        'type': 'float',
        'description': 'Default per-command timeout in seconds.',
        'value': 180.0,
    },
    'docker_image': {
        'type': 'str',
        'description': 'Docker image used as the per-sample sandbox.',
        'value': _DEFAULT_DOCKER_IMAGE,
    },
    'auto_build_docker_image': {
        'type': 'bool',
        'description': 'Automatically build the default GDPval Docker image if it is missing locally.',
        'value': True,
    },
    'network_enabled': {
        'type': 'bool',
        'description': 'Allow the sandbox to access the network.',
        'value': True,
    },
    'download_reference_files': {
        'type': 'bool',
        'description': 'Download each selected sample reference file from the dataset hub before inference.',
        'value': True,
    },
    'scoring_mode': {
        'type': 'str',
        'description': (
            'Scoring mode: export a local submission package, prepare it for external OpenAI grading, or run '
            'EvalScope\'s own non-official local rubric scorer for deterministic checks.'
        ),
        'value': 'export_only',
        'choices': ['export_only', 'openai_auto_grader_submission', 'local_rubric_judge'],
    },
}

_GDPVAL_DESCRIPTION = """
## Overview

GDPval evaluates whether models can complete realistic economically valuable work tasks and produce requested
deliverable files. This adapter targets OpenAI's public 220-task gold subset mirrored on ModelScope as
`openai-mirror/gdpval`.

## Task Description

- **Task Type**: Agentic professional work / deliverable generation
- **Input**: A workplace-style task prompt, optionally with reference files
- **Output**: Final response text and requested files under `deliverable_files/`
- **Dataset**: OpenAI public GDPval gold subset with 220 tasks

## Key Features

- Uses the native EvalScope `AgentLoopAdapter` with bash and Python execution tools.
- Loads records and reference files from ModelScope by default.
- Mounts selected reference files read-only into the sandbox under `/reference_files`.
- Extracts files written to `deliverable_files/` before sandbox teardown.
- Generates a GDPval submission package with `deliverable_text` and `deliverable_files` columns.

## Evaluation Notes

- The default Docker image is `evalscope/gdpval:latest` and is built automatically from the bundled Dockerfile when
  missing. Set `extra_params.auto_build_docker_image=false` to require a pre-built image, or override
  `extra_params.docker_image`.
- `submission_ready` is a local readiness metric: it is 1 when the model produced final text or at least one
  deliverable file. It is not an official GDPval quality score.
- `local_rubric_score` is an EvalScope-implemented, non-official score for rubric items that can be checked
  deterministically from local deliverables and reference files. It reports scorer coverage because not every
  GDPval natural-language rubric item is locally machine-checkable.
- Full document/spreadsheet/slide quality depends on the GDPval runtime image. Thin Python images are useful only for
  plumbing smoke tests.

## Scoring and Submission

- `scoring_mode=export_only` writes a local submission folder under the EvalScope reports directory.
- `scoring_mode=openai_auto_grader_submission` writes the same submission folder and records next-step metadata for
  submitting it to the external OpenAI GDPval grader.
- `scoring_mode=local_rubric_judge` additionally runs EvalScope's own local deterministic rubric scorer. This is useful
  for smoke and regression checks, but it is not the official GDPval score.
- EvalScope does not compute the official GDPval score locally. The Inspect Evals implementation also exports a
  submission dataset for external grading rather than shipping a local official judge.
"""

_PROMPT_SUFFIX = f"""

Reference files, when present, are mounted under `{_SANDBOX_REFERENCE_DIR}`. Use the bash or python_exec tools to inspect
them and create the requested deliverables.

Write all submitted files under a new folder named `{_SANDBOX_DELIVERABLE_DIR}` in the sandbox working directory.
We will grade your final message as part of the deliverable, but requested documents, spreadsheets, slides, media,
or archives should be actual files in `{_SANDBOX_DELIVERABLE_DIR}`.
"""


@register_benchmark(
    BenchmarkMeta(
        name='gdpval',
        pretty_name='GDPval',
        tags=[Tags.AGENT, Tags.KNOWLEDGE, Tags.MULTI_TURN],
        description=_GDPVAL_DESCRIPTION,
        dataset_id=_DATASET_ID,
        subset_list=['default'],
        default_subset='default',
        eval_split='train',
        prompt_template='{question}',
        metric_list=['submission_ready', 'local_rubric_score'],
        extra_params=_GDPVAL_EXTRA_PARAMS,
    )
)
class GDPvalAdapter(AgentLoopAdapter):
    """GDPval adapter using ModelScope data and EvalScope's native agent loop."""

    strategy_name = 'function_calling'
    max_steps_default = 250

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.command_timeout = float(self.extra_params.get('command_timeout', 180.0))
        self.docker_image = self.extra_params.get('docker_image') or _DEFAULT_DOCKER_IMAGE
        self.auto_build_docker_image = bool(self.extra_params.get('auto_build_docker_image', True))
        self.network_enabled = bool(self.extra_params.get('network_enabled', True))
        self.download_reference_files = bool(self.extra_params.get('download_reference_files', True))
        self.scoring_mode = self.extra_params.get('scoring_mode', 'export_only')
        self.use_local_rubric_scorer = self.scoring_mode == 'local_rubric_judge'
        self._current_output_dir: Optional[str] = None
        self._docker_image_checked = False
        self._submission_records: List[Dict[str, Any]] = []

    @property
    def source_dataset_hub(self) -> str:
        return self.extra_params.get('dataset_hub') or self.dataset_hub or HubType.MODELSCOPE

    @property
    def source_dataset_revision(self) -> Optional[str]:
        return self.extra_params.get('dataset_revision') or None

    @property
    def source_dataset(self) -> DatasetHub:
        return DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.source_dataset_hub,
            revision=self.source_dataset_revision,
            force_redownload=self.force_redownload,
        )

    def load_dataset(self) -> DatasetDict:
        dataset_dict: Dict[str, MemoryDataset] = {}
        for subset in self.subset_list:
            with self._temporary_attribute('current_subset_name', subset):
                records = self._load_records()
                if subset == self.default_subset:
                    self._submission_records = copy.deepcopy(records)
                if self.shuffle:
                    random.Random(self.seed).shuffle(records)
                records = self._apply_limit(records)
                samples = [self.record_to_sample(record) for record in records]
                if self.download_reference_files and not is_build_doc():
                    self._resolve_sample_reference_files(samples)
                if self.repeats > 1:
                    samples = [copy.deepcopy(sample) for sample in samples for _ in range(self.repeats)]
                dataset = MemoryDataset(samples=samples, name=self.name, location=self.dataset_id)
                dataset.reindex(group_size=self.repeats)
                dataset_dict[subset] = dataset

        self.test_dataset = DatasetDict(dataset_dict)
        self.fewshot_dataset = None
        self._post_process_samples()
        return self.test_dataset

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        reference_files = _as_list(record.get('reference_files'))
        reference_file_urls = _as_list(record.get('reference_file_urls'))
        reference_file_hf_uris = _as_list(record.get('reference_file_hf_uris'))
        sandbox_reference_paths = [_sandbox_reference_path(path) for path in reference_files]
        reference_hint = _format_reference_hint(sandbox_reference_paths)
        prompt = f"{record['prompt'].strip()}\n{reference_hint}{_PROMPT_SUFFIX}"

        return Sample(
            input=prompt,
            target='',
            metadata={
                'task_id': record.get('task_id'),
                'sector': record.get('sector'),
                'occupation': record.get('occupation'),
                'reference_files': reference_files,
                'reference_file_urls': reference_file_urls,
                'reference_file_hf_uris': reference_file_hf_uris,
                'reference_paths': [_remove_reference_hash(path) for path in reference_files],
                'sandbox_reference_paths': sandbox_reference_paths,
                'rubric_pretty': record.get('rubric_pretty'),
                'rubric_json': record.get('rubric_json'),
                'dataset_id': self.dataset_id,
                'dataset_hub': self.source_dataset_hub,
            },
        )

    def _post_process_samples(self) -> None:
        for subset_samples in self.test_dataset.values():
            for sample in subset_samples:
                tools = list(sample.tools or [])
                if not any(tool.name == 'bash' for tool in tools):
                    tools.append(BASH_TOOL_INFO)
                if not any(tool.name == 'python_exec' for tool in tools):
                    tools.append(PYTHON_EXEC_TOOL_INFO)
                sample.tools = tools
        super()._post_process_samples()

    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs: Any) -> TaskState:
        self._current_output_dir = output_dir
        try:
            return super().run_inference(model, sample, output_dir, **kwargs)
        finally:
            self._current_output_dir = None

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        return {
            'bash': run_bash,
            'python_exec': run_python_exec,
        }

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

        self._ensure_docker_image()
        volumes = self._build_reference_volumes(sample)
        sandbox_config: Dict[str, Any] = {
            'image': self.docker_image,
            'working_dir': '/workspace',
            'network_enabled': self.network_enabled,
            'environment': {
                'PAGER': 'cat',
                'MANPAGER': 'cat',
                'PIP_PROGRESS_BAR': 'off',
                'TQDM_DISABLE': '1',
            },
        }
        if volumes:
            sandbox_config['volumes'] = volumes

        env = EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config=sandbox_config,
            timeout=self.command_timeout,
        )
        artifact_dir = self._artifact_dir(sample)
        return _GDPvalArtifactEnvironment(env=env, artifact_dir=artifact_dir, metadata=sample.metadata)

    def get_build_context(self) -> tuple[str, str]:
        docker_context = Path(__file__).parent
        return docker_context.as_posix(), (docker_context / 'Dockerfile').as_posix()

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        deliverables = task_state.metadata.get('deliverable_files') or []
        ready = bool(filtered_prediction.strip() or deliverables)
        value = {'submission_ready': 1.0 if ready else 0.0}
        metadata: Dict[str, Any] = {
            'scoring_mode': self.scoring_mode,
            'deliverable_count': len(deliverables),
            'artifact_dir': task_state.metadata.get('artifact_dir'),
            'official_gdpval_score': None,
            'official_gdpval_score_note': 'GDPval official grading is external; this local metric is readiness only.',
        }
        main_score_name = 'submission_ready'
        if self.use_local_rubric_scorer:
            local_score = GDPvalLocalScorer(task_state.metadata, filtered_prediction).score()
            value.update({
                'local_rubric_score': local_score.score,
                'local_rubric_score_all_items': local_score.score_all_items,
                'local_rubric_coverage': local_score.coverage,
            })
            metadata['local_rubric_score'] = local_score.summary()
            metadata['local_rubric_item_results'] = [result.to_dict() for result in local_score.item_results]
            metadata['local_rubric_score_note'] = (
                'EvalScope local deterministic rubric score. This is not the official GDPval score.'
            )
            main_score_name = 'local_rubric_score'
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=value,
            metadata=metadata,
            main_score_name=main_score_name,
        )
        return score

    def _on_generate_report_end(self, report: Any, output_dir: str, **kwargs: Any) -> None:
        if is_build_doc():
            return
        self._export_submission(Path(output_dir))

    def _load_records(self) -> List[Dict[str, Any]]:
        logger.info(
            f'Loading GDPval from {self.source_dataset_hub}: '
            f'{self.dataset_id}, subset=default, split={self.eval_split}.'
        )
        dataset = self.source_dataset.load(split=self.eval_split, subset='default')
        return list(dataset)

    def _apply_limit(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.limit:
            return records
        if isinstance(self.limit, float):
            limit = int(len(records) * self.limit)
        else:
            limit = int(self.limit)
        return records[:max(limit, 0)]

    def _resolve_sample_reference_files(self, samples: List[Sample]) -> None:
        for sample in samples:
            host_files = []
            for file_path in sample.metadata.get('reference_files') or []:
                try:
                    local_path = self.source_dataset.download_file(file_path)
                except Exception as exc:
                    logger.warning(f'Failed to download GDPval reference file {file_path!r}: {exc}')
                    continue
                if local_path:
                    host_files.append(local_path)
            sample.metadata['host_reference_files'] = host_files

    def _build_reference_volumes(self, sample: Sample) -> Dict[str, Dict[str, str]]:
        volumes: Dict[str, Dict[str, str]] = {}
        for host_file in sample.metadata.get('host_reference_files') or []:
            host_path = Path(host_file)
            if not host_path.is_file():
                continue
            hash_dir = host_path.parent.name
            bind_dir = f'{_SANDBOX_REFERENCE_DIR}/{hash_dir}'
            volumes[str(host_path.parent)] = {'bind': bind_dir, 'mode': 'ro'}
        return volumes

    def _ensure_docker_image(self) -> None:
        if self._docker_image_checked or is_build_doc():
            return
        self._docker_image_checked = True

        if not self.auto_build_docker_image or self.docker_image != _DEFAULT_DOCKER_IMAGE:
            return

        build_ctx, dockerfile = self.get_build_context()
        ensure_docker_image_built(self.docker_image, path=build_ctx, dockerfile=dockerfile, label='GDPval docker image')

    def _artifact_dir(self, sample: Sample) -> Path:
        task_id = str(sample.metadata.get('task_id') or sample.id or 'unknown')
        safe_task_id = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in task_id)
        output_dir = self._current_output_dir or os.path.join(DEFAULT_EVALSCOPE_CACHE_DIR, self.name)
        return Path(output_dir) / _HOST_ARTIFACT_ROOT / safe_task_id

    def _export_submission(self, report_dir: Path) -> None:
        review_items = self._load_review_items(report_dir)
        if not review_items:
            logger.warning('No GDPval review cache found; skipping submission export.')
            return

        submission_dir = report_dir / _SUBMISSION_DIR_NAME
        if submission_dir.exists():
            shutil.rmtree(submission_dir)
        (submission_dir / 'data').mkdir(parents=True, exist_ok=True)

        results = self._submission_results(review_items, submission_dir)
        records = copy.deepcopy(self._submission_records) or self._load_records()
        for record in records:
            task_id = str(record.get('task_id') or '')
            result = results.get(task_id, {})
            record['deliverable_text'] = result.get('deliverable_text', '')
            record['deliverable_files'] = result.get('deliverable_files', [])

        try:
            import pandas as pd
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('GDPval submission export requires pandas and pyarrow.') from exc

        table = pd.DataFrame(records)
        if 'deliverable_text' not in table:
            table['deliverable_text'] = ''
        if 'deliverable_files' not in table:
            table['deliverable_files'] = [[] for _ in range(len(table))]
        table.to_parquet(submission_dir / 'data' / 'train-00000-of-00001.parquet', index=False, engine='pyarrow')

        info = {
            'benchmark': self.name,
            'dataset_id': self.dataset_id,
            'dataset_hub': self.source_dataset_hub,
            'scoring_mode': self.scoring_mode,
            'submission_dir': str(submission_dir),
            'local_metric': 'local_rubric_score' if self.use_local_rubric_scorer else 'submission_ready',
            'official_gdpval_score_computed_locally': False,
            'local_rubric_score_note': (
                'EvalScope local deterministic rubric score is non-official and only covers locally checkable '
                'rubric items.' if self.use_local_rubric_scorer else None
            ),
            'next_step': (
                'Upload this folder as a Hugging Face dataset and submit it to the external OpenAI GDPval grader.'
                if self.scoring_mode == 'openai_auto_grader_submission' else
                'Local submission package exported. External GDPval grading is not run locally.'
            ),
        }
        with open(submission_dir / 'submission_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info(f'GDPval submission package exported to: {submission_dir}')

    def _load_review_items(self, report_dir: Path) -> List[Dict[str, Any]]:
        outputs_dir = report_dir.parent.parent
        model_name = report_dir.name
        review_dir = outputs_dir / 'reviews' / model_name
        review_items: List[Dict[str, Any]] = []
        for subset in self.subset_list:
            review_file = review_dir / f'{self.name}_{subset}.jsonl'
            if review_file.exists():
                review_items.extend(jsonl_to_list(str(review_file)))
        return review_items

    def _submission_results(self, review_items: List[Dict[str, Any]],
                            submission_dir: Path) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for item in review_items:
            sample_score = item.get('sample_score') or {}
            metadata = sample_score.get('sample_metadata') or {}
            score = sample_score.get('score') or {}
            task_id = str(metadata.get('task_id') or sample_score.get('sample_id') or item.get('index') or '')
            if not task_id:
                continue
            deliverable_files = self._copy_submission_deliverables(
                task_id=task_id,
                deliverables=metadata.get('deliverable_files') or [],
                submission_dir=submission_dir,
            )
            results[task_id] = {
                'deliverable_text': score.get('prediction') or score.get('extracted_prediction') or '',
                'deliverable_files': deliverable_files,
            }
        return results

    def _copy_submission_deliverables(
        self,
        task_id: str,
        deliverables: List[Dict[str, Any]],
        submission_dir: Path,
    ) -> List[str]:
        copied_paths = []
        task_dir = _safe_path_name(task_id)
        for deliverable in deliverables:
            local_path = Path(str(deliverable.get('local_path') or ''))
            relative_path = _relative_deliverable_path(str(deliverable.get('path') or '')) or local_path.name
            if not local_path.is_file() or not _is_safe_relative_path(relative_path):
                continue
            target_relative = f'{_SANDBOX_DELIVERABLE_DIR}/{task_dir}/{relative_path}'
            target_path = submission_dir / target_relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(local_path, target_path)
                copied_paths.append(target_relative)
            except Exception as exc:
                logger.warning(f'Failed to copy GDPval deliverable {local_path} to {target_path}: {exc}')
        return copied_paths


class _GDPvalArtifactEnvironment(AgentEnvironment):
    name = 'gdpval_artifact'

    def __init__(self, env: AgentEnvironment, artifact_dir: Path, metadata: Dict[str, Any]) -> None:
        self._env = env
        self._artifact_dir = artifact_dir
        self._metadata = metadata

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        return await self._env.exec(cmd, cwd=cwd, input=input, timeout=timeout, env=env)

    async def close(self) -> None:
        try:
            await self._extract_deliverables()
        finally:
            await self._env.close()

    async def _extract_deliverables(self) -> None:
        check = await self._env.exec(['test', '-d', _SANDBOX_DELIVERABLE_DIR], timeout=30)
        if check.returncode != 0:
            self._metadata['deliverable_files'] = []
            self._metadata['artifact_dir'] = str(self._artifact_dir)
            return

        listing = await self._env.exec(
            ['find', _SANDBOX_DELIVERABLE_DIR, '-type', 'f', '-print0'],
            timeout=120,
        )
        if listing.returncode != 0:
            logger.warning(f'Failed to list GDPval deliverables in sandbox: {listing.stderr}')
            self._metadata['deliverable_files'] = []
            self._metadata['artifact_dir'] = str(self._artifact_dir)
            return
        file_paths = [path for path in listing.stdout.split('\0') if path.strip()]
        deliverables = []
        for file_path in file_paths:
            relative_path = _relative_deliverable_path(file_path)
            if not relative_path:
                continue
            encoded = await self._env.exec(['base64', '-w', '0', file_path], timeout=120)
            if encoded.returncode != 0 or not encoded.stdout:
                logger.warning(f'Failed to extract GDPval deliverable {file_path!r}: {encoded.stderr}')
                continue
            try:
                host_path = self._artifact_dir / _SANDBOX_DELIVERABLE_DIR / relative_path
                host_path.parent.mkdir(parents=True, exist_ok=True)
                host_path.write_bytes(base64.b64decode(encoded.stdout))
                deliverables.append({
                    'path': f'{_SANDBOX_DELIVERABLE_DIR}/{relative_path}',
                    'local_path': str(host_path),
                })
            except Exception as exc:
                logger.warning(f'Failed to write extracted GDPval deliverable {file_path!r} to host: {exc}')

        self._metadata['deliverable_files'] = deliverables
        self._metadata['artifact_dir'] = str(self._artifact_dir)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _remove_reference_hash(file_path: str) -> str:
    parts = file_path.split('/')
    if len(parts) == 3 and parts[0] == 'reference_files':
        return f'{parts[0]}/{parts[2]}'
    return file_path


def _sandbox_reference_path(file_path: str) -> str:
    parts = file_path.split('/')
    if len(parts) == 3 and parts[0] == 'reference_files':
        return f'{_SANDBOX_REFERENCE_DIR}/{parts[1]}/{parts[2]}'
    return f'{_SANDBOX_REFERENCE_DIR}/{Path(file_path).name}'


def _format_reference_hint(reference_paths: List[str]) -> str:
    if not reference_paths:
        return ''
    lines = ['\nReference files for this task:']
    for path in reference_paths:
        lines.append(f'- {path}')
    return '\n'.join(lines)


def _relative_deliverable_path(file_path: str) -> str:
    prefix = f'{_SANDBOX_DELIVERABLE_DIR}/'
    if not file_path.startswith(prefix):
        return ''
    relative_path = file_path[len(prefix):].strip()
    if not _is_safe_relative_path(relative_path):
        return ''
    return relative_path


def _is_safe_relative_path(path: str) -> bool:
    return bool(path and not path.startswith('/') and '..' not in Path(path).parts)


def _safe_path_name(value: str) -> str:
    safe_value = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in value)
    return safe_value or 'unknown'
