from __future__ import annotations

import base64
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.api.dataset import Sample
from evalscope.api.sandbox import DockerImageSpec
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()

SANDBOX_REFERENCE_DIR = '/reference_files'
SANDBOX_DELIVERABLE_DIR = 'deliverable_files'
HOST_ARTIFACT_ROOT = 'artifacts/gdpval'
SUBMISSION_DIR_NAME = 'gdpval_submission'


def gdpval_image_spec(*, context_dir: Path, benchmark_name: str) -> DockerImageSpec:
    return DockerImageSpec(
        name_prefix='evalscope/gdpval',
        context_dir=context_dir.as_posix(),
        dockerfile=(context_dir / 'Dockerfile').as_posix(),
        cache_key_parts=[benchmark_name, 'gdpval'],
    )


class GDPvalArtifactEnvironment(AgentEnvironment):
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
        check = await self._env.exec(['test', '-d', SANDBOX_DELIVERABLE_DIR], timeout=30)
        if check.returncode != 0:
            self._metadata['deliverable_files'] = []
            self._metadata['artifact_dir'] = str(self._artifact_dir)
            return

        listing = await self._env.exec(
            ['find', SANDBOX_DELIVERABLE_DIR, '-type', 'f', '-print0'],
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
            relative_path = relative_deliverable_path(file_path)
            if not relative_path:
                continue
            encoded = await self._env.exec(['base64', '-w', '0', file_path], timeout=120)
            if encoded.returncode != 0 or not encoded.stdout:
                logger.warning(f'Failed to extract GDPval deliverable {file_path!r}: {encoded.stderr}')
                continue
            try:
                host_path = self._artifact_dir / SANDBOX_DELIVERABLE_DIR / relative_path
                host_path.parent.mkdir(parents=True, exist_ok=True)
                host_path.write_bytes(base64.b64decode(encoded.stdout))
                deliverables.append({
                    'path': f'{SANDBOX_DELIVERABLE_DIR}/{relative_path}',
                    'local_path': str(host_path),
                })
            except Exception as exc:
                logger.warning(f'Failed to write extracted GDPval deliverable {file_path!r} to host: {exc}')

        self._metadata['deliverable_files'] = deliverables
        self._metadata['artifact_dir'] = str(self._artifact_dir)


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def remove_reference_hash(file_path: str) -> str:
    parts = file_path.split('/')
    if len(parts) == 3 and parts[0] == 'reference_files':
        return f'{parts[0]}/{parts[2]}'
    return file_path


def sandbox_reference_path(file_path: str) -> str:
    parts = file_path.split('/')
    if len(parts) == 3 and parts[0] == 'reference_files':
        return f'{SANDBOX_REFERENCE_DIR}/{parts[1]}/{parts[2]}'
    return f'{SANDBOX_REFERENCE_DIR}/{Path(file_path).name}'


def format_reference_hint(reference_paths: List[str]) -> str:
    if not reference_paths:
        return ''
    lines = ['\nReference files for this task:']
    for path in reference_paths:
        lines.append(f'- {path}')
    return '\n'.join(lines)


def build_reference_volumes(sample: Sample) -> Dict[str, Dict[str, str]]:
    volumes: Dict[str, Dict[str, str]] = {}
    for host_file in sample.metadata.get('host_reference_files') or []:
        host_path = Path(host_file)
        if not host_path.is_file():
            continue
        hash_dir = host_path.parent.name
        bind_dir = f'{SANDBOX_REFERENCE_DIR}/{hash_dir}'
        volumes[str(host_path.parent)] = {'bind': bind_dir, 'mode': 'ro'}
    return volumes


def artifact_dir(sample: Sample, current_output_dir: Optional[str], benchmark_name: str) -> Path:
    task_id = str(sample.metadata.get('task_id') or sample.id or 'unknown')
    output_dir = current_output_dir or os.path.join(DEFAULT_EVALSCOPE_CACHE_DIR, benchmark_name)
    return Path(output_dir) / HOST_ARTIFACT_ROOT / safe_path_name(task_id)


def export_submission(
    *,
    report_dir: Path,
    submission_records: List[Dict[str, Any]],
    subset_list: List[str],
    benchmark_name: str,
    dataset_id: str,
    dataset_hub: str,
) -> None:
    review_items = load_review_items(report_dir, benchmark_name, subset_list)
    if not review_items:
        logger.warning('No GDPval review cache found; skipping submission export.')
        return

    import pandas as pd

    submission_dir = report_dir / SUBMISSION_DIR_NAME
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    (submission_dir / 'data').mkdir(parents=True, exist_ok=True)

    results = submission_results(review_items, submission_dir)
    records = copy.deepcopy(submission_records) or submission_records_from_review_items(review_items)
    for record in records:
        task_id = str(record.get('task_id') or '')
        result = results.get(task_id, {})
        record['deliverable_text'] = result.get('deliverable_text', '')
        record['deliverable_files'] = result.get('deliverable_files', [])

    table = pd.DataFrame(records)
    if 'deliverable_text' not in table:
        table['deliverable_text'] = ''
    if 'deliverable_files' not in table:
        table['deliverable_files'] = [[] for _ in range(len(table))]
    table.to_parquet(submission_dir / 'data' / 'train-00000-of-00001.parquet', index=False, engine='pyarrow')

    info = {
        'benchmark': benchmark_name,
        'dataset_id': dataset_id,
        'dataset_hub': dataset_hub,
        'submission_dir': str(submission_dir),
        'local_metric': 'submission_ready',
        'official_gdpval_score_computed_locally': False,
        'next_step': (
            'Local submission package exported. Run OpenAI\'s official GDPval judge externally to compute quality scores.'
        ),
    }
    with open(submission_dir / 'submission_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    logger.info(f'GDPval submission package exported to: {submission_dir}')


def submission_records_from_samples(test_dataset: Any, default_subset: str) -> List[Dict[str, Any]]:
    if test_dataset is None:
        return []
    records = []
    seen = set()
    for sample in test_dataset.get(default_subset, []):
        record = submission_record_from_metadata(sample.metadata)
        task_id = str(record.get('task_id') or '')
        if task_id in seen:
            continue
        seen.add(task_id)
        records.append(record)
    return records


def submission_records_from_review_items(review_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    seen = set()
    for item in review_items:
        sample_score = item.get('sample_score') or {}
        metadata = sample_score.get('sample_metadata') or {}
        record = submission_record_from_metadata(metadata)
        task_id = str(record.get('task_id') or '')
        if task_id in seen:
            continue
        seen.add(task_id)
        records.append(record)
    return records


def submission_record_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'task_id': metadata.get('task_id'),
        'sector': metadata.get('sector'),
        'occupation': metadata.get('occupation'),
        'prompt': metadata.get('prompt'),
        'reference_files': metadata.get('reference_files') or [],
        'reference_file_urls': metadata.get('reference_file_urls') or [],
        'reference_file_hf_uris': metadata.get('reference_file_hf_uris') or [],
        'rubric_pretty': metadata.get('rubric_pretty'),
        'rubric_json': metadata.get('rubric_json'),
    }


def load_review_items(report_dir: Path, benchmark_name: str, subset_list: List[str]) -> List[Dict[str, Any]]:
    outputs_dir = report_dir.parent.parent
    model_name = report_dir.name
    review_dir = outputs_dir / 'reviews' / model_name
    review_items: List[Dict[str, Any]] = []
    for subset in subset_list:
        review_file = review_dir / f'{benchmark_name}_{subset}.jsonl'
        if review_file.exists():
            review_items.extend(jsonl_to_list(str(review_file)))
    return review_items


def submission_results(review_items: List[Dict[str, Any]], submission_dir: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for item in review_items:
        sample_score = item.get('sample_score') or {}
        metadata = sample_score.get('sample_metadata') or {}
        score = sample_score.get('score') or {}
        task_id = str(metadata.get('task_id') or sample_score.get('sample_id') or item.get('index') or '')
        if not task_id:
            continue
        deliverable_files = copy_submission_deliverables(
            task_id=task_id,
            deliverables=metadata.get('deliverable_files') or [],
            submission_dir=submission_dir,
        )
        results[task_id] = {
            'deliverable_text': score.get('prediction') or score.get('extracted_prediction') or '',
            'deliverable_files': deliverable_files,
        }
    return results


def copy_submission_deliverables(
    *,
    task_id: str,
    deliverables: List[Dict[str, Any]],
    submission_dir: Path,
) -> List[str]:
    copied_paths = []
    task_dir = safe_path_name(task_id)
    for deliverable in deliverables:
        local_path = Path(str(deliverable.get('local_path') or ''))
        relative_path = relative_deliverable_path(str(deliverable.get('path') or '')) or local_path.name
        if not local_path.is_file() or not is_safe_relative_path(relative_path):
            continue
        target_relative = f'{SANDBOX_DELIVERABLE_DIR}/{task_dir}/{relative_path}'
        target_path = submission_dir / target_relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(local_path, target_path)
            copied_paths.append(target_relative)
        except Exception as exc:
            logger.warning(f'Failed to copy GDPval deliverable {local_path} to {target_path}: {exc}')
    return copied_paths


def relative_deliverable_path(file_path: str) -> str:
    prefix = f'{SANDBOX_DELIVERABLE_DIR}/'
    if not file_path.startswith(prefix):
        return ''
    relative_path = file_path[len(prefix):].strip()
    if not is_safe_relative_path(relative_path):
        return ''
    return relative_path


def is_safe_relative_path(path: str) -> bool:
    return bool(path and not path.startswith('/') and '..' not in Path(path).parts)


def safe_path_name(value: str) -> str:
    safe_value = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in value)
    return safe_value or 'unknown'
