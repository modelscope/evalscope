from __future__ import annotations

import base64
import copy
import os
import random
import shutil
import subprocess
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
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, HubType, Tags
from evalscope.utils.import_utils import is_build_doc
from evalscope.utils.logger import get_logger

logger = get_logger()

_DATASET_ID = 'openai-mirror/gdpval'
_DEFAULT_DOCKER_IMAGE = 'evalscope/gdpval:latest'
_SANDBOX_REFERENCE_DIR = '/reference_files'
_SANDBOX_DELIVERABLE_DIR = 'deliverable_files'
_HOST_ARTIFACT_ROOT = 'artifacts/gdpval'

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
        'description': 'Scoring mode. The initial native implementation supports export_only.',
        'value': 'export_only',
        'choices': ['export_only'],
    },
}

_PROMPT_SUFFIX = f"""

Reference files, when present, are mounted under `{_SANDBOX_REFERENCE_DIR}`. Use the bash or python_exec tools to inspect
them and create the requested deliverables.

Write all submitted files under a new folder named `{_SANDBOX_DELIVERABLE_DIR}` in the sandbox working directory.
We will grade your final message as part of the deliverable, but requested documents, spreadsheets, slides, media,
or archives should be actual files in `{_SANDBOX_DELIVERABLE_DIR}`.
"""


@register_benchmark(
    BenchmarkMeta(
        name='gdpval_openai_gold',
        pretty_name='GDPval OpenAI Gold',
        tags=[Tags.AGENT, Tags.KNOWLEDGE, Tags.MULTI_TURN],
        description="""
## Overview

GDPval evaluates whether models can complete realistic economically valuable work tasks and produce requested
deliverable files. This adapter targets OpenAI's public 220-task gold subset mirrored on ModelScope as
`openai-mirror/gdpval`.

## Evaluation Notes

- Uses the native EvalScope `AgentLoopAdapter` with bash and Python execution tools.
- Loads records and reference files from ModelScope by default.
- Mounts selected reference files read-only into the sandbox under `/reference_files`.
- Extracts files written to `deliverable_files/` before sandbox teardown and stores a manifest in sample metadata.
- First native scoring mode is `export_only`: `submission_ready` is 1 when the model produced final text or at least
  one deliverable file.
- The default Docker image is built automatically if missing. Set `extra_params.auto_build_docker_image=false` to
  require a pre-built image, or override `extra_params.docker_image`.
""",
        dataset_id=_DATASET_ID,
        subset_list=['default'],
        default_subset='default',
        eval_split='train',
        prompt_template='{question}',
        metric_list=['submission_ready'],
        extra_params=_GDPVAL_EXTRA_PARAMS,
    )
)
class GDPvalOpenAIGoldAdapter(AgentLoopAdapter):
    """GDPval OpenAI gold adapter using ModelScope data and EvalScope's native agent loop."""

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
        self._current_output_dir: Optional[str] = None
        self._docker_image_checked = False

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

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        deliverables = task_state.metadata.get('deliverable_files') or []
        ready = bool(filtered_prediction.strip() or deliverables)
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'submission_ready': 1.0 if ready else 0.0},
            metadata={
                'scoring_mode': self.scoring_mode,
                'deliverable_count': len(deliverables),
                'artifact_dir': task_state.metadata.get('artifact_dir'),
            },
            main_score_name='submission_ready',
        )
        return score

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

        if shutil.which('docker') is None:
            raise RuntimeError(
                f'GDPval default docker image {self.docker_image!r} is not available and Docker CLI was not found. '
                'Install Docker, pre-build the image, or set extra_params.docker_image to an existing image.'
            )

        inspect_result = subprocess.run(
            ['docker', 'image', 'inspect', self.docker_image],
            capture_output=True,
            text=True,
        )
        if inspect_result.returncode == 0:
            return

        docker_context = Path(__file__).parent
        logger.info(f'GDPval docker image {self.docker_image!r} not found. Building from {docker_context} ...')
        build_result = subprocess.run(['docker', 'build', '-t', self.docker_image, str(docker_context)])
        if build_result.returncode != 0:
            raise RuntimeError(
                f'Failed to build GDPval docker image {self.docker_image!r}. '
                'Build it manually with `bash evalscope/benchmarks/gdpval/build_docker_image.sh` '
                'or set extra_params.docker_image to an existing image.'
            )

    def _artifact_dir(self, sample: Sample) -> Path:
        task_id = str(sample.metadata.get('task_id') or sample.id or 'unknown')
        safe_task_id = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in task_id)
        output_dir = self._current_output_dir or os.path.join(DEFAULT_EVALSCOPE_CACHE_DIR, self.name)
        return Path(output_dir) / _HOST_ARTIFACT_ROOT / safe_task_id


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
            host_path = self._artifact_dir / _SANDBOX_DELIVERABLE_DIR / relative_path
            host_path.parent.mkdir(parents=True, exist_ok=True)
            host_path.write_bytes(base64.b64decode(encoded.stdout))
            deliverables.append({
                'path': f'{_SANDBOX_DELIVERABLE_DIR}/{relative_path}',
                'local_path': str(host_path),
            })

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
    if not relative_path or relative_path.startswith('/') or '..' in Path(relative_path).parts:
        return ''
    return relative_path
