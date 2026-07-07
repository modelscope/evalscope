from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.agent.tools.python_exec import PYTHON_EXEC_TOOL_INFO, run_python_exec
from evalscope.api.agent import AgentEnvironment
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import DatasetHub, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.model import Model
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import DockerImageSpec, prepare_docker_image
from evalscope.constants import HubType, Tags
from evalscope.utils.import_utils import check_import, is_build_doc
from evalscope.utils.logger import get_logger
from .utils import (
    SANDBOX_DELIVERABLE_DIR,
    SANDBOX_REFERENCE_DIR,
    GDPvalArtifactEnvironment,
    artifact_dir,
    as_list,
    build_reference_volumes,
    export_submission,
    format_reference_hint,
    remove_reference_hash,
    sandbox_reference_path,
    submission_records_from_samples,
)

logger = get_logger()

_DATASET_ID = 'openai-mirror/gdpval'
_DEFAULT_DOCKER_IMAGE = 'evalscope/gdpval:latest'

_GDPVAL_EXTRA_PARAMS: Dict[str, Any] = {
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

- The default Docker image is built automatically from the bundled Dockerfile into a content-hashed local tag. Set
  `extra_params.auto_build_docker_image=false` to require a pre-built `evalscope/gdpval:latest`, or override
  `extra_params.docker_image`.
- `submission_ready` is a local readiness metric: it is 1 when the model produced final text or at least one
  deliverable file. It is not an official GDPval quality score.
- EvalScope does not run a local GDPval judge. Use the exported submission package with OpenAI's official GDPval judge
  to obtain quality scores.
- Full document/spreadsheet/slide quality depends on the GDPval runtime image. Thin Python images are useful only for
  plumbing smoke tests.

## Scoring and Submission

- EvalScope writes a local submission folder under the reports directory.
- The submission contains `deliverable_text` and `deliverable_files` fields in the GDPval dataset format.
- Official GDPval grading is external. Run OpenAI's official GDPval judge on the exported submission package.
"""

_PROMPT_SUFFIX = f"""

Reference files, when present, are mounted under `{SANDBOX_REFERENCE_DIR}`. Use the bash or python_exec tools to inspect
them and create the requested deliverables.

Write all submitted files under a new folder named `{SANDBOX_DELIVERABLE_DIR}` in the sandbox working directory.
We will grade your final message as part of the deliverable, but requested documents, spreadsheets, slides, media,
or archives should be actual files in `{SANDBOX_DELIVERABLE_DIR}`.
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
        metric_list=['submission_ready'],
        extra_params=_GDPVAL_EXTRA_PARAMS,
    )
)
class GDPvalAdapter(AgentLoopAdapter):
    """GDPval adapter using ModelScope data and EvalScope's native agent loop."""

    strategy_name = 'function_calling'
    max_steps_default = 250

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        check_import(
            module_name=['pandas', 'pyarrow'],
            package=['pandas', 'pyarrow'],
            raise_error=True,
            feature_name='GDPval submission export',
        )
        self.command_timeout = float(self.extra_params.get('command_timeout', 180.0))
        self.docker_image = self.extra_params.get('docker_image') or _DEFAULT_DOCKER_IMAGE
        self.auto_build_docker_image = bool(self.extra_params.get('auto_build_docker_image', True))
        self.network_enabled = bool(self.extra_params.get('network_enabled', True))
        self.download_reference_files = bool(self.extra_params.get('download_reference_files', True))
        self._current_output_dir: Optional[str] = None
        self._docker_image_checked = False
        self._submission_records: List[Dict[str, Any]] = []

    @property
    def source_dataset_hub(self) -> str:
        return self.dataset_hub or HubType.MODELSCOPE

    @property
    def source_dataset(self) -> DatasetHub:
        return DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.source_dataset_hub,
            force_redownload=self.force_redownload,
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        reference_files = as_list(record.get('reference_files'))
        reference_file_urls = as_list(record.get('reference_file_urls'))
        reference_file_hf_uris = as_list(record.get('reference_file_hf_uris'))
        sandbox_reference_paths = [sandbox_reference_path(path) for path in reference_files]
        reference_hint = format_reference_hint(sandbox_reference_paths)
        prompt = f"{record['prompt'].strip()}\n{reference_hint}{_PROMPT_SUFFIX}"

        return Sample(
            input=prompt,
            target='',
            tools=[BASH_TOOL_INFO, PYTHON_EXEC_TOOL_INFO],
            metadata={
                'task_id': record.get('task_id'),
                'sector': record.get('sector'),
                'occupation': record.get('occupation'),
                'prompt': record.get('prompt'),
                'reference_files': reference_files,
                'reference_file_urls': reference_file_urls,
                'reference_file_hf_uris': reference_file_hf_uris,
                'reference_paths': [remove_reference_hash(path) for path in reference_files],
                'sandbox_reference_paths': sandbox_reference_paths,
                'rubric_pretty': record.get('rubric_pretty'),
                'rubric_json': record.get('rubric_json'),
                'dataset_id': self.dataset_id,
                'dataset_hub': self.source_dataset_hub,
            },
        )

    def _post_process_samples(self) -> None:
        if self.download_reference_files and not is_build_doc():
            for subset_samples in self.test_dataset.values():
                self._resolve_sample_reference_files(list(subset_samples))
        self._submission_records = submission_records_from_samples(self.test_dataset, self.default_subset)
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
        volumes = build_reference_volumes(sample)
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
        return GDPvalArtifactEnvironment(
            env=env,
            artifact_dir=artifact_dir(sample, self._current_output_dir, self.name),
            metadata=sample.metadata,
        )

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
            'deliverable_count': len(deliverables),
            'artifact_dir': task_state.metadata.get('artifact_dir'),
            'official_gdpval_score': None,
            'official_gdpval_score_note': (
                'GDPval official grading is external. Use OpenAI\'s official GDPval judge on the exported submission.'
            ),
        }
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=value,
            metadata=metadata,
            main_score_name='submission_ready',
        )
        return score

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = self.match_score(
            original_prediction=original_prediction,
            filtered_prediction=filtered_prediction,
            reference=reference,
            task_state=task_state,
        )
        score.metadata['judge_strategy_note'] = (
            'GDPval does not use EvalScope local LLM judging. Use OpenAI\'s official GDPval judge on the exported '
            'submission package.'
        )
        return score

    def _on_generate_report_end(self, report: Any, output_dir: str, **kwargs: Any) -> None:
        if is_build_doc():
            return
        self._export_submission(Path(output_dir))

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

    def _ensure_docker_image(self) -> None:
        if self._docker_image_checked or is_build_doc():
            return
        self._docker_image_checked = True

        if not self.auto_build_docker_image or self.docker_image != _DEFAULT_DOCKER_IMAGE:
            return

        build_ctx, dockerfile = self.get_build_context()
        result = prepare_docker_image(
            DockerImageSpec(
                name_prefix='evalscope/gdpval',
                context_dir=build_ctx,
                dockerfile=dockerfile,
                cache_key_parts=[self.name, 'gdpval'],
            )
        )
        self.docker_image = result.image_tag
        logger.info(f'GDPval docker image prepared: {result.image_tag} (reused={result.reused})')

    def _export_submission(self, report_dir: Path) -> None:
        export_submission(
            report_dir=report_dir,
            submission_records=self._submission_records,
            subset_list=self.subset_list,
            benchmark_name=self.name,
            dataset_id=self.dataset_id,
            dataset_hub=self.source_dataset_hub,
        )
