from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

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
from evalscope.api.sandbox import merge_sandbox_config_dicts
from evalscope.constants import HubType, Tags
from evalscope.utils.import_utils import is_build_doc
from evalscope.utils.logger import get_logger
from .utils import (
    SANDBOX_OUTPUT_DIR,
    SANDBOX_REFERENCE_DIR,
    JobBenchArtifactEnvironment,
    artifact_dir,
    evaluate_job_bench_output,
    parse_rubrics,
)

logger = get_logger()

_DATASET_ID = 'evalscope/job-bench'
_DEFAULT_DOCKER_IMAGE = 'python:3.11-slim-bookworm'

_DESCRIPTION = """
## Overview

JobBench evaluates agentic systems on realistic professional work tasks that require reading reference files, producing
deliverables, and reconciling multi-source information. This adapter uses the ModelScope dataset
`evalscope/job-bench`.

## Task Description

- **Task Type**: Agentic professional work / deliverable generation
- **Input**: Workplace-style task prompt with optional reference files under `reference_files/`
- **Output**: Final deliverable files written to `jobbench_output/`
- **Dataset**: ModelScope `evalscope/job-bench`
- **Metric**: Weighted LLM-judge rubric score (`normalized_score`)

## Evaluation Notes

- The default evaluation split is `main`.
- Configure `judge_model_args` for rubric scoring.
- `normalized_score` is the primary weighted score (`total_score / max_score`); `pass_rate` is the unweighted
  proportion of fully passed rubrics; `total_score` is the raw sum of passed rubric weights.
- Docker runs use `python:3.11-slim-bookworm` by default. For formal evaluation, provide an image with the Office,
  PDF, and spreadsheet tools required by the tasks.
"""

_PROMPT_SUFFIX = f"""

Reference files, when present, are available under `{SANDBOX_REFERENCE_DIR}/`.
Use the bash or python_exec tools to inspect inputs and create the requested deliverables.

Write every final deliverable file under `{SANDBOX_OUTPUT_DIR}`. Do not put intermediate scratch files there.
Your final message may summarize what you produced, but files requested by the task must be actual files in
`{SANDBOX_OUTPUT_DIR}`.
"""


@register_benchmark(
    BenchmarkMeta(
        name='job_bench',
        pretty_name='JobBench',
        tags=[Tags.AGENT, Tags.KNOWLEDGE, Tags.MULTI_TURN],
        description=_DESCRIPTION,
        dataset_id=_DATASET_ID,
        subset_list=['default'],
        default_subset='default',
        eval_split='main',
        prompt_template='{question}',
        metric_list=['normalized_score', 'pass_rate', 'total_score'],
    )
)
class JobBenchAdapter(AgentLoopAdapter):
    """JobBench adapter using ModelScope data and EvalScope's agent loop."""

    llm_judge_default = True
    strategy_name = 'function_calling'
    max_steps_default = 250

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot_dir: Optional[Path] = None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        if self._snapshot_dir is None and not is_build_doc():
            self._snapshot_dir = Path(
                DatasetHub(
                    data_id_or_path=self.dataset_id,
                    data_source=self.dataset_hub or HubType.MODELSCOPE,
                    force_redownload=self.force_redownload,
                    cache_dir=self.dataset_dir,
                ).download_snapshot()
            ).resolve()

        reference_files = [str(path) for path in record.get('reference_files') or []]
        reference_hint = ''
        if reference_files:
            reference_hint = '\n\nReference files for this task:\n' + '\n'.join(
                f'- {SANDBOX_REFERENCE_DIR}/{Path(path).name}' for path in reference_files
            )

        metadata = {
            'task_id': record['task_id'],
            'reference_files': reference_files,
            'rubric_json': record.get('rubric_json'),
        }
        if self._snapshot_dir is not None and reference_files:
            reference_dir = (self._snapshot_dir / Path(reference_files[0]).parent).resolve()
            if reference_dir.is_relative_to(self._snapshot_dir) and reference_dir.is_dir():
                metadata['reference_dir'] = str(reference_dir)
            else:
                logger.warning(f'JobBench reference directory not found: {reference_dir}')

        return Sample(
            input=f"{record['prompt'].strip()}{reference_hint}{_PROMPT_SUFFIX}",
            tools=[BASH_TOOL_INFO, PYTHON_EXEC_TOOL_INFO],
            metadata=metadata,
        )

    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs: Any) -> TaskState:
        sample.metadata['artifact_dir'] = str(artifact_dir(sample, output_dir))
        return super().run_inference(model, sample, output_dir, **kwargs)

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        return {
            'bash': run_bash,
            'python_exec': run_python_exec,
        }

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        agent_config = self._task_config.agent_config if self._task_config is not None else None
        if getattr(agent_config, 'environment', None) == 'docker':
            return self._build_docker_environment(sample)

        from evalscope.agent.environments.local import TemporaryLocalAgentEnvironment

        env = TemporaryLocalAgentEnvironment(sample_id=sample.id, prefix='evalscope-jobbench-')
        reference_dir = sample.metadata.get('reference_dir')
        if reference_dir:
            shutil.copytree(reference_dir, env.working_dir / SANDBOX_REFERENCE_DIR)
        host_artifact_dir = Path(sample.metadata['artifact_dir'])
        host_output_dir = host_artifact_dir / SANDBOX_OUTPUT_DIR
        host_output_dir.mkdir(parents=True, exist_ok=True)
        (env.working_dir / SANDBOX_OUTPUT_DIR).symlink_to(host_output_dir, target_is_directory=True)
        return JobBenchArtifactEnvironment(
            env=env,
            artifact_dir=host_artifact_dir,
            metadata=sample.metadata,
        )

    def _build_docker_environment(self, sample: Sample) -> AgentEnvironment:
        from evalscope.agent.environments.enclave import EnclaveAgentEnvironment

        defaults: Dict[str, Any] = {
            'image': _DEFAULT_DOCKER_IMAGE,
            'working_dir': '/workspace',
            'network_enabled': True,
        }
        sandbox_config = merge_sandbox_config_dicts(defaults, self._task_sandbox_config())
        host_artifact_dir = Path(sample.metadata['artifact_dir'])
        host_output_dir = host_artifact_dir / SANDBOX_OUTPUT_DIR
        host_output_dir.mkdir(parents=True, exist_ok=True)
        volumes = {
            str(host_output_dir): {
                'bind': f'/workspace/{SANDBOX_OUTPUT_DIR}',
                'mode': 'rw',
            }
        }
        reference_dir = sample.metadata.get('reference_dir')
        if reference_dir and Path(reference_dir).is_dir():
            volumes[reference_dir] = {
                'bind': f'/workspace/{SANDBOX_REFERENCE_DIR}',
                'mode': 'ro',
            }
        sandbox_config['volumes'] = {**sandbox_config.get('volumes', {}), **volumes}

        env = EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config=sandbox_config,
            timeout=self._native_command_timeout(),
        )
        return JobBenchArtifactEnvironment(
            env=env,
            artifact_dir=host_artifact_dir,
            metadata=sample.metadata,
        )

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                'normalized_score': 0.0,
                'pass_rate': 0.0,
                'total_score': 0.0,
            },
            metadata={
                'official_score_computed': False,
                'note': 'JobBench quality scoring requires judge_model_args.',
            },
            main_score_name='normalized_score',
        )

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        if self.llm_judge is None:
            raise ValueError('JobBench requires judge_model_args for LLM-based rubric scoring.')

        evaluation = evaluate_job_bench_output(
            output_dir=Path(task_state.metadata['artifact_dir']) / SANDBOX_OUTPUT_DIR,
            rubrics=parse_rubrics(task_state.metadata.get('rubric_json')),
            judge=self.llm_judge.judge,
        )
        scorecard = evaluation['scorecard']
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                'normalized_score': float(scorecard['normalized_score']),
                'pass_rate': float(scorecard['pass_rate']),
                'total_score': float(scorecard['total_score']),
            },
            metadata={
                'judge_model': self.llm_judge.model_id,
                'task_id': task_state.metadata.get('task_id'),
                'artifact_dir': task_state.metadata.get('artifact_dir'),
                'output_files': task_state.metadata.get('output_files') or [],
                'max_score': float(scorecard['max_score']),
                'rubrics': evaluation['rubrics'],
                'error': evaluation.get('error'),
            },
            main_score_name='normalized_score',
        )
