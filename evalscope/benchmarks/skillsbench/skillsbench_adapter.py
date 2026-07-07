from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
from evalscope.agent.skills import (
    DEFAULT_SKILLS_INSTALL_DIR,
    TASK_BUNDLED_SKILL_SOURCE,
    ResolvedSkills,
    discover_skills,
)
from evalscope.agent.tools.bash import BASH_TOOL_INFO
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import DockerImageSpec, prepare_docker_image
from evalscope.constants import Tags
from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.import_utils import is_build_doc
from .utils import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_WITH_SKILL,
    SKILL_MODES,
    SKILLS_SANDBOX_DIR,
    as_str_list,
    dockerfile_workdir,
    nested_float,
    optional_float,
    read_sandbox_text,
    read_task_md,
    safe_path_part,
    sample_instruction,
    skillsbench_sandbox_config,
    stage_environment_context,
)

_DEFAULT_SUBSET = 'default'
_VERIFIER_DIR = '/verifier'
_ORACLE_DIR = '/oracle'
_LOGS_VERIFIER_DIR = '/logs/verifier'

_EXTRA_PARAMS: Dict[str, Any] = {
    'tasks_dir': {
        'type': 'str',
        'description': 'Path to the SkillsBench tasks directory.',
        'value': '',
    },
    'task_ids': {
        'type': 'list',
        'description': 'Optional list of task ids to run.',
        'value': [],
    },
    'skill_mode': {
        'type': 'str',
        'description': 'SkillsBench skill mode.',
        'value': SKILL_MODE_NO_SKILL,
        'choices': [SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL],
    },
    'force_rebuild': {
        'type': 'bool',
        'description': 'Force rebuilding task Docker images.',
        'value': False,
    },
    'agent_timeout_sec': {
        'type': 'float',
        'description': 'Override task agent timeout.',
        'value': None,
    },
    'verifier_timeout_sec': {
        'type': 'float',
        'description': 'Override task verifier timeout.',
        'value': None,
    },
    'runner': {
        'type': 'str',
        'description': 'Use "oracle" to run oracle/solve.sh instead of an agent.',
        'value': 'agent',
        'choices': ['agent', 'oracle'],
    },
}

_DESCRIPTION = """
## Overview

SkillsBench evaluates whether coding agents can discover and apply task-bundled Agent Skills. Each task contains an
instruction, an optional skill directory, a Docker environment, an oracle solution, and a verifier. EvalScope builds the
task Docker image, runs the selected agent or oracle in that image, then executes the task verifier.

## Task Description

- **Task Type**: Agent skill usage / tool-assisted task completion
- **Input**: The natural-language task prompt from `task.md`
- **Output**: Files or state changes produced inside the task container, scored by the task verifier
- **Dataset**: Local SkillsBench task repository supplied through `extra_params.tasks_dir`
- **Environment**: Per-task Docker image built from `environment/Dockerfile`
- **Skills**: Optional task-bundled skills from `environment/skills`
- **Metric**: `score` from `/logs/verifier/reward.txt`; `success` is 1 when `score > 0`

## Key Features

- Builds or reuses a content-hashed Docker image for each selected task.
- Runs the task in `no-skill` or `with-skill` mode without mixing the two conditions in one EvalScope run.
- Injects task-bundled skills through EvalScope's agent skill runtime instead of baking runner-specific skill paths into
  the image.
- Supports EvalScope native agents and external agent runners through the shared agent environment interface.
- Saves verifier stdout, reward, and optional CTRF artifacts under the run output directory.

## Evaluation Notes

- Default `skill_mode` is `no-skill`.
- Run `no-skill` and `with-skill` separately, then compare runs with EvalScope's run comparison tools.
- `self-gen` and `tasks-extra` are not supported by this adapter version.
- `runner='oracle'` runs the official `oracle/solve.sh` for smoke testing; agent runs require `agent_config`.
- The verifier executes `verifier/test.sh` and may install dependencies from the network.

## Scoring and Comparison

- `score` is the verifier reward parsed from `/logs/verifier/reward.txt`.
- `success` is derived locally as `1.0` when `score > 0`, otherwise `0.0`.
- EvalScope does not automatically compute the no-skill versus with-skill delta; run both modes and compare their reports.
"""


@register_benchmark(
    BenchmarkMeta(
        name='skillsbench',
        pretty_name='SkillsBench',
        tags=[Tags.AGENT, Tags.MULTI_TURN],
        description=_DESCRIPTION,
        dataset_id='skillsbench',
        subset_list=[_DEFAULT_SUBSET],
        default_subset=_DEFAULT_SUBSET,
        metric_list=['score'],
        prompt_template='{question}',
        extra_params=_EXTRA_PARAMS,
    )
)
class SkillsBenchAdapter(AgentAdapter):
    """SkillsBench adapter for the current task.md + verifier/test.sh contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tasks_dir = Path(str(self.extra_params.get('tasks_dir') or self.dataset_id)).expanduser()
        self.task_ids = as_str_list(self.extra_params.get('task_ids'))
        self.skill_mode = str(self.extra_params.get('skill_mode') or SKILL_MODE_NO_SKILL)
        if self.skill_mode not in SKILL_MODES:
            raise ValueError(f'skill_mode must be one of: {sorted(SKILL_MODES)}')
        self.force_rebuild = bool(self.extra_params.get('force_rebuild', False))
        self.agent_timeout_override = optional_float(self.extra_params.get('agent_timeout_sec'))
        self.verifier_timeout_override = optional_float(self.extra_params.get('verifier_timeout_sec'))
        self.runner = str(self.extra_params.get('runner') or 'agent')
        self._build_contexts: List[str] = []
        self._current_output_dir: Optional[str] = None

    def load_dataset(self) -> DatasetDict:
        if is_build_doc():
            return DatasetDict({_DEFAULT_SUBSET: MemoryDataset([], name=_DEFAULT_SUBSET)})
        if not self.tasks_dir.is_dir():
            raise FileNotFoundError(
                f'SkillsBench tasks_dir not found: {self.tasks_dir}. '
                'Set dataset_args.tasks_dir to the local skillsbench/tasks path.'
            )
        task_dirs = self._select_task_dirs()
        samples = [self._task_to_sample(task_dir, idx) for idx, task_dir in enumerate(task_dirs)]
        self.test_dataset = DatasetDict({_DEFAULT_SUBSET: MemoryDataset(samples, name=_DEFAULT_SUBSET)})
        self._post_process_samples()
        return self.test_dataset

    def _select_task_dirs(self) -> List[Path]:
        if self.task_ids:
            dirs = [self.tasks_dir / task_id for task_id in self.task_ids]
        else:
            dirs = sorted(path for path in self.tasks_dir.iterdir() if path.is_dir() and not path.name.startswith('.'))
        missing = [str(path) for path in dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f'SkillsBench task directories not found: {missing}')
        limit = self.limit
        if isinstance(limit, int):
            dirs = dirs[:limit]
        elif isinstance(limit, float) and 0 < limit < 1:
            dirs = dirs[:max(1, int(len(dirs) * limit))]
        return dirs

    def _task_to_sample(self, task_dir: Path, index: int) -> Sample:
        frontmatter, prompt = read_task_md(task_dir / 'task.md')
        task_id = task_dir.name
        agent_timeout = self.agent_timeout_override or nested_float(frontmatter, ['agent', 'timeout_sec']) or 900.0
        verifier_timeout = (
            self.verifier_timeout_override or nested_float(frontmatter, ['verifier', 'timeout_sec']) or 900.0
        )
        skills_dir = task_dir / 'environment' / 'skills'
        skill_metadata, skill_errors = discover_skills(skills_dir, path_prefix=DEFAULT_SKILLS_INSTALL_DIR)
        agent_skills = ResolvedSkills(
            enabled=self.skill_mode == SKILL_MODE_WITH_SKILL,
            source=TASK_BUNDLED_SKILL_SOURCE if self.skill_mode == SKILL_MODE_WITH_SKILL else 'none',
            host_dir=str(skills_dir) if self.skill_mode == SKILL_MODE_WITH_SKILL else None,
            sandbox_dir=SKILLS_SANDBOX_DIR if self.skill_mode == SKILL_MODE_WITH_SKILL else None,
            prompt_base_dir=DEFAULT_SKILLS_INSTALL_DIR if self.skill_mode == SKILL_MODE_WITH_SKILL else None,
            install_paths=[DEFAULT_SKILLS_INSTALL_DIR] if self.skill_mode == SKILL_MODE_WITH_SKILL else [],
            skills=skill_metadata if self.skill_mode == SKILL_MODE_WITH_SKILL else [],
            metadata_errors=skill_errors,
        )
        image = self._prepare_image(task_dir, task_id)
        working_dir = dockerfile_workdir(task_dir / 'environment' / 'Dockerfile')
        metadata: Dict[str, Any] = {
            'task_id': task_id,
            'task_dir': str(task_dir),
            'skill_mode': self.skill_mode,
            'skill_source': agent_skills.source,
            'skills_dir': str(skills_dir) if self.skill_mode == SKILL_MODE_WITH_SKILL else None,
            'skills_sandbox_dir': agent_skills.sandbox_dir,
            'skill_names': [skill.name for skill in agent_skills.skills],
            'agent_skills': agent_skills.model_dump(),
            'agent_timeout_sec': agent_timeout,
            'verifier_timeout_sec': verifier_timeout,
            'image_tag': image.image_tag,
            'image_reused': image.reused,
            'image_context_hash': image.context_hash,
            'working_dir': working_dir,
            'frontmatter': frontmatter,
        }
        return Sample(
            id=index,
            input=prompt,
            target='',
            tools=[BASH_TOOL_INFO],
            metadata=metadata,
        )

    def _prepare_image(self, task_dir: Path, task_id: str) -> Any:
        context_dir = stage_environment_context(
            task_dir=task_dir,
            skill_mode=self.skill_mode,
        )
        self._build_contexts.append(context_dir)
        return prepare_docker_image(
            DockerImageSpec(
                name_prefix=f'evalscope-skillsbench-{task_id}-{self.skill_mode}',
                context_dir=context_dir,
                dockerfile='Dockerfile',
                cache_key_parts=['skillsbench', task_id, self.skill_mode],
                force_rebuild=self.force_rebuild,
            ),
        )

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        if self.runner == 'oracle':
            return AsyncioLoopRunner.run(self._run_oracle(model, sample))
        if self._task_config is None or self._task_config.agent_config is None:
            raise RuntimeError(
                'SkillsBench requires agent_config, or set dataset_args.runner="oracle" for oracle smoke.'
            )

        from evalscope.agent.external.adapter import run_external_agent
        from evalscope.agent.external.config import ExternalAgentConfig
        from evalscope.agent.runner import run_native_agent
        from evalscope.api.agent.types import NativeAgentConfig

        agent_config = self._task_config.agent_config
        if isinstance(agent_config, ExternalAgentConfig):
            env = self._build_environment(sample)
            try:
                result = run_external_agent(
                    config=agent_config,
                    model=model,
                    sample=sample,
                    environment_override=env,
                    instruction_override=sample_instruction(sample),
                    close_environment=False,
                )
                AsyncioLoopRunner.run(self._run_verifier(env, sample))
                return result
            finally:
                AsyncioLoopRunner.run(env.close())

        if isinstance(agent_config, NativeAgentConfig):
            env = self._build_environment(sample)
            try:
                result = run_native_agent(
                    task_config=self._task_config,
                    model=model,
                    sample=sample,
                    build_sandbox_config=lambda _: None,
                    extract_final_answer=lambda loop_result, strategy: strategy.extract_final_answer(loop_result),
                    environment_override=env,
                )
                AsyncioLoopRunner.run(self._run_verifier(env, sample))
                return result
            finally:
                AsyncioLoopRunner.run(env.close())

        raise RuntimeError('SkillsBench supports native/external agent_config or dataset_args.runner="oracle".')

    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs: Any) -> TaskState:
        self._current_output_dir = output_dir
        try:
            return super().run_inference(model, sample, output_dir, **kwargs)
        finally:
            self._current_output_dir = None

    async def _run_oracle(self, model: Model, sample: Sample) -> InferenceResult:
        env = self._build_environment(sample)
        try:
            await env.put_dir(Path(sample.metadata['task_dir']) / 'oracle', _ORACLE_DIR)
            result = await env.exec(['bash', f'{_ORACLE_DIR}/solve.sh'], timeout=sample.metadata['agent_timeout_sec'])
            sample.metadata['oracle_returncode'] = result.returncode
            sample.metadata['oracle_stdout'] = result.stdout[-4000:]
            sample.metadata['oracle_stderr'] = result.stderr[-4000:]
            if result.returncode != 0:
                sample.metadata['agent_error'] = f'oracle exited with code {result.returncode}'
            await self._run_verifier(env, sample)
            text = f'oracle_returncode={result.returncode}'
            output = ModelOutput(
                model='oracle',
                choices=[ChatCompletionChoice.from_content(text)],
                metadata={'source': 'skillsbench.oracle'},
            )
            return InferenceResult(
                output=output,
                messages=[ChatMessageUser(content=sample_instruction(sample)),
                          ChatMessageAssistant(content=text)],
            )
        finally:
            await env.close()

    def _on_inference_end(
        self,
        model: Model,
        sample: Sample,
        model_output: ModelOutput,
        output_dir: str,
        **kwargs: Any,
    ) -> TaskState:
        return TaskState(
            model=model_output.model or 'skillsbench',
            sample=sample,
            messages=list(sample.input) + [model_output.message] if isinstance(sample.input, list) else [],
            output=model_output,
            completed=True,
        )

    def _build_environment(self, sample: Sample) -> EnclaveAgentEnvironment:
        return EnclaveAgentEnvironment(
            engine='docker',
            sandbox_config=skillsbench_sandbox_config(sample.metadata),
            timeout=60.0,
            interpreter=['bash', '-lc'],
        )

    async def _run_verifier(self, env: EnclaveAgentEnvironment, sample: Sample) -> None:
        task_dir = Path(sample.metadata['task_dir'])
        await env.put_dir(task_dir / 'verifier', _VERIFIER_DIR)
        await env.exec(['bash', '-lc', f'rm -rf {_LOGS_VERIFIER_DIR} && mkdir -p {_LOGS_VERIFIER_DIR}'], timeout=30)
        verifier = await env.exec(
            ['bash', '-lc', f'bash {_VERIFIER_DIR}/test.sh > {_LOGS_VERIFIER_DIR}/test-stdout.txt 2>&1'],
            timeout=sample.metadata['verifier_timeout_sec'],
        )
        sample.metadata['verifier_returncode'] = verifier.returncode
        sample.metadata['verifier_timed_out'] = verifier.timed_out
        stdout = await read_sandbox_text(env, f'{_LOGS_VERIFIER_DIR}/test-stdout.txt')
        sample.metadata['verifier_stdout_tail'] = stdout[-8000:]
        reward_text = await read_sandbox_text(env, f'{_LOGS_VERIFIER_DIR}/reward.txt')
        ctrf_text = await read_sandbox_text(env, f'{_LOGS_VERIFIER_DIR}/ctrf.json')
        self._save_verifier_artifacts(sample, stdout=stdout, reward_text=reward_text, ctrf_text=ctrf_text)
        if verifier.timed_out:
            sample.metadata['verifier_error'] = 'Verifier timed out'
            sample.metadata['reward'] = 0.0
            return
        if not reward_text.strip():
            sample.metadata['verifier_error'] = (
                f'No reward file found after verifier exited with code {verifier.returncode}'
            )
            sample.metadata['reward'] = 0.0
            return
        try:
            reward = float(reward_text.strip())
        except ValueError:
            sample.metadata['verifier_error'] = f'Invalid reward.txt value: {reward_text.strip()!r}'
            sample.metadata['reward'] = 0.0
            return
        sample.metadata['reward'] = reward
        sample.metadata['verifier_error'] = None

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        reward = float(task_state.metadata.get('reward') or 0.0)
        value = {
            'score': reward,
            'success': 1.0 if reward > 0 else 0.0,
        }
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=value,
            metadata={
                'task_id': task_state.metadata.get('task_id'),
                'skill_mode': task_state.metadata.get('skill_mode'),
                'skill_names': task_state.metadata.get('skill_names') or [],
                'image_tag': task_state.metadata.get('image_tag'),
                'image_reused': task_state.metadata.get('image_reused'),
                'verifier_error': task_state.metadata.get('verifier_error'),
                'artifact_dir': task_state.metadata.get('artifact_dir'),
                'verifier_stdout_path': task_state.metadata.get('verifier_stdout_path'),
                'reward_path': task_state.metadata.get('reward_path'),
                'ctrf_path': task_state.metadata.get('ctrf_path'),
            },
            main_score_name='score',
        )

    def _save_verifier_artifacts(
        self,
        sample: Sample,
        *,
        stdout: str,
        reward_text: str,
        ctrf_text: str,
    ) -> None:
        if not self._current_output_dir:
            return
        artifact_dir = (
            Path(self._current_output_dir) / 'artifacts' / 'skillsbench'
            / safe_path_part(str(sample.metadata.get('task_id') or sample.id))
            / safe_path_part(str(sample.metadata.get('skill_mode') or self.skill_mode))
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = artifact_dir / 'test-stdout.txt'
        reward_path = artifact_dir / 'reward.txt'
        stdout_path.write_text(stdout, encoding='utf-8')
        reward_path.write_text(reward_text, encoding='utf-8')
        sample.metadata['artifact_dir'] = str(artifact_dir)
        sample.metadata['verifier_stdout_path'] = str(stdout_path)
        sample.metadata['reward_path'] = str(reward_path)
        if ctrf_text.strip():
            ctrf_path = artifact_dir / 'ctrf.json'
            ctrf_path.write_text(ctrf_text, encoding='utf-8')
            sample.metadata['ctrf_path'] = str(ctrf_path)

    def finalize(self, *args: Any, **kwargs: Any) -> None:
        for context_dir in self._build_contexts:
            shutil.rmtree(context_dir, ignore_errors=True)
        self._build_contexts = []
        super().finalize(*args, **kwargs)
