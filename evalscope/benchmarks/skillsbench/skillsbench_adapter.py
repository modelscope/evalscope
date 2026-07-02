from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.agent.skills import ResolvedSkills, discover_skills
from evalscope.agent.tools.bash import BASH_TOOL_INFO
from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.evaluator import InferenceResult, TaskState
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import DockerImageBuilder, DockerImageSpec
from evalscope.constants import Tags
from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.import_utils import check_import, is_build_doc
from evalscope.utils.logger import get_logger

logger = get_logger()

_DEFAULT_SUBSET = 'default'
_SKILL_MODE_NO_SKILL = 'no-skill'
_SKILL_MODE_WITH_SKILL = 'with-skill'
_SKILL_MODES = {_SKILL_MODE_NO_SKILL, _SKILL_MODE_WITH_SKILL}
_SKILLS_SANDBOX_DIR = '/skills'
_COMMON_SKILL_INSTALL_PATH = '$HOME/.agents/skills'
_VERIFIER_DIR = '/verifier'
_ORACLE_DIR = '/oracle'
_LOGS_VERIFIER_DIR = '/logs/verifier'

_AGENT_SKILL_PATH_MARKERS = (
    '/.codex/skills',
    '/.agents/skills',
    '/.claude/skills',
    '/.gemini/skills',
    '/.goose/skills',
    '/.factory/skills',
    '/.opencode/skill',
    '/.config/opencode/skills',
)

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
        'value': _SKILL_MODE_NO_SKILL,
        'choices': [_SKILL_MODE_NO_SKILL, _SKILL_MODE_WITH_SKILL],
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

SkillsBench evaluates whether agents can use task-bundled Agent Skills. EvalScope runs each task inside a Docker image
built from the task's `environment/Dockerfile`, then executes the task's verifier script and reads
`/logs/verifier/reward.txt`.

## Evaluation Notes

- Default `skill_mode` is `no-skill`.
- Run `no-skill` and `with-skill` separately, then compare runs with EvalScope's run comparison tools.
- `self-gen` and `tasks-extra` are not supported by this adapter version.
- The verifier executes the task's `verifier/test.sh` and may install dependencies from the network.
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
        self.task_ids = _as_str_list(self.extra_params.get('task_ids'))
        self.skill_mode = str(self.extra_params.get('skill_mode') or _SKILL_MODE_NO_SKILL)
        if self.skill_mode not in _SKILL_MODES:
            raise ValueError(f'skill_mode must be one of: {sorted(_SKILL_MODES)}')
        self.force_rebuild = bool(self.extra_params.get('force_rebuild', False))
        self.agent_timeout_override = _optional_float(self.extra_params.get('agent_timeout_sec'))
        self.verifier_timeout_override = _optional_float(self.extra_params.get('verifier_timeout_sec'))
        self.runner = str(self.extra_params.get('runner') or 'agent')
        self._image_builder = DockerImageBuilder()
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
            dirs = sorted(path for path in self.tasks_dir.iterdir() if path.is_dir())
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
        frontmatter, prompt = _read_task_md(task_dir / 'task.md')
        task_id = task_dir.name
        agent_timeout = self.agent_timeout_override or _nested_float(frontmatter, ['agent', 'timeout_sec']) or 900.0
        verifier_timeout = (
            self.verifier_timeout_override or _nested_float(frontmatter, ['verifier', 'timeout_sec']) or 900.0
        )
        skills_dir = task_dir / 'environment' / 'skills'
        skill_metadata, skill_errors = discover_skills(skills_dir, path_prefix=f'{_COMMON_SKILL_INSTALL_PATH}')
        agent_skills = ResolvedSkills(
            enabled=self.skill_mode == _SKILL_MODE_WITH_SKILL,
            source='task_bundled' if self.skill_mode == _SKILL_MODE_WITH_SKILL else 'none',
            host_dir=str(skills_dir) if self.skill_mode == _SKILL_MODE_WITH_SKILL else None,
            sandbox_dir=_SKILLS_SANDBOX_DIR if self.skill_mode == _SKILL_MODE_WITH_SKILL else None,
            prompt_base_dir=_COMMON_SKILL_INSTALL_PATH if self.skill_mode == _SKILL_MODE_WITH_SKILL else None,
            install_paths=[_COMMON_SKILL_INSTALL_PATH] if self.skill_mode == _SKILL_MODE_WITH_SKILL else [],
            skills=skill_metadata if self.skill_mode == _SKILL_MODE_WITH_SKILL else [],
            metadata_errors=skill_errors,
        )
        image = self._prepare_image(task_dir, task_id)
        metadata: Dict[str, Any] = {
            'task_id': task_id,
            'task_dir': str(task_dir),
            'skill_mode': self.skill_mode,
            'skill_source': agent_skills.source,
            'skills_dir': str(skills_dir) if self.skill_mode == _SKILL_MODE_WITH_SKILL else None,
            'skills_sandbox_dir': agent_skills.sandbox_dir,
            'skill_names': [skill.name for skill in agent_skills.skills],
            'agent_skills': agent_skills.model_dump(),
            'agent_timeout_sec': agent_timeout,
            'verifier_timeout_sec': verifier_timeout,
            'image_tag': image.image_tag,
            'image_reused': image.reused,
            'image_context_hash': image.context_hash,
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
        context_dir = _stage_environment_context(
            task_dir=task_dir,
            skill_mode=self.skill_mode,
        )
        self._build_contexts.append(context_dir)
        return self._image_builder.build_or_reuse(
            DockerImageSpec(
                name_prefix=f'evalscope-skillsbench-{task_id}-{self.skill_mode}',
                context_dir=context_dir,
                dockerfile='Dockerfile',
                cache_key_parts=['skillsbench', task_id, self.skill_mode],
                force_rebuild=self.force_rebuild,
            )
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

        if not isinstance(self._task_config.agent_config, ExternalAgentConfig):
            raise RuntimeError('SkillsBench v1 supports external agent_config or dataset_args.runner="oracle".')

        env = self._build_environment(sample)
        wrapper = _NoCloseEnvironment(env)
        try:
            result = run_external_agent(
                config=self._task_config.agent_config,
                model=model,
                sample=sample,
                environment_override=wrapper,
                instruction_override=_sample_instruction(sample),
            )
            AsyncioLoopRunner.run(self._run_verifier(env, sample))
            return result
        finally:
            AsyncioLoopRunner.run(env.close())

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
                messages=[ChatMessageUser(content=_sample_instruction(sample)),
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

    def _build_environment(self, sample: Sample) -> '_SkillsBenchDockerEnvironment':
        check_import('docker', package='docker', raise_error=True, feature_name='SkillsBench Docker runner')
        metadata = sample.metadata
        return _SkillsBenchDockerEnvironment(
            image=str(metadata['image_tag']),
            network_enabled=_network_enabled(metadata.get('frontmatter') or {}),
            labels={
                'evalscope.benchmark': 'skillsbench',
                'evalscope.task_id': str(metadata.get('task_id') or ''),
            },
        )

    async def _run_verifier(self, env: '_SkillsBenchDockerEnvironment', sample: Sample) -> None:
        task_dir = Path(sample.metadata['task_dir'])
        await env.put_dir(task_dir / 'verifier', _VERIFIER_DIR)
        await env.exec(['bash', '-lc', f'rm -rf {_LOGS_VERIFIER_DIR} && mkdir -p {_LOGS_VERIFIER_DIR}'], timeout=30)
        verifier = await env.exec(
            ['bash', '-lc', f'bash {_VERIFIER_DIR}/test.sh > {_LOGS_VERIFIER_DIR}/test-stdout.txt 2>&1'],
            timeout=sample.metadata['verifier_timeout_sec'],
        )
        sample.metadata['verifier_returncode'] = verifier.returncode
        sample.metadata['verifier_timed_out'] = verifier.timed_out
        stdout = await env.read_text(f'{_LOGS_VERIFIER_DIR}/test-stdout.txt')
        sample.metadata['verifier_stdout_tail'] = stdout[-8000:]
        reward_text = await env.read_text(f'{_LOGS_VERIFIER_DIR}/reward.txt')
        ctrf_text = await env.read_text(f'{_LOGS_VERIFIER_DIR}/ctrf.json')
        self._save_verifier_artifacts(sample, stdout=stdout, reward_text=reward_text, ctrf_text=ctrf_text)
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
            / _safe_path_part(str(sample.metadata.get('task_id') or sample.id))
            / _safe_path_part(str(sample.metadata.get('skill_mode') or self.skill_mode))
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


class _SkillsBenchDockerEnvironment(AgentEnvironment):
    name = 'docker'

    def __init__(self, *, image: str, network_enabled: bool, labels: Dict[str, str]) -> None:
        self._image = image
        self._network_enabled = network_enabled
        self._labels = labels
        self._client = None
        self._container = None

    async def _ensure_container(self) -> Any:
        if self._container is None:
            from docker import from_env

            self._client = from_env()
            kwargs: Dict[str, Any] = {
                'image': self._image,
                'command': 'sleep infinity',
                'detach': True,
                'labels': self._labels,
            }
            if not self._network_enabled:
                kwargs['network_disabled'] = True
            else:
                kwargs['extra_hosts'] = {'host.docker.internal': 'host-gateway'}
            self._container = await asyncio.to_thread(self._client.containers.run, **kwargs)
        return self._container

    async def exec(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        input: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        container = await self._ensure_container()
        command = _wrap_timeout_command(_wrap_input_command(cmd, input), timeout)
        wait_timeout = timeout + 15 if timeout is not None else None

        def _run() -> Any:
            return container.exec_run(
                command,
                workdir=cwd,
                environment=env,
                demux=True,
            )

        started = time.monotonic()
        try:
            result = await asyncio.wait_for(asyncio.to_thread(_run), timeout=wait_timeout)
        except asyncio.TimeoutError:
            await self.close()
            return ExecResult(returncode=-1, timed_out=True, duration=time.monotonic() - started)
        duration = time.monotonic() - started
        stdout_b, stderr_b = result.output if isinstance(result.output, tuple) else (result.output, b'')
        returncode = int(result.exit_code or 0)
        return ExecResult(
            returncode=returncode,
            stdout=(stdout_b or b'').decode('utf-8', errors='replace'),
            stderr=(stderr_b or b'').decode('utf-8', errors='replace'),
            timed_out=returncode in {124, 137},
            duration=duration,
        )

    async def put_dir(self, source_dir: Path, target_dir: str) -> None:
        container = await self._ensure_container()
        if not source_dir.is_dir():
            raise FileNotFoundError(f'Directory not found: {source_dir}')
        mkdir = await self.exec(['mkdir', '-p', target_dir], timeout=30)
        if mkdir.returncode != 0:
            raise RuntimeError(f'Failed to create {target_dir}: {mkdir.stderr}')
        data = _tar_directory(source_dir)
        ok = await asyncio.to_thread(container.put_archive, target_dir, data)
        if not ok:
            raise RuntimeError(f'Failed to copy {source_dir} into container:{target_dir}')

    async def read_text(self, path: str) -> str:
        result = await self.exec(['bash', '-lc', f'cat {sh_quote(path)} 2>/dev/null || true'], timeout=30)
        return result.stdout or ''

    async def close(self) -> None:
        if self._container is not None:
            container = self._container
            self._container = None
            try:
                await asyncio.to_thread(container.remove, force=True)
            except Exception as exc:
                logger.warning(f'Failed to remove SkillsBench container: {exc}')
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


class _NoCloseEnvironment(AgentEnvironment):
    name = 'docker'

    def __init__(self, env: _SkillsBenchDockerEnvironment) -> None:
        self._env = env

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
        return None


def _stage_environment_context(*, task_dir: Path, skill_mode: str) -> str:
    env_dir = task_dir / 'environment'
    if not env_dir.is_dir():
        raise FileNotFoundError(f'SkillsBench environment directory not found: {env_dir}')
    tmp = tempfile.mkdtemp(prefix=f'evalscope-skillsbench-{task_dir.name}-')
    shutil.copytree(env_dir, tmp, dirs_exist_ok=True)
    dockerfile = Path(tmp) / 'Dockerfile'
    if not dockerfile.is_file():
        raise FileNotFoundError(f'SkillsBench Dockerfile not found: {dockerfile}')
    if skill_mode == _SKILL_MODE_NO_SKILL:
        shutil.rmtree(Path(tmp) / 'skills', ignore_errors=True)
        shutil.rmtree(Path(tmp) / '_deps' / 'skills', ignore_errors=True)
        _rewrite_dockerfile_without_skill_copies(dockerfile)
        _assert_no_skill_path_residue(dockerfile, include_neutral=True)
    else:
        _rewrite_dockerfile_without_skill_copies(dockerfile)
        _assert_no_skill_path_residue(dockerfile, include_neutral=False)
        content = dockerfile.read_text(encoding='utf-8')
        dockerfile.write_text(
            f'{content.rstrip()}\n\n# SkillsBench skill injection.\nCOPY skills /skills\n', encoding='utf-8'
        )
    return tmp


def _rewrite_dockerfile_without_skill_copies(dockerfile: Path) -> None:
    kept = []
    for line in dockerfile.read_text(encoding='utf-8').splitlines():
        if _is_skill_copy_line(line):
            continue
        kept.append(line)
    dockerfile.write_text('\n'.join(kept).rstrip() + '\n', encoding='utf-8')


def _is_skill_copy_line(line: str) -> bool:
    return bool(re.search(r'^\s*COPY\s+(_deps/)?skills\b', line))


def _assert_no_skill_path_residue(dockerfile: Path, *, include_neutral: bool) -> None:
    markers = list(_AGENT_SKILL_PATH_MARKERS)
    if include_neutral:
        markers.append('/skills')
    for lineno, line in enumerate(dockerfile.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if any(marker in line for marker in markers):
            raise ValueError(f'Skill path residue in {dockerfile}:{lineno}: {line}')


def _read_task_md(path: Path) -> tuple[Dict[str, Any], str]:
    text = path.read_text(encoding='utf-8')
    if not text.startswith('---'):
        return {}, text.strip()
    end = text.find('\n---', 3)
    if end < 0:
        return {}, text.strip()
    frontmatter_text = text[3:end]
    body = text[text.find('\n', end + 1) + 1:].strip()
    return _parse_simple_yaml(frontmatter_text), body


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith('#') or ':' not in raw:
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        key, value = raw.strip().split(':', 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value == '':
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    text = value.strip().strip('"\'')
    if text.lower() in {'true', 'false'}:
        return text.lower() == 'true'
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _nested_float(data: Dict[str, Any], keys: List[str]) -> Optional[float]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return _optional_float(current)


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ''):
        return None
    return float(value)


def _network_enabled(frontmatter: Dict[str, Any]) -> bool:
    mode = (((frontmatter.get('environment') or {}) if isinstance(frontmatter, dict) else {}).get('network_mode'))
    if isinstance(mode, str) and mode.lower() in {'none', 'disabled', 'off'}:
        return False
    return True


def _as_str_list(value: Any) -> List[str]:
    if value is None or value == '':
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _sample_instruction(sample: Sample) -> str:
    if isinstance(sample.input, str):
        return sample.input
    return '\n\n'.join(getattr(message, 'text', '') or '' for message in sample.input)


def _tar_directory(source_dir: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as tar:
        for path in source_dir.rglob('*'):
            tar.add(path, arcname=path.relative_to(source_dir).as_posix())
    stream.seek(0)
    return stream.getvalue()


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _safe_path_part(value: str) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '-', value).strip('.-')
    return text or 'unknown'


def _wrap_timeout_command(cmd: List[str], timeout: Optional[float]) -> List[str]:
    if timeout is None or timeout <= 0:
        return cmd
    seconds = max(1, int(timeout))
    quoted_cmd = ' '.join(sh_quote(part) for part in cmd)
    return ['bash', '-lc', f'exec timeout --kill-after=5s {seconds}s {quoted_cmd}']


def _wrap_input_command(cmd: List[str], input: Optional[str]) -> List[str]:
    if input is None:
        return cmd
    quoted_cmd = ' '.join(sh_quote(part) for part in cmd)
    return ['bash', '-lc', f'printf %s {sh_quote(input)} | {quoted_cmd}']
