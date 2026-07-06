import json
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.request import url2pathname

from evalscope.api.agent import AgentTrace, AgentTraceEvent, EventType
from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import DatasetDict, DictDataLoader, Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, HubType, Tags
from evalscope.utils.function_utils import AsyncioLoopRunner
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

DEFAULT_MODELSCOPE_DATASET_ID = 'evalscope/deep-swe'

COMMON_EXTRA_PARAMS = {
    'task_ids': {
        'type': 'list',
        'description': 'Optional list of DeepSWE task ids to evaluate.',
        'value': [],
    },
    'languages': {
        'type': 'list',
        'description': 'Optional task language filter from manifest metadata.',
        'value': [],
    },
    'categories': {
        'type': 'list',
        'description': 'Optional task category filter from manifest metadata.',
        'value': [],
    },
    'sample_seed': {
        'type': 'int',
        'description': 'Optional deterministic shuffle seed applied before limit.',
        'value': '',
    },
    'agent_kwargs': {
        'type': 'dict',
        'description': 'Extra kwargs passed to Pier AgentConfig.',
        'value': {},
    },
}


def _validate_environment_requirements() -> None:
    if shutil.which('docker') is None:
        raise RuntimeError('DeepSWE with environment_type=\'docker\' requires the Docker CLI to be installed.')


class DeepSWEAdapter(AgentAdapter):
    """EvalScope adapter for DeepSWE through Pier Python API jobs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        extra_params = self.extra_params or {}
        self.task_ids = self._as_list(extra_params.get('task_ids') or [])
        self.languages = self._as_list(extra_params.get('languages') or [])
        self.categories = self._as_list(extra_params.get('categories') or [])
        self.sample_seed = extra_params.get('sample_seed')
        self.agent_kwargs = dict(extra_params.get('agent_kwargs') or {})

    @staticmethod
    def _as_list(value: Union[str, List[Any], Tuple[Any, ...]]) -> List[str]:
        if isinstance(value, str):
            return [value] if value else []
        return [str(item) for item in value]

    def load(self) -> Tuple[DatasetDict, None]:
        _validate_environment_requirements()

        snapshot_path = self._download_snapshot()
        task_records = self._load_task_records(snapshot_path)
        task_records = self._filter_task_records(task_records)
        task_records = self._apply_seed(task_records)

        datasets = {
            self.eval_split: DictDataLoader(
                dict_list=task_records,
                limit=self.limit,
                repeats=self.repeats,
                sample_fields=self.record_to_sample,
                shuffle=False,
            ).load()
        }
        return DatasetDict(datasets), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(input=record.get('instruction', ''), target='', metadata=record)

    def _download_snapshot(self) -> Path:
        dataset_id = self.dataset_id
        cache_dir = Path(DEFAULT_EVALSCOPE_CACHE_DIR) / self.name / 'snapshots'
        if self.dataset_hub == HubType.LOCAL or Path(dataset_id).exists():
            logger.info(f'Loading DeepSWE dataset from local path: {dataset_id}')
            return Path(dataset_id)

        if self.dataset_hub == HubType.HUGGINGFACE:
            from huggingface_hub import snapshot_download

            logger.info(f'Downloading DeepSWE dataset from HuggingFace: {dataset_id}')
            return Path(
                snapshot_download(
                    repo_id=dataset_id,
                    repo_type='dataset',
                    cache_dir=str(cache_dir),
                    force_download=self.force_redownload,
                )
            )

        from modelscope import dataset_snapshot_download

        kwargs = {
            'dataset_id': dataset_id,
            'cache_dir': str(cache_dir),
        }
        logger.info(f'Downloading DeepSWE dataset from ModelScope: {dataset_id}')
        return Path(dataset_snapshot_download(**kwargs))

    def _load_task_records(self, snapshot_path: Path) -> List[Dict[str, Any]]:
        tasks_dir = snapshot_path / 'tasks'
        manifest_path = tasks_dir / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'DeepSWE snapshot must contain {manifest_path}')

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest_tasks = manifest.get('tasks')
        if not isinstance(manifest_tasks, list):
            raise ValueError(f'DeepSWE manifest must contain a tasks list: {manifest_path}')

        records = []
        for item in manifest_tasks:
            if not isinstance(item, dict):
                raise ValueError(f'DeepSWE manifest task entries must be objects: {manifest_path}')
            task_id = item.get('task_id') or item.get('id')
            if not task_id:
                raise ValueError(f'DeepSWE manifest task entry missing task_id: {item}')

            task_path = tasks_dir / str(task_id)
            task_toml = task_path / 'task.toml'
            if not task_toml.is_file():
                raise FileNotFoundError(f'DeepSWE task missing task.toml: {task_toml}')

            record = dict(item)
            record['task_id'] = str(task_id)
            record['task_path'] = str(task_path)
            record['task_toml_path'] = str(task_toml)
            record['instruction'] = self._load_instruction(task_path, record)
            records.append(record)

        return records

    @staticmethod
    def _load_instruction(task_path: Path, record: Dict[str, Any]) -> str:
        instruction_path = task_path / 'instruction.md'
        if instruction_path.is_file():
            return instruction_path.read_text(encoding='utf-8')
        return str(record.get('instruction') or record.get('display_description') or record.get('description') or '')

    def _filter_task_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        task_ids = set(self.task_ids)
        languages = {item.lower() for item in self.languages}
        categories = {item.lower() for item in self.categories}

        filtered = []
        for record in records:
            if task_ids and record['task_id'] not in task_ids:
                continue
            if languages and str(record.get('language', '')).lower() not in languages:
                continue
            if categories and str(record.get('category', '')).lower() not in categories:
                continue
            filtered.append(record)
        return filtered

    def _apply_seed(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.sample_seed in (None, ''):
            return records
        shuffled = list(records)
        random.Random(int(self.sample_seed)).shuffle(shuffled)
        return shuffled

    def _on_inference(self, model: Model, sample: Sample) -> InferenceResult:
        result_dict = self._run_pier_job(model, sample)
        sample.metadata['result'] = result_dict

        trial_uri = self._first_trial_result(result_dict).get('trial_uri') or result_dict.get('trial_uri', '')
        output = ModelOutput.from_content(model=model.name, content=trial_uri)
        trace, messages = self._load_pier_trace(result_dict)
        return InferenceResult(output=output, trace=trace, messages=messages)

    def _run_pier_job(self, model: Model, sample: Sample) -> Dict[str, Any]:
        check_import('pier', extra='deep_swe', raise_error=True, feature_name=self.pretty_name)

        from pier.job import Job
        from pier.models.job.config import JobConfig
        from pier.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, VerifierConfig

        task_id = str(sample.metadata['task_id'])

        config = JobConfig(
            job_name=f'{task_id[:48].rstrip("_-")}__{uuid.uuid4().hex[:8]}',
            jobs_dir=Path(self.output_dir) / 'deep_swe_jobs',
            n_attempts=1,
            n_concurrent_trials=1,
            quiet=True,
            timeout_multiplier=1.0,
            agent_timeout_multiplier=1.0,
            verifier_timeout_multiplier=1.0,
            environment_build_timeout_multiplier=1.0,
            agents=[AgentConfig(
                name='mini-swe-agent',
                model_name=model.name,
                kwargs=self.agent_kwargs,
                env={},
            )],
            environment=EnvironmentConfig(type='docker'),
            verifier=VerifierConfig(env={}),
            tasks=[TaskConfig(path=Path(sample.metadata['task_path']))],
        )

        async def _run_job() -> Any:
            job = await Job.create(config)
            return await job.run()

        result = AsyncioLoopRunner.run(_run_job())
        result_dict = result.model_dump(mode='json')
        result_dict['job_result_path'] = str(Path(config.jobs_dir) / config.job_name)
        self._raise_for_pier_failures(result_dict)
        return result_dict

    @staticmethod
    def _raise_for_pier_failures(result_dict: Dict[str, Any]) -> None:
        trial_results = result_dict.get('trial_results') or []
        if not trial_results:
            raise RuntimeError('Pier DeepSWE job did not return any trial results.')

        trial_result = trial_results[0]
        rewards = ((trial_result.get('verifier_result') or {}).get('rewards') or {})
        if rewards.get('reward') is not None:
            return

        exception_info = trial_result.get('exception_info')
        if exception_info:
            exc_type = exception_info.get('exception_type') or exception_info.get('type') or 'UnknownPierError'
            exc_msg = exception_info.get('message') or exception_info.get('exception_message') or str(exception_info)
            raise RuntimeError(f'Pier DeepSWE trial failed with {exc_type}: {exc_msg}')
        raise RuntimeError('Pier DeepSWE trial did not return a reward or exception info.')

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state: Any) -> Score:
        result = task_state.metadata.get('result', {})
        metadata = self._build_score_metadata(result)
        reward = metadata.get('reward')
        acc = float(reward if reward is not None else 0.0)
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'acc': acc},
            metadata=metadata,
        )

    def _build_score_metadata(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        trial_result = self._first_trial_result(result_dict)
        verifier_result = trial_result.get('verifier_result') or {}
        rewards = dict(verifier_result.get('rewards') or {})
        reward_json = self._load_json_if_exists(self._artifact_path(result_dict, 'verifier/reward.json'))
        if reward_json:
            rewards.update(reward_json)

        metadata = {
            'reward': rewards.get('reward'),
            'partial': rewards.get('partial'),
            'f2p': rewards.get('f2p'),
            'p2p': rewards.get('p2p'),
            'apply_failed': rewards.get('apply_failed'),
            'reward_txt': self._load_text_if_exists(self._artifact_path(result_dict, 'verifier/reward.txt')),
            'reward_json': reward_json,
            'trial_result': trial_result,
            'agent_execution_failed': self._is_agent_execution_failed(trial_result),
        }
        metadata.update(self._artifact_metadata(result_dict))
        return metadata

    @staticmethod
    def _first_trial_result(result_dict: Dict[str, Any]) -> Dict[str, Any]:
        trial_results = result_dict.get('trial_results') or []
        if not trial_results:
            return {}
        return trial_results[0] or {}

    @staticmethod
    def _is_agent_execution_failed(trial_result: Dict[str, Any]) -> bool:
        exception_info = trial_result.get('exception_info') or {}
        exception_type = exception_info.get('exception_type') or exception_info.get('type')
        return exception_type == 'NonZeroAgentExitCodeError'

    def _artifact_metadata(self, result_dict: Dict[str, Any]) -> Dict[str, Optional[str]]:
        return {
            'pier_job_result_path': result_dict.get('job_result_path'),
            'trial_uri': self._first_trial_result(result_dict).get('trial_uri') or result_dict.get('trial_uri'),
            'verifier_reward_json_path': self._artifact_str(result_dict, 'verifier/reward.json'),
            'verifier_ctrf_json_path': self._artifact_str(result_dict, 'verifier/ctrf.json'),
            'verifier_test_stdout_path': self._artifact_str(result_dict, 'verifier/test-stdout.txt'),
            'verifier_run_log_path': self._artifact_str(result_dict, 'verifier/run.log'),
            'model_patch_path': self._artifact_str(result_dict, 'artifacts/model.patch'),
            'trajectory_path': self._artifact_str(result_dict, 'agent/trajectory.json'),
        }

    def _artifact_str(self, result_dict: Dict[str, Any], relative_path: str) -> Optional[str]:
        path = self._artifact_path(result_dict, relative_path)
        return str(path) if path else None

    def _artifact_path(self, result_dict: Dict[str, Any], relative_path: str) -> Optional[Path]:
        trial_uri = self._first_trial_result(result_dict).get('trial_uri') or result_dict.get('trial_uri') or ''
        if not trial_uri.startswith('file://'):
            return None
        return Path(url2pathname(trial_uri[7:])) / relative_path

    @staticmethod
    def _load_json_if_exists(path: Optional[Path]) -> Dict[str, Any]:
        if path is None or not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    @staticmethod
    def _load_text_if_exists(path: Optional[Path]) -> Optional[str]:
        if path is None or not path.is_file():
            return None
        try:
            return path.read_text(encoding='utf-8').strip()
        except Exception:
            return None

    def _load_pier_trace(self, result_dict: Dict[str, Any]) -> Tuple[Optional[AgentTrace], Optional[List[ChatMessage]]]:
        trajectory_path = self._artifact_path(result_dict, 'agent/trajectory.json')
        if trajectory_path is None or not trajectory_path.is_file():
            return None, None
        try:
            raw = json.loads(trajectory_path.read_text(encoding='utf-8'))
        except Exception:
            return None, None

        agent_info = raw.get('agent', {})
        trace = AgentTrace(
            framework=agent_info.get('name', 'mini-swe-agent'),
            environment='docker',
        )
        messages: List[ChatMessage] = []
        for index, step in enumerate(raw.get('steps', [])):
            message_id = uuid.uuid4().hex[:8]
            source = step.get('source', '')
            content = step.get('message') or step.get('content') or ''
            step_id = int(step.get('step_id') or index)
            timestamp = self._parse_timestamp(step.get('timestamp'))

            if source == 'agent':
                messages.append(ChatMessageAssistant(id=message_id, content=content))
                trace.add(
                    AgentTraceEvent(
                        step=step_id,
                        type=EventType.MODEL_GENERATE,
                        message_id=message_id,
                        timestamp=timestamp,
                    )
                )
                self._append_tool_calls(trace, messages, step, step_id, message_id, timestamp)
                continue

            if source == 'tool':
                messages.append(ChatMessageTool(id=message_id, content=content, tool_call_id=step.get('tool_call_id')))
                trace.add(
                    AgentTraceEvent(
                        step=step_id,
                        type=EventType.TOOL_RESULT,
                        message_id=message_id,
                        timestamp=timestamp,
                        payload={'source': source},
                    )
                )
                continue

            messages.append(ChatMessageUser(id=message_id, content=content))
            trace.add(
                AgentTraceEvent(
                    step=step_id,
                    type=EventType.ENV_EXEC,
                    message_id=message_id,
                    timestamp=timestamp,
                    payload={'source': source},
                )
            )

        return trace, messages

    @staticmethod
    def _parse_timestamp(value: Any) -> float:
        if not value:
            return 0
        try:
            timestamp = str(value)
            if timestamp.endswith('Z'):
                timestamp = f'{timestamp[:-1]}+00:00'
            return datetime.fromisoformat(timestamp).timestamp()
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _append_tool_calls(
        trace: AgentTrace,
        messages: List[ChatMessage],
        step: Dict[str, Any],
        step_id: int,
        message_id: str,
        timestamp: float,
    ) -> None:
        tool_calls = step.get('tool_calls') or []
        for tool_call in tool_calls:
            function = tool_call.get('function') or {}
            call_id = tool_call.get('id') or uuid.uuid4().hex[:8]
            messages[-1].tool_calls = [
                *(messages[-1].tool_calls or []),
                ToolCall(
                    id=call_id,
                    function=ToolFunction(name=function.get('name', ''), arguments=function.get('arguments') or {})
                )
            ]
            trace.add(
                AgentTraceEvent(
                    step=step_id,
                    type=EventType.TOOL_CALL,
                    message_id=message_id,
                    timestamp=timestamp,
                    payload={'tool_call_id': call_id},
                )
            )


@register_benchmark(
    BenchmarkMeta(
        name='deep_swe',
        pretty_name='DeepSWE',
        tags=[Tags.CODING, Tags.AGENT, Tags.MULTI_TURN],
        description="""
## Overview

DeepSWE is a coding-agent benchmark for evaluating repository-level software engineering tasks. EvalScope
integrates it through Pier and runs each benchmark sample as one Pier Python API job.

## Task Description

- **Task Type**: Agentic software engineering
- **Input**: DeepSWE task directory containing task metadata and verifier assets
- **Output**: A repository patch produced by a Pier built-in agent
- **Scoring**: Binary verifier reward exposed as `acc`

## Evaluation Notes

- Requires **Python>=3.12**, Docker, and `pip install evalscope[deep_swe]`
- Dataset defaults to ModelScope `evalscope/deep-swe`
- DeepSWE runs through Pier's Docker environment in EvalScope
- Use `agent_kwargs={'model_class': 'litellm'}` for OpenAI-compatible providers that do not support Responses API
""",
        dataset_id=DEFAULT_MODELSCOPE_DATASET_ID,
        eval_split='test',
        prompt_template='{question}',
        metric_list=['acc'],
        extra_params=COMMON_EXTRA_PARAMS,
    )
)
class DeepSWEBenchmarkAdapter(DeepSWEAdapter):
    pass
