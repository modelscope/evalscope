import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import url2pathname

from evalscope.api.agent import AgentTrace, AgentTraceEvent, EventType
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.constants import HubType


def download_snapshot(
    data_id_or_path: str,
    data_source: str,
    cache_dir: str,
    force_redownload: bool = False,
) -> Path:
    """Download or resolve a DeepSWE dataset snapshot root."""
    data_source = data_source or HubType.MODELSCOPE
    if data_source == HubType.LOCAL or os.path.exists(data_id_or_path):
        return Path(data_id_or_path).resolve()

    if data_source == HubType.HUGGINGFACE:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=data_id_or_path,
                repo_type='dataset',
                cache_dir=cache_dir,
                force_download=force_redownload,
            )
        )

    if data_source == HubType.MODELSCOPE:
        from modelscope import dataset_snapshot_download

        return Path(dataset_snapshot_download(data_id_or_path, cache_dir=cache_dir))

    raise ValueError(f'Unsupported dataset hub for DeepSWE: {data_source}')


def load_task_records(snapshot_path: Path) -> List[Dict[str, Any]]:
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
        record['instruction'] = load_instruction(task_path, record)
        records.append(record)

    return records


def load_instruction(task_path: Path, record: Dict[str, Any]) -> str:
    instruction_path = task_path / 'instruction.md'
    if instruction_path.is_file():
        return instruction_path.read_text(encoding='utf-8')
    return str(record.get('instruction') or record.get('display_description') or record.get('description') or '')


def filter_task_records(
    records: List[Dict[str, Any]],
    task_ids: List[str],
    languages: List[str],
    categories: List[str],
) -> List[Dict[str, Any]]:
    task_id_set = set(task_ids)
    language_set = {item.lower() for item in languages}
    category_set = {item.lower() for item in categories}

    filtered = []
    for record in records:
        if task_id_set and record['task_id'] not in task_id_set:
            continue
        if language_set and str(record.get('language', '')).lower() not in language_set:
            continue
        if category_set and str(record.get('category', '')).lower() not in category_set:
            continue
        filtered.append(record)
    return filtered


def first_trial_result(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    trial_results = result_dict.get('trial_results') or []
    if not trial_results:
        return {}
    return trial_results[0] or {}


def is_agent_execution_failed(trial_result: Dict[str, Any]) -> bool:
    exception_info = trial_result.get('exception_info') or {}
    exception_type = exception_info.get('exception_type') or exception_info.get('type')
    return exception_type == 'NonZeroAgentExitCodeError'


def build_score_metadata(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    trial_result = first_trial_result(result_dict)
    verifier_result = trial_result.get('verifier_result') or {}
    rewards = dict(verifier_result.get('rewards') or {})
    reward_json = load_json_if_exists(artifact_path(result_dict, 'verifier/reward.json'))
    if reward_json:
        rewards.update(reward_json)

    metadata = {
        'reward': rewards.get('reward'),
        'partial': rewards.get('partial'),
        'f2p': rewards.get('f2p'),
        'p2p': rewards.get('p2p'),
        'apply_failed': rewards.get('apply_failed'),
        'reward_txt': load_text_if_exists(artifact_path(result_dict, 'verifier/reward.txt')),
        'reward_json': reward_json,
        'trial_result': trial_result,
        'agent_execution_failed': is_agent_execution_failed(trial_result),
    }
    metadata.update(artifact_metadata(result_dict))
    return metadata


def artifact_metadata(result_dict: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {
        'pier_job_result_path': result_dict.get('job_result_path'),
        'trial_uri': first_trial_result(result_dict).get('trial_uri') or result_dict.get('trial_uri'),
        'verifier_reward_json_path': artifact_str(result_dict, 'verifier/reward.json'),
        'verifier_ctrf_json_path': artifact_str(result_dict, 'verifier/ctrf.json'),
        'verifier_test_stdout_path': artifact_str(result_dict, 'verifier/test-stdout.txt'),
        'verifier_run_log_path': artifact_str(result_dict, 'verifier/run.log'),
        'model_patch_path': artifact_str(result_dict, 'artifacts/model.patch'),
        'trajectory_path': artifact_str(result_dict, 'agent/trajectory.json'),
    }


def artifact_str(result_dict: Dict[str, Any], relative_path: str) -> Optional[str]:
    path = artifact_path(result_dict, relative_path)
    return str(path) if path else None


def artifact_path(result_dict: Dict[str, Any], relative_path: str) -> Optional[Path]:
    trial_uri = first_trial_result(result_dict).get('trial_uri') or result_dict.get('trial_uri') or ''
    if not trial_uri.startswith('file://'):
        return None
    return Path(url2pathname(trial_uri[7:])) / relative_path


def load_json_if_exists(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def load_text_if_exists(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.is_file():
        return None
    try:
        return path.read_text(encoding='utf-8').strip()
    except Exception:
        return None


def load_pier_trace(result_dict: Dict[str, Any]) -> Tuple[Optional[AgentTrace], Optional[List[ChatMessage]]]:
    trajectory_path = artifact_path(result_dict, 'agent/trajectory.json')
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
        timestamp = parse_timestamp(step.get('timestamp'))

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
            append_tool_calls(trace, messages, step, step_id, message_id, timestamp)
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


def parse_timestamp(value: Any) -> float:
    if not value:
        return 0
    try:
        timestamp = str(value)
        if timestamp.endswith('Z'):
            timestamp = f'{timestamp[:-1]}+00:00'
        return datetime.fromisoformat(timestamp).timestamp()
    except (TypeError, ValueError):
        return 0


def append_tool_calls(
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
            ToolCall(id=call_id, function=ToolFunction(name=function.get('name', ''), arguments=function.get('arguments') or {}))
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
