import json
import os
import shutil
import tarfile
import urllib.request
import uuid
import yaml
import zipfile
from dataclasses import dataclass
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evalscope.api.agent import AgentTrace, AgentTraceEvent, EventType
from evalscope.api.dataset import DatasetHub
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from evalscope.api.model import ModelUsage
from evalscope.api.sandbox import build_docker_image, should_build_docker_image
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.constants import HubType
from evalscope.utils.logger import get_logger

DEFAULT_CLAW_EVAL_DATASET_ID = 'claw-eval/Claw-Eval'
DEFAULT_CLAW_EVAL_COMMIT = 'd3f02d4938ab0832377d90535013def2b1a2fdc0'
DEFAULT_CLAW_EVAL_PACKAGE = (
    f'claw-eval[sandbox,mock,web] @ git+https://github.com/claw-eval/claw-eval.git@{DEFAULT_CLAW_EVAL_COMMIT}'
)
DEFAULT_CLAW_EVAL_REPO_ARCHIVE = f'https://github.com/claw-eval/claw-eval/archive/{DEFAULT_CLAW_EVAL_COMMIT}.zip'
DEFAULT_CLAW_EVAL_SANDBOX_IMAGE = 'claw-eval-agent:latest'
FIXTURES_FILE_PATH = 'data/fixtures.tar.gz'

logger = get_logger()


@dataclass(frozen=True)
class ClawEvalAssets:
    repo_root: Path
    tasks_dir: Path
    fixtures_archive: Path
    fixtures_dir: Path


def load_task_manifest(
    dataset_id: str,
    data_source: str,
    splits: Iterable[str],
    force_redownload: bool = False,
) -> List[Dict[str, Any]]:
    """Load Claw-Eval task rows through EvalScope's shared dataset hub."""
    hub = DatasetHub(
        data_id_or_path=dataset_id,
        data_source=data_source,
        force_redownload=force_redownload,
    )
    records: List[Dict[str, Any]] = []
    for split in splits:
        dataset = hub.load(split=split)
        for row in dataset:
            record = dict(row)
            record['split'] = split
            records.append(record)
    return records


def prepare_claw_eval_assets(
    dataset_id: str = DEFAULT_CLAW_EVAL_DATASET_ID,
    data_source: str = HubType.MODELSCOPE,
    cache_dir: Optional[str] = None,
    force_redownload: bool = False,
    official_repo_path: Optional[str] = None,
) -> ClawEvalAssets:
    """Prepare official Claw-Eval task code and ModelScope fixtures."""
    root = Path(cache_dir).expanduser() if cache_dir else Path.home() / '.cache' / 'evalscope' / 'claw_eval'
    root.mkdir(parents=True, exist_ok=True)
    repo_root = _prepare_official_repo(
        cache_root=root / 'official_repo',
        force_redownload=force_redownload,
        official_repo_path=official_repo_path,
    )

    fixtures_archive = Path(
        DatasetHub(
            data_id_or_path=dataset_id,
            data_source=data_source,
            force_redownload=force_redownload,
            cache_dir=str(root / 'modelscope'),
        ).download_file(FIXTURES_FILE_PATH)
    )
    fixtures_dir = _extract_fixtures(fixtures_archive, root / 'fixtures', force_redownload=force_redownload)
    _link_fixtures_into_tasks(fixtures_dir, repo_root / 'tasks')

    tasks_dir = repo_root / 'tasks'
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f'Claw-Eval official repo does not contain tasks/: {repo_root}')
    return ClawEvalAssets(
        repo_root=repo_root,
        tasks_dir=tasks_dir,
        fixtures_archive=fixtures_archive,
        fixtures_dir=fixtures_dir,
    )


def ensure_claw_eval_sandbox_image(repo_root: Path) -> str:
    """Build the official Claw-Eval sandbox image locally when it is missing."""
    dockerfile = repo_root / 'Dockerfile.agent'
    if not dockerfile.is_file():
        raise FileNotFoundError(f'Claw-Eval official repo does not contain Dockerfile.agent: {repo_root}')
    if should_build_docker_image(DEFAULT_CLAW_EVAL_SANDBOX_IMAGE):
        logger.info(
            f'Claw-Eval sandbox image {DEFAULT_CLAW_EVAL_SANDBOX_IMAGE!r} not found. '
            f'Building from official Dockerfile: {dockerfile}'
        )
        build_docker_image(
            image=DEFAULT_CLAW_EVAL_SANDBOX_IMAGE,
            path=str(repo_root),
            dockerfile='Dockerfile.agent',
        )
    return DEFAULT_CLAW_EVAL_SANDBOX_IMAGE


def materialize_task_root(source_tasks_dir: Path, selected_task_ids: List[str], output_root: Path) -> Path:
    """Create a runtime tasks/ root that runs selected tasks while keeping fixture-only cross refs."""
    tasks_root = output_root / 'tasks'
    if tasks_root.exists():
        shutil.rmtree(tasks_root)
    tasks_root.mkdir(parents=True)

    selected = set(selected_task_ids)
    for task_dir in source_tasks_dir.iterdir():
        if not task_dir.is_dir():
            continue
        dest = tasks_root / task_dir.name
        if task_dir.name in selected:
            _symlink_or_copy(task_dir, dest)
            continue
        fixtures = task_dir / 'fixtures'
        if fixtures.exists():
            (dest).mkdir(parents=True, exist_ok=True)
            _symlink_or_copy(fixtures, dest / 'fixtures')

    missing = [task_id for task_id in selected_task_ids if not (tasks_root / task_id / 'task.yaml').is_file()]
    if missing:
        raise FileNotFoundError(f'Claw-Eval task ids not found in official tasks/: {missing[:10]}')

    mock_services = source_tasks_dir.parent / 'mock_services'
    if mock_services.is_dir():
        runtime_mock_services = output_root / 'mock_services'
        if runtime_mock_services.is_symlink() or runtime_mock_services.is_file():
            runtime_mock_services.unlink()
        elif runtime_mock_services.exists():
            shutil.rmtree(runtime_mock_services)
        _symlink_or_copy(mock_services, runtime_mock_services)
    return tasks_root


def write_claw_eval_config(path: Path, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')


def run_claw_eval_task(
    task_dir: Path,
    config_path: Path,
    trace_root: Path,
    repo_root: Path,
    model_id: str,
    api_key: Optional[str],
    base_url: Optional[str],
    port_offset: int = 0,
    judge_model: Optional[str] = None,
    no_judge: bool = False,
    proxy: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one Claw-Eval task through the pinned official private API."""
    validate_claw_eval_private_api()
    task_dir = task_dir.resolve()
    trace_root = trace_root.resolve()
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Claw-Eval config file was not found: {config_path}')
    if not (task_dir / 'task.yaml').is_file():
        raise FileNotFoundError(f'Claw-Eval task.yaml was not found: {task_dir}')

    from claw_eval.cli import _run_single_task

    result = _run_single_task(
        task_dir=str(task_dir),
        config_path=str(config_path),
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        trace_dir=str(trace_root),
        port_offset=port_offset,
        no_judge=no_judge,
        judge_model=judge_model,
        trials=1,
        proxy=proxy,
        sandbox=True,
        sandbox_image=None,
        sandbox_tools=False,
    )
    return parse_single_task_result(result, trace_root=trace_root, task_dir=task_dir, repo_root=repo_root)


def validate_claw_eval_private_api() -> None:
    """Fail fast when the installed Claw-Eval package is not the pinned private API shape."""
    try:
        from claw_eval.cli import _run_single_task
    except ImportError as exc:
        raise RuntimeError(
            'Claw-Eval official runner is not installed. Install the pinned package with: '
            f'pip install "{DEFAULT_CLAW_EVAL_PACKAGE}"'
        ) from exc

    expected_run_params = {
        'task_dir',
        'config_path',
        'model',
        'api_key',
        'base_url',
        'trace_dir',
        'port_offset',
        'no_judge',
        'judge_model',
        'trials',
        'proxy',
        'sandbox',
        'sandbox_image',
        'sandbox_tools',
    }
    actual_run_params = set(signature(_run_single_task).parameters)
    if not expected_run_params.issubset(actual_run_params):
        raise RuntimeError(
            'Installed Claw-Eval private API is incompatible with EvalScope claw_eval. '
            f'Expected _run_single_task parameters {sorted(expected_run_params)}, got {sorted(actual_run_params)}. '
            f'Install the pinned package with: pip install "{DEFAULT_CLAW_EVAL_PACKAGE}"'
        )


def parse_single_task_result(
    result: Dict[str, Any],
    trace_root: Path,
    task_dir: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    """Normalize official _run_single_task output into EvalScope metadata."""
    trials = list(result.get('trials') or [])
    first_trial = trials[0] if trials else {}
    task_score = float(first_trial.get('task_score') or result.get('avg_score') or 0.0)
    errored = bool(result.get('error') or first_trial.get('error'))
    passed = bool(first_trial.get('passed') or result.get('avg_passed') or False)
    trace_path = first_trial.get('trace')
    metrics = {
        'avg_score': task_score,
        'task_score': task_score,
        'passed': 1.0 if passed else 0.0,
        'error_rate': 1.0 if errored else 0.0,
    }
    for key in (
        'model_input_tokens',
        'model_output_tokens',
        'input_tokens',
        'output_tokens',
        'tokens',
        'model_time_s',
        'tool_time_s',
        'other_time_s',
        'wall_time_s',
        'completion',
        'robustness',
        'communication',
        'safety',
    ):
        if key in first_trial:
            metrics[key] = float(first_trial[key])

    return {
        'task_id': result.get('task_id') or task_dir.name,
        'task_name': result.get('task_name') or '',
        'difficulty': result.get('difficulty') or '',
        'trace_root': str(trace_root),
        'trace_path': str(trace_path) if trace_path else None,
        'task_dir': str(task_dir),
        'repo_root': str(repo_root),
        'raw_result': result,
        'trials': trials,
        'metrics': metrics,
        'error': result.get('error') or first_trial.get('error'),
    }


def load_claw_eval_trace(trace_path: Optional[str]) -> Tuple[Optional[AgentTrace], Optional[List[ChatMessage]]]:
    """Convert an official Claw-Eval trace JSONL file into EvalScope agent visualization data."""
    if not trace_path:
        return None, None
    path = Path(trace_path)
    if not path.is_file():
        return None, None

    trace = AgentTrace(framework='claw-eval', environment='docker')
    messages: List[ChatMessage] = []
    dispatches: Dict[str, Dict[str, Any]] = {}
    step = 0
    last_timestamp: Optional[float] = None

    try:
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None, None

    for row in rows:
        event_type = row.get('type')
        timestamp = _parse_timestamp(row.get('timestamp'))
        if event_type == 'trace_start':
            trace.trial_id = row.get('trace_id')
            continue

        if event_type == 'tool_dispatch':
            dispatches[str(row.get('tool_use_id') or '')] = row
            continue

        if event_type == 'message':
            message = row.get('message') or {}
            role = message.get('role')
            content = message.get('content') or []
            usage = row.get('usage') or {}

            if role == 'assistant':
                msg_id = uuid.uuid4().hex[:8]
                text, tool_calls = _parse_assistant_content(content)
                messages.append(
                    ChatMessageAssistant(
                        id=msg_id, content=text, tool_calls=tool_calls or None, model=row.get('model')
                    )
                )
                trace.add(
                    AgentTraceEvent(
                        step=step,
                        type=EventType.MODEL_GENERATE,
                        message_id=msg_id,
                        timestamp=timestamp,
                        latency_ms=_latency_ms(last_timestamp, timestamp),
                        token_usage=_token_usage_dict(usage),
                    )
                )
                for tool_call in tool_calls:
                    trace.add(
                        AgentTraceEvent(
                            step=step,
                            type=EventType.TOOL_CALL,
                            message_id=msg_id,
                            timestamp=timestamp,
                            payload={
                                'tool_call_id': tool_call.id,
                                'name': tool_call.function.name,
                                'arguments': tool_call.function.arguments,
                            },
                        )
                    )
                step += 1
            elif role == 'user':
                for part in content if isinstance(content, list) else [{'type': 'text', 'text': str(content)}]:
                    if not isinstance(part, dict):
                        continue
                    if part.get('type') == 'tool_result':
                        _append_tool_result(trace, messages, part, dispatches, step, timestamp)
                    else:
                        msg_id = uuid.uuid4().hex[:8]
                        messages.append(ChatMessageUser(id=msg_id, content=_content_part_to_text(part)))

            last_timestamp = timestamp or last_timestamp
            continue

        if event_type == 'trace_end':
            trace.total_usage = ModelUsage(
                input_tokens=int(row.get('input_tokens') or row.get('model_input_tokens') or 0),
                output_tokens=int(row.get('output_tokens') or row.get('model_output_tokens') or 0),
                total_tokens=int(row.get('total_tokens') or 0),
            )
            trace.add(
                AgentTraceEvent(
                    step=step,
                    type=EventType.RUN_END,
                    timestamp=timestamp,
                    token_usage={
                        'input': trace.total_usage.input_tokens,
                        'output': trace.total_usage.output_tokens,
                        'total': trace.total_usage.total_tokens,
                    },
                    payload={
                        'total_turns': row.get('total_turns'),
                        'wall_time_s': row.get('wall_time_s'),
                        'task_score': row.get('task_score'),
                        'passed': row.get('passed'),
                    },
                )
            )
            continue

        if event_type == 'grading_result':
            trace.add(
                AgentTraceEvent(
                    step=step,
                    type=EventType.SUBMIT,
                    timestamp=timestamp,
                    payload={
                        'task_score': row.get('task_score'),
                        'passed': row.get('passed'),
                        'scores': row.get('scores') or {},
                    },
                )
            )

    return trace, messages or None


def _prepare_official_repo(
    cache_root: Path,
    force_redownload: bool,
    official_repo_path: Optional[str],
) -> Path:
    if official_repo_path:
        repo_root = Path(official_repo_path).expanduser().resolve()
        if not (repo_root / 'tasks').is_dir():
            raise FileNotFoundError(f'Claw-Eval official_repo_path must contain tasks/: {repo_root}')
        return repo_root

    extract_root = cache_root / 'extracted'
    if force_redownload and extract_root.exists():
        shutil.rmtree(extract_root)
    existing = _find_repo_root(extract_root)
    if existing is not None:
        return existing

    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / 'claw_eval_official.zip'
    if force_redownload or not archive_path.is_file():
        urllib.request.urlretrieve(DEFAULT_CLAW_EVAL_REPO_ARCHIVE, archive_path)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_root)

    repo_root = _find_repo_root(extract_root)
    if repo_root is None:
        raise FileNotFoundError(f'Could not find tasks/ after extracting {archive_path}')
    return repo_root


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


def _latency_ms(previous: Optional[float], current: float) -> Optional[float]:
    if not previous or not current:
        return None
    return round((current - previous) * 1000, 2)


def _token_usage_dict(usage: Dict[str, Any]) -> Optional[Dict[str, int]]:
    if not usage:
        return None
    input_tokens = int(usage.get('input_tokens') or 0)
    output_tokens = int(usage.get('output_tokens') or 0)
    return {
        'input': input_tokens,
        'output': output_tokens,
        'total': input_tokens + output_tokens,
    }


def _parse_assistant_content(content: Any) -> Tuple[str, List[ToolCall]]:
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    parts = content if isinstance(content, list) else [{'type': 'text', 'text': str(content or '')}]
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get('type')
        if part_type == 'tool_use':
            tool_calls.append(
                ToolCall(
                    id=str(part.get('id') or uuid.uuid4().hex[:8]),
                    function=ToolFunction(name=str(part.get('name') or ''), arguments=part.get('input') or {}),
                )
            )
        elif part_type == 'text':
            text_parts.append(str(part.get('text') or ''))
    return ''.join(text_parts), tool_calls


def _append_tool_result(
    trace: AgentTrace,
    messages: List[ChatMessage],
    part: Dict[str, Any],
    dispatches: Dict[str, Dict[str, Any]],
    step: int,
    timestamp: float,
) -> None:
    tool_call_id = str(part.get('tool_use_id') or '')
    dispatch = dispatches.get(tool_call_id) or {}
    msg_id = uuid.uuid4().hex[:8]
    content = _content_part_to_text(part)
    is_error = bool(part.get('is_error') or dispatch.get('response_status', 200) >= 400)
    messages.append(
        ChatMessageTool(
            id=msg_id,
            content=content,
            tool_call_id=tool_call_id,
            function=dispatch.get('tool_name'),
        )
    )
    trace.add(
        AgentTraceEvent(
            step=step,
            type=EventType.TOOL_RESULT if not is_error else EventType.ERROR,
            message_id=msg_id,
            timestamp=timestamp,
            latency_ms=dispatch.get('latency_ms'),
            payload={
                'tool_call_id': tool_call_id,
                'name': dispatch.get('tool_name'),
                'arguments': dispatch.get('request_body') or {},
                'status': dispatch.get('response_status'),
                'endpoint_url': dispatch.get('endpoint_url'),
            },
        )
    )


def _content_part_to_text(part: Dict[str, Any]) -> str:
    content = part.get('content')
    if isinstance(content, list):
        return ''.join(_content_part_to_text(item) for item in content if isinstance(item, dict))
    if content is not None:
        return str(content)
    return str(part.get('text') or '')


def _extract_fixtures(archive_path: Path, fixtures_dir: Path, force_redownload: bool) -> Path:
    marker_path = fixtures_dir / '.extract_complete.json'
    archive_sig = {
        'archive': str(archive_path),
        'size': archive_path.stat().st_size,
        'mtime_ns': archive_path.stat().st_mtime_ns,
    }
    if marker_path.is_file() and not force_redownload:
        try:
            if json.loads(marker_path.read_text(encoding='utf-8')) == archive_sig:
                return fixtures_dir
        except json.JSONDecodeError:
            pass

    if fixtures_dir.exists():
        shutil.rmtree(fixtures_dir)
    fixtures_dir.mkdir(parents=True)
    with tarfile.open(archive_path) as tf:
        _safe_extract_tar(tf, fixtures_dir)
    marker_path.write_text(json.dumps(archive_sig, indent=2), encoding='utf-8')
    return fixtures_dir


def _link_fixtures_into_tasks(fixtures_dir: Path, tasks_dir: Path) -> None:
    for fixture_task_dir in fixtures_dir.iterdir():
        if not fixture_task_dir.is_dir():
            continue
        source_fixtures = fixture_task_dir / 'fixtures'
        target_fixtures = tasks_dir / fixture_task_dir.name / 'fixtures'
        if not source_fixtures.is_dir() or not target_fixtures.parent.is_dir():
            continue
        for source_file in source_fixtures.rglob('*'):
            if not source_file.is_file():
                continue
            rel_path = source_file.relative_to(source_fixtures)
            target_file = target_fixtures / rel_path
            if target_file.exists():
                continue
            target_file.parent.mkdir(parents=True, exist_ok=True)
            _symlink_or_copy(source_file, target_file)


def _safe_extract_tar(tf: tarfile.TarFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in tf.getmembers():
        member_path = (target_dir / member.name).resolve()
        if os.path.commonpath([target_root, member_path]) != str(target_root):
            raise ValueError(f'Unsafe tar member path: {member.name}')
    tf.extractall(target_dir)


def _find_repo_root(extract_root: Path) -> Optional[Path]:
    if not extract_root.exists():
        return None
    if (extract_root / 'tasks').is_dir():
        return extract_root
    for child in extract_root.iterdir():
        if child.is_dir() and (child / 'tasks').is_dir():
            return child
    return None


def _symlink_or_copy(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        os.symlink(source, dest, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, dest, symlinks=True)
        else:
            shutil.copy2(source, dest)
