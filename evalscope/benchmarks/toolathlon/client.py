import asyncio
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from evalscope.utils.function_utils import cancel_and_wait
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

CLIENT_VERSION = '1.3'
WS_CLIENT_VERSION = '1.3'
DEFAULT_SERVER_HOST = '47.253.6.47'
DEFAULT_SERVER_PORT = 8080
DEFAULT_WS_PROXY_PORT = 8081
DEFAULT_TIMEOUT_SECONDS = 240 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 5
WS_HEARTBEAT_INTERVAL_SECONDS = 30
WS_HEARTBEAT_TIMEOUT_SECONDS = 120


@dataclass
class ToolathlonServiceConfig:
    """Configuration for the official Toolathlon remote evaluation service."""

    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT
    ws_proxy_port: int = DEFAULT_WS_PROXY_PORT
    base_url: str = ''
    model_name: str = ''
    api_key: Optional[str] = None
    workers: int = 10
    provider: str = 'unified'
    task_list: Optional[List[str]] = None
    model_params: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    force_redownload: bool = False
    override_output_dir: bool = False
    skip_container_restart: bool = False
    trust_env_in_httpx: bool = False
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS
    output_dir: Path = field(default_factory=lambda: Path('outputs/toolathlon'))

    @property
    def server_url(self) -> str:
        return f'http://{self.server_host}:{self.server_port}'

    @property
    def ws_server_url(self) -> str:
        parsed = urlparse(self.server_url)
        return f'http://{parsed.hostname}:{self.ws_proxy_port}'


class ToolathlonServiceClient:
    """Minimal client for Toolathlon private-mode official service protocol."""

    def __init__(self, config: ToolathlonServiceConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)

    def run_private(self) -> Dict[str, Any]:
        check_import('httpx', extra='toolathlon', raise_error=True, feature_name='Toolathlon')
        check_import('websockets', extra='toolathlon', raise_error=True, feature_name='Toolathlon')

        self._prepare_output_dir()
        job = self._submit_job()
        job_id = job['job_id']
        client_id = job.get('client_id')
        if not client_id:
            raise RuntimeError('Toolathlon private mode did not return client_id.')

        ws_process = self._start_ws_client(job_id)
        try:
            self._poll_until_finished(job_id, ws_process)
        finally:
            self._stop_process(ws_process)

        return self.load_results()

    def load_results(self) -> Dict[str, Any]:
        stats = self._read_json_if_exists(self.output_dir / 'eval_stats.json')
        task_results = self._read_jsonl_if_exists(self.output_dir / 'eval_res_all.jsonl')
        if not task_results:
            task_results = self._read_jsonl_if_exists(self.output_dir / 'traj_log_all.jsonl')

        score = _extract_accuracy(stats, task_results)
        return {
            'job_id': self.config.job_id,
            'output_dir': str(self.output_dir),
            'eval_stats': stats,
            'task_results': task_results,
            'acc': score,
        }

    def _prepare_output_dir(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not self.config.override_output_dir:
                raise RuntimeError(
                    f'Toolathlon output directory is not empty: {self.output_dir}. '
                    'Set extra_params.override_output_dir=True to clear it.'
                )
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _submit_job(self) -> Dict[str, Any]:
        import httpx

        submit_data: Dict[str, Any] = {
            'client_version': CLIENT_VERSION,
            'ws_client_version': WS_CLIENT_VERSION,
            'mode': 'private',
            'base_url': self.config.base_url,
            'api_key': 'dummy',
            'model_name': self.config.model_name,
            'workers': self.config.workers,
            'custom_job_id': self.config.job_id,
            'skip_container_restart': self.config.skip_container_restart,
            'provider': self.config.provider,
        }
        if self.config.model_params:
            submit_data['model_params'] = self.config.model_params
        if self.config.task_list:
            submit_data['task_list_content'] = '\n'.join(self.config.task_list) + '\n'

        with httpx.Client(timeout=30.0, trust_env=self.config.trust_env_in_httpx) as client:
            response = client.post(f'{self.config.server_url}/submit_evaluation', json=submit_data)
            if response.status_code >= 400:
                raise RuntimeError(_format_submit_error(response))
            response.raise_for_status()
            data = response.json()

        job_id = data.get('job_id') or data.get('final_job_id')
        if not job_id:
            raise RuntimeError(f'Toolathlon service response missing job_id: {data}')
        self.config.job_id = job_id
        return data

    def _start_ws_client(self, job_id: str) -> subprocess.Popen:
        log_file = self.output_dir / 'ws_client.log'
        command = [
            sys.executable,
            '-m',
            'evalscope.benchmarks.toolathlon.ws_client',
            '--server-url',
            self.config.ws_server_url,
            '--llm-base-url',
            self.config.base_url,
            '--llm-api-key',
            self.config.api_key or '',
            '--job-id',
            job_id,
        ]
        with open(log_file, 'w', encoding='utf-8') as log:
            return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def _poll_until_finished(self, job_id: str, ws_process: subprocess.Popen) -> None:
        import httpx

        start_time = time.monotonic()
        downloaded_tasks = set()
        with httpx.Client(timeout=30.0, trust_env=self.config.trust_env_in_httpx) as client:
            while True:
                exit_code = ws_process.poll()
                if exit_code is not None:
                    self._cancel_job(job_id, 'WebSocket relay exited')
                    raise RuntimeError(f'Toolathlon WebSocket relay exited unexpectedly with code {exit_code}.')
                if time.monotonic() - start_time > self.config.timeout_seconds:
                    self._cancel_job(job_id, 'Timeout')
                    raise TimeoutError(f'Toolathlon job exceeded {self.config.timeout_seconds} seconds.')

                downloaded_tasks.update(self._download_completed_tasks(client, job_id, downloaded_tasks))
                status = self._get_job_status(client, job_id)
                state = str(status.get('status') or status.get('state') or '').lower()
                if state in {'completed', 'finished', 'success', 'succeeded'}:
                    self._download_completed_tasks(client, job_id, downloaded_tasks)
                    self._download_static_files(client, job_id)
                    return
                if state in {'failed', 'error', 'cancelled', 'canceled'}:
                    self._download_static_files(client, job_id)
                    raise RuntimeError(f'Toolathlon job failed: {status}')

                time.sleep(self.config.poll_interval)

    def _get_job_status(self, client: Any, job_id: str) -> Dict[str, Any]:
        for endpoint in ('poll_job_status', 'status'):
            response = client.get(f'{self.config.server_url}/{endpoint}', params={'job_id': job_id})
            if response.status_code == 404:
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError('Toolathlon service does not expose a supported job status endpoint.')

    def _download_completed_tasks(self, client: Any, job_id: str, downloaded_tasks: set) -> set:
        response = client.get(f'{self.config.server_url}/get_completed_tasks', params={'job_id': job_id})
        if response.status_code == 404:
            return set()
        response.raise_for_status()
        data = response.json()
        task_names = data.get('completed_tasks') or data.get('tasks') or []
        new_tasks = {name for name in task_names if name not in downloaded_tasks}
        for task_name in new_tasks:
            self._download_task_archive(client, job_id, task_name)
        return new_tasks

    def _download_task_archive(self, client: Any, job_id: str, task_name: str) -> None:
        response = client.get(
            f'{self.config.server_url}/get_task_archive',
            params={
                'job_id': job_id,
                'task_name': task_name
            },
            timeout=120.0,
        )
        response.raise_for_status()
        expected_md5 = response.headers.get('X-Content-MD5')
        archive_bytes = response.content
        actual_md5 = md5(archive_bytes).hexdigest()
        if expected_md5 and expected_md5 != actual_md5:
            raise RuntimeError(f'Toolathlon archive MD5 mismatch for {task_name}: {actual_md5} != {expected_md5}')

        finalpool_dir = self.output_dir / 'finalpool'
        finalpool_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=_BytesIO(archive_bytes), mode='r:gz') as archive:
            _extract_tar_safely(archive, finalpool_dir)

    def _download_static_files(self, client: Any, job_id: str) -> None:
        response = client.get(f'{self.config.server_url}/get_static_files', params={'job_id': job_id}, timeout=60.0)
        if response.status_code == 404:
            return
        response.raise_for_status()
        for filename, content in response.json().items():
            if content is None:
                continue
            target = _resolve_output_path(self.output_dir, filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')

    def _cancel_job(self, job_id: str, reason: str) -> None:
        try:
            import httpx

            with httpx.Client(timeout=10.0, trust_env=self.config.trust_env_in_httpx) as client:
                client.post(f'{self.config.server_url}/cancel_job', params={'job_id': job_id})
        except Exception as exc:
            logger.warning(f'Failed to cancel Toolathlon job after {reason}: {exc}')

    def _stop_process(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def _read_json_if_exists(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding='utf-8'))

    def _read_jsonl_if_exists(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


def _extract_accuracy(stats: Dict[str, Any], task_results: List[Dict[str, Any]]) -> float:
    for key in ('acc', 'accuracy', 'pass_rate', 'success_rate', 'score'):
        value = stats.get(key)
        if isinstance(value, (int, float, bool)):
            return float(value)

    total = stats.get('total') or stats.get('num_tasks')
    passed = stats.get('passed') or stats.get('num_passed') or stats.get('success')
    if isinstance(total, int) and total > 0 and isinstance(passed, int):
        return passed / total

    pass_values = []
    for row in task_results:
        value = row.get('pass')
        if isinstance(value, bool):
            pass_values.append(float(value))
        elif isinstance(value, (int, float)):
            pass_values.append(float(value))
    if pass_values:
        return sum(pass_values) / len(pass_values)
    return 0.0


def _format_submit_error(response: Any) -> str:
    detail = _response_detail(response)
    status_code = response.status_code
    prefix = f'Toolathlon official service rejected evaluation submission with HTTP {status_code}.'
    if status_code == 503:
        return (
            f'{prefix} The public official service is currently busy and can run only one evaluation job at a time. '
            f'Retry later, request a dedicated official service, or self-deploy Toolathlon. Response: {detail}'
        )
    if status_code == 429:
        return (
            f'{prefix} The official service rate limit was reached. The public service allows 180 minutes of '
            f'cumulative execution time per IP per 24 hours; after that threshold, requests are capped at 3 per IP '
            f'per 24 hours. Response: {detail}'
        )
    return f'{prefix} Response: {detail}'


def _response_detail(response: Any) -> str:
    try:
        data = response.json()
    except Exception:
        data = getattr(response, 'text', '')
    return str(data)[:1000] if data else '<empty>'


def _BytesIO(content: bytes) -> Any:
    from io import BytesIO

    return BytesIO(content)


def _extract_tar_safely(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    members = archive.getmembers()
    for member in members:
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise RuntimeError(f'Unsafe Toolathlon archive member type: {member.name}')
        _resolve_output_path(destination, member.name)
    archive.extractall(destination, members=members)


def _resolve_output_path(base_dir: Path, relative_path: str) -> Path:
    base = base_dir.resolve()
    target = (base / relative_path).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f'Unsafe Toolathlon output path: {relative_path}') from exc
    return target


async def run_ws_proxy(server_url: str, llm_base_url: str, llm_api_key: str, job_id: str) -> None:
    import httpx
    from websockets import connect

    ws_url = server_url.replace('http://', 'ws://').replace('https://', 'wss://') + f'/ws?job_id={job_id}'
    async with connect(ws_url, ping_interval=20, ping_timeout=120, max_size=32 * 1024 * 1024) as websocket:
        request_queue: asyncio.Queue = asyncio.Queue()
        last_heartbeat_ack = {'time': time.monotonic()}
        active_request_tasks: set[asyncio.Task] = set()

        async def receive_messages() -> None:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type')
                if msg_type == 'heartbeat_ack':
                    last_heartbeat_ack['time'] = time.monotonic()
                elif msg_type == 'error':
                    raise RuntimeError(data.get('message'))
                elif msg_type == 'new_requests':
                    for request in data.get('requests', []):
                        await request_queue.put(request)

        async def process_request(request: Dict[str, Any], client: Any) -> None:
            request_id = request['request_id']
            endpoint = request.get('_endpoint', '/chat/completions')
            payload = {
                key: value
                for key, value in request.items()
                if key not in {'request_id', 'pushed', '_server_push_time'} and not key.startswith('_')
            }
            headers = {'Authorization': f'Bearer {llm_api_key}'} if llm_api_key else {}
            try:
                try:
                    response = await client.post(f'{llm_base_url}{endpoint}', json=payload, headers=headers)
                    response_data = {'status_code': response.status_code, 'body': response.json()}
                except Exception as exc:
                    response_data = {
                        'status_code': 500,
                        'body': {
                            'error': {
                                'message': f'{type(exc).__name__}: {exc}',
                                'type': 'network_error',
                                'code': 'client_error',
                            }
                        },
                    }
                await websocket.send(json.dumps({'type': 'response', 'request_id': request_id, 'data': response_data}))
            finally:
                request_queue.task_done()

        async def process_requests(client: Any) -> None:
            queue_get_task = asyncio.create_task(request_queue.get())
            try:
                while True:
                    done, _ = await asyncio.wait({queue_get_task, *active_request_tasks},
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if queue_get_task in done:
                        request = queue_get_task.result()
                        request_task = asyncio.create_task(process_request(request, client))
                        active_request_tasks.add(request_task)
                        queue_get_task = asyncio.create_task(request_queue.get())

                    completed_requests = done.intersection(active_request_tasks)
                    active_request_tasks.difference_update(completed_requests)
                    if completed_requests:
                        await asyncio.gather(*completed_requests)
            finally:
                if queue_get_task.done() and not queue_get_task.cancelled():
                    try:
                        queue_get_task.result()
                    except Exception:
                        pass
                    else:
                        request_queue.task_done()
                else:
                    await cancel_and_wait(queue_get_task)
                await asyncio.gather(
                    *(cancel_and_wait(task) for task in active_request_tasks),
                    return_exceptions=True,
                )
                active_request_tasks.clear()

        async def send_heartbeat() -> None:
            while True:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL_SECONDS)
                await websocket.send(json.dumps({'type': 'heartbeat'}))
                if time.monotonic() - last_heartbeat_ack['time'] > WS_HEARTBEAT_TIMEOUT_SECONDS:
                    raise TimeoutError('Toolathlon WebSocket heartbeat timed out.')

        async with httpx.AsyncClient(timeout=600.0) as client:
            tasks = {
                asyncio.create_task(receive_messages()),
                asyncio.create_task(process_requests(client)),
                asyncio.create_task(send_heartbeat()),
            }
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                await asyncio.gather(*(cancel_and_wait(task) for task in tasks), return_exceptions=True)
