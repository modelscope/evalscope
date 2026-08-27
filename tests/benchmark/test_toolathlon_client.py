import asyncio
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

if importlib.util.find_spec('httpx') is None or importlib.util.find_spec('websockets') is None:
    raise unittest.SkipTest('Toolathlon client tests require `evalscope[toolathlon]`.')

from evalscope.benchmarks.toolathlon import client as toolathlon_client
from evalscope.benchmarks.toolathlon import ws_client as toolathlon_ws_client
from evalscope.benchmarks.toolathlon.client import (
    ToolathlonServiceClient,
    ToolathlonServiceConfig,
    _extract_accuracy,
    run_ws_proxy,
)


class _HttpxContext:

    def __enter__(self) -> '_HttpxContext':
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False


class _RelayProcess:

    def __init__(self, exit_code: Optional[int]) -> None:
        self.exit_code = exit_code

    def poll(self) -> Optional[int]:
        return self.exit_code


class TestToolathlonClient(unittest.TestCase):

    def test_extract_accuracy_from_stats(self) -> None:
        self.assertEqual(_extract_accuracy({'passed': 3, 'total': 4}, []), 0.75)
        self.assertEqual(_extract_accuracy({'passed': 0, 'total': 4}, []), 0.0)
        self.assertEqual(_extract_accuracy({'pass_rate': 0.5}, []), 0.5)
        self.assertEqual(_extract_accuracy({}, [{'pass': True}, {'pass': False}]), 0.5)
        self.assertEqual(_extract_accuracy({}, [{'pass': False}, {'pass': False}]), 0.0)

    def test_extract_accuracy_raises_when_results_missing(self) -> None:
        for stats, task_results in [
            ({}, []),
            ({'unexpected': 0.5}, []),
            ({'acc': None}, []),
        ]:
            with self.subTest(stats=stats, task_results=task_results):
                with self.assertRaisesRegex(RuntimeError, 'score unavailable'):
                    _extract_accuracy(stats, task_results)

    def test_extract_accuracy_rejects_invalid_values(self) -> None:
        invalid_results = [
            ({'acc': math.nan}, []),
            ({'acc': math.inf}, []),
            ({'acc': -0.1}, []),
            ({'acc': 1.1}, []),
            ({'acc': True}, []),
            ({'passed': 0}, []),
            ({'total': 4}, []),
            ({'passed': True, 'total': 4}, []),
            ({'passed': 0, 'total': 0}, []),
            ({'passed': -1, 'total': 4}, []),
            ({'passed': 5, 'total': 4}, []),
            ({}, [{'task': 'missing-pass'}]),
            ({}, [{'pass': math.nan}]),
            ({}, [{'pass': math.inf}]),
            ({}, [{'pass': -0.1}]),
            ({}, [{'pass': 1.1}]),
        ]
        for stats, task_results in invalid_results:
            with self.subTest(stats=stats, task_results=task_results):
                with self.assertRaisesRegex(RuntimeError, 'score unavailable'):
                    _extract_accuracy(stats, task_results)

    def test_submit_private_job_does_not_send_local_api_key(self):
        captured = {}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {'job_id': 'job-1', 'client_id': 'client-1'}

        class FakeHttpxClient:

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def post(self, url, json):
                captured['url'] = url
                captured['json'] = json
                return FakeResponse()

        config = ToolathlonServiceConfig(
            server_host='toolathlon.example',
            server_port=8080,
            base_url='http://localhost:8000/v1',
            model_name='local-model',
            api_key='secret-local-key',
            task_list=['find-alita-paper', 'git-milestone'],
            model_params={'temperature': 0.0},
            output_dir=Path('/tmp/toolathlon-test'),
        )

        with patch('httpx.Client', FakeHttpxClient):
            result = ToolathlonServiceClient(config)._submit_job()

        self.assertEqual(result['job_id'], 'job-1')
        self.assertEqual(captured['url'], 'http://toolathlon.example:8080/submit_evaluation')
        self.assertEqual(captured['json']['mode'], 'private')
        self.assertEqual(captured['json']['api_key'], 'dummy')
        self.assertNotIn('secret-local-key', str(captured['json']))
        self.assertEqual(captured['json']['task_list_content'], 'find-alita-paper\ngit-milestone\n')
        self.assertEqual(captured['json']['model_params'], {'temperature': 0.0})

    def test_submit_private_job_reports_busy_service(self):

        class FakeResponse:
            status_code = 503
            text = 'Service Unavailable'

            def json(self):
                return {'detail': 'Server is busy'}

        class FakeHttpxClient:

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def post(self, url, json):
                return FakeResponse()

        config = ToolathlonServiceConfig(
            server_host='toolathlon.example',
            server_port=8080,
            base_url='http://localhost:8000/v1',
            model_name='local-model',
        )

        with patch('httpx.Client', FakeHttpxClient):
            with self.assertRaisesRegex(RuntimeError, 'currently busy.*one evaluation job'):
                ToolathlonServiceClient(config)._submit_job()

    def test_submit_private_job_reports_rate_limit(self):

        class FakeResponse:
            status_code = 429
            text = 'Too Many Requests'

            def json(self):
                return {'detail': 'Rate limit exceeded'}

        class FakeHttpxClient:

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def post(self, url, json):
                return FakeResponse()

        config = ToolathlonServiceConfig(
            server_host='toolathlon.example',
            server_port=8080,
            base_url='http://localhost:8000/v1',
            model_name='local-model',
        )

        with patch('httpx.Client', FakeHttpxClient):
            with self.assertRaisesRegex(RuntimeError, '180 minutes.*3 per IP'):
                ToolathlonServiceClient(config)._submit_job()

    def test_run_private_with_mock_service_flow(self) -> None:
        calls = []
        archive_bytes = _make_task_archive()

        class FakeProcess:

            def poll(self) -> None:
                return None

        class FakeResponse:

            def __init__(
                self,
                payload: Any = None,
                content: bytes = b'',
                status_code: int = 200,
                headers: Optional[dict] = None,
            ) -> None:
                self.payload = payload
                self.content = content
                self.status_code = status_code
                self.headers = headers or {}
                self.text = json.dumps(payload) if payload is not None else ''

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f'HTTP {self.status_code}')

            def json(self) -> Any:
                return self.payload

        class FakeHttpxClient:

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> 'FakeHttpxClient':
                return self

            def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                return False

            def post(self, url: str, json: dict) -> FakeResponse:
                calls.append(('post', url, json))
                return FakeResponse({'job_id': 'job-1', 'client_id': 'client-1'})

            def get(self, url: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> FakeResponse:
                calls.append(('get', url, params))
                if url.endswith('/get_completed_tasks'):
                    return FakeResponse({'completed_tasks': ['find-alita-paper']})
                if url.endswith('/get_task_archive'):
                    return FakeResponse(content=archive_bytes)
                if url.endswith('/poll_job_status'):
                    return FakeResponse({'status': 'completed'})
                if url.endswith('/get_static_files'):
                    return FakeResponse({
                        'eval_stats.json': json.dumps({
                            'passed': 1,
                            'total': 1
                        }),
                        'eval_res_all.jsonl': json.dumps({
                            'task': 'find-alita-paper',
                            'pass': True
                        }) + '\n',
                    })
                return FakeResponse(status_code=404, payload={'detail': 'not found'})

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ToolathlonServiceConfig(
                server_host='toolathlon.example',
                server_port=8080,
                base_url='http://localhost:8000/v1',
                model_name='local-model',
                api_key='secret-local-key',
                task_list=['find-alita-paper'],
                output_dir=Path(tmp_dir) / 'toolathlon',
                poll_interval=0,
            )

            with patch('httpx.Client', FakeHttpxClient):
                with patch.object(ToolathlonServiceClient, '_start_ws_client', return_value=FakeProcess()) as start_ws:
                    with patch.object(ToolathlonServiceClient, '_stop_process') as stop_process:
                        result = ToolathlonServiceClient(config).run_private()

            self.assertEqual(result['job_id'], 'job-1')
            self.assertEqual(result['acc'], 1.0)
            self.assertEqual(result['eval_stats'], {'passed': 1, 'total': 1})
            self.assertEqual(result['task_results'], [{'task': 'find-alita-paper', 'pass': True}])
            self.assertTrue((Path(tmp_dir) / 'toolathlon/finalpool/find-alita-paper/README.md').exists())
            start_ws.assert_called_once_with('job-1')
            stop_process.assert_called_once()

        called_urls = [item[1] for item in calls]
        self.assertIn('http://toolathlon.example:8080/submit_evaluation', called_urls)
        self.assertIn('http://toolathlon.example:8080/get_completed_tasks', called_urls)
        self.assertIn('http://toolathlon.example:8080/get_task_archive', called_urls)
        self.assertIn('http://toolathlon.example:8080/poll_job_status', called_urls)
        self.assertIn('http://toolathlon.example:8080/get_static_files', called_urls)

    def test_rejects_unsafe_task_archive_path(self) -> None:

        class FakeResponse:
            status_code = 200
            headers = {}
            content = _make_task_archive('../../escape.txt')

            def raise_for_status(self) -> None:
                pass

        class FakeHttpxClient:

            def get(self, url: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> FakeResponse:
                return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ToolathlonServiceConfig(output_dir=Path(tmp_dir) / 'toolathlon')
            client = ToolathlonServiceClient(config)
            with self.assertRaisesRegex(RuntimeError, 'Unsafe Toolathlon output path'):
                client._download_task_archive(FakeHttpxClient(), 'job-1', 'find-alita-paper')

    def test_rejects_unsafe_static_file_path(self) -> None:

        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {'../../escape.txt': 'bad'}

        class FakeHttpxClient:

            def get(self, url: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> FakeResponse:
                return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ToolathlonServiceConfig(output_dir=Path(tmp_dir) / 'toolathlon')
            client = ToolathlonServiceClient(config)
            with self.assertRaisesRegex(RuntimeError, 'Unsafe Toolathlon output path'):
                client._download_static_files(FakeHttpxClient(), 'job-1')

    def test_poll_fails_when_ws_process_exits(self) -> None:

        class FakeProcess:

            def poll(self) -> int:
                return 1

        config = ToolathlonServiceConfig(server_host='toolathlon.example', poll_interval=0)
        client = ToolathlonServiceClient(config)
        with patch('httpx.Client', return_value=_HttpxContext()), patch.object(client, '_cancel_job') as cancel_job:
            with self.assertRaisesRegex(RuntimeError, 'WebSocket relay exited unexpectedly'):
                client._poll_until_finished('job-1', FakeProcess())

        cancel_job.assert_called_once()

    def test_poll_downloads_results_after_clean_relay_exit(self) -> None:
        calls = []
        client = ToolathlonServiceClient(ToolathlonServiceConfig(poll_interval=0))
        with (
            patch('httpx.Client', return_value=_HttpxContext()),
            patch.object(client, '_get_job_status', return_value={'status': 'completed'}),
            patch.object(client, '_download_completed_tasks', return_value=set()) as download_tasks,
            patch.object(client, '_download_static_files') as download_static,
            patch.object(client, '_cancel_job') as cancel_job,
        ):
            client._poll_until_finished('job-1', _RelayProcess(0))
            calls.append('finished')

        self.assertEqual(calls, ['finished'])
        download_tasks.assert_called_once()
        download_static.assert_called_once()
        cancel_job.assert_not_called()

    def test_poll_raises_when_clean_relay_exit_but_job_not_completed(self) -> None:
        client = ToolathlonServiceClient(ToolathlonServiceConfig(poll_interval=0))
        with (
            patch('httpx.Client', return_value=_HttpxContext()),
            patch.object(client, '_get_job_status', return_value={'status': 'running'}),
            patch.object(client, '_cancel_job') as cancel_job,
        ):
            with self.assertRaisesRegex(RuntimeError, 'exited unexpectedly with code 0'):
                client._poll_until_finished('job-1', _RelayProcess(0))

        cancel_job.assert_called_once_with('job-1', 'WebSocket relay exited')

    def test_poll_does_not_cancel_failed_job_after_clean_relay_exit(self) -> None:
        client = ToolathlonServiceClient(ToolathlonServiceConfig(poll_interval=0))
        with (
            patch('httpx.Client', return_value=_HttpxContext()),
            patch.object(client, '_get_job_status', return_value={'status': 'failed'}),
            patch.object(client, '_download_static_files') as download_static,
            patch.object(client, '_cancel_job') as cancel_job,
        ):
            with self.assertRaisesRegex(RuntimeError, 'Toolathlon job failed'):
                client._poll_until_finished('job-1', _RelayProcess(0))

        download_static.assert_called_once()
        cancel_job.assert_not_called()

    def test_poll_cancels_job_when_status_poll_fails(self) -> None:
        class FakeResponse:
            status_code = 500

            def raise_for_status(self) -> None:
                raise RuntimeError('HTTP 500')

        class FakeHttpxClient(_HttpxContext):

            def get(self, url: str, params: Optional[dict] = None) -> FakeResponse:
                return FakeResponse()

        client = ToolathlonServiceClient(ToolathlonServiceConfig(poll_interval=0))
        with (
            patch('httpx.Client', return_value=FakeHttpxClient()),
            patch.object(client, '_download_completed_tasks', return_value=set()),
            patch.object(client, '_cancel_job') as cancel_job,
        ):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 500'):
                client._poll_until_finished('job-1', _RelayProcess(None))

        cancel_job.assert_called_once()

    def test_poll_cancels_job_when_status_body_is_not_a_dict(self) -> None:
        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> list:
                return ['not', 'an', 'object']

        class FakeHttpxClient(_HttpxContext):

            def get(self, url: str, params: Optional[dict] = None) -> FakeResponse:
                return FakeResponse()

        client = ToolathlonServiceClient(ToolathlonServiceConfig(poll_interval=0))
        with (
            patch('httpx.Client', return_value=FakeHttpxClient()),
            patch.object(client, '_download_completed_tasks', return_value=set()),
            patch.object(client, '_cancel_job') as cancel_job,
        ):
            with self.assertRaisesRegex(RuntimeError, 'status must be an object'):
                client._poll_until_finished('job-1', _RelayProcess(None))

        cancel_job.assert_called_once()

    def test_poll_cancels_once_and_preserves_abort_exception(self) -> None:
        for error in [RuntimeError('download failed'), KeyboardInterrupt('ctrl-c'), SystemExit(2)]:
            with self.subTest(error=type(error).__name__):
                client = ToolathlonServiceClient(ToolathlonServiceConfig(poll_interval=0))
                with (
                    patch('httpx.Client', return_value=_HttpxContext()),
                    patch.object(client, '_download_completed_tasks', side_effect=error),
                    patch.object(client, '_cancel_job') as cancel_job,
                ):
                    with self.assertRaises(type(error)) as raised:
                        client._poll_until_finished('job-1', _RelayProcess(None))

                self.assertIs(raised.exception, error)
                cancel_job.assert_called_once()

    def test_cancel_failure_does_not_replace_poll_error(self) -> None:
        client = ToolathlonServiceClient(ToolathlonServiceConfig(poll_interval=0))
        poll_error = RuntimeError('poll failed')
        with (
            patch('httpx.Client', return_value=_HttpxContext()),
            patch.object(client, '_download_completed_tasks', side_effect=poll_error),
            patch.object(client, '_cancel_job', side_effect=RuntimeError('cancel failed')),
        ):
            with self.assertLogs('evalscope', level='WARNING'):
                with self.assertRaises(RuntimeError) as raised:
                    client._poll_until_finished('job-1', _RelayProcess(None))

        self.assertIs(raised.exception, poll_error)

    def test_completed_job_download_failure_does_not_cancel(self) -> None:
        client = ToolathlonServiceClient(ToolathlonServiceConfig(poll_interval=0))
        with (
            patch('httpx.Client', return_value=_HttpxContext()),
            patch.object(client, '_download_completed_tasks', return_value=set()),
            patch.object(client, '_get_job_status', return_value={'status': 'completed'}),
            patch.object(client, '_download_static_files', side_effect=RuntimeError('download failed')),
            patch.object(client, '_cancel_job') as cancel_job,
        ):
            with self.assertRaisesRegex(RuntimeError, 'download failed'):
                client._poll_until_finished('job-1', _RelayProcess(None))

        cancel_job.assert_not_called()

    def test_run_private_cancels_job_when_ws_client_start_fails(self) -> None:
        client = ToolathlonServiceClient(ToolathlonServiceConfig())
        start_error = RuntimeError('start failed')
        with (
            patch.object(client, '_prepare_output_dir'),
            patch.object(client, '_submit_job', return_value={'job_id': 'job-1', 'client_id': 'client-1'}),
            patch.object(client, '_start_ws_client', side_effect=start_error),
            patch.object(client, '_cancel_job') as cancel_job,
        ):
            with self.assertRaises(RuntimeError) as raised:
                client.run_private()

        self.assertIs(raised.exception, start_error)
        cancel_job.assert_called_once()

    def test_run_private_cancels_job_when_client_id_is_missing(self) -> None:
        client = ToolathlonServiceClient(ToolathlonServiceConfig())
        with (
            patch.object(client, '_prepare_output_dir'),
            patch.object(client, '_submit_job', return_value={'job_id': 'job-1'}),
            patch.object(client, '_cancel_job', side_effect=RuntimeError('cancel failed')) as cancel_job,
        ):
            with self.assertLogs('evalscope', level='WARNING'):
                with self.assertRaisesRegex(RuntimeError, 'did not return client_id'):
                    client.run_private()

        cancel_job.assert_called_once_with('job-1', 'Missing WebSocket client ID')

    def test_start_ws_client_passes_api_key_via_env_not_argv(self) -> None:
        captured: dict[str, Any] = {}

        class FakePopen:

            def __init__(self, command: list[str], **kwargs: Any) -> None:
                captured['command'] = command
                captured.update(kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = ToolathlonServiceClient(
                ToolathlonServiceConfig(
                    server_host='toolathlon.example',
                    base_url='http://localhost:8000/v1',
                    api_key='secret-local-key',
                    output_dir=Path(tmp_dir),
                )
            )
            with patch.object(toolathlon_client.subprocess, 'Popen', FakePopen):
                client._start_ws_client('job-1')

        command = captured['command']
        self.assertNotIn('--llm-api-key', command)
        self.assertNotIn('secret-local-key', ' '.join(command))
        self.assertEqual(captured['env'][toolathlon_client.WS_CLIENT_API_KEY_ENV], 'secret-local-key')
        self.assertIn('PATH', captured['env'])

    def test_start_ws_client_clears_parent_api_key_when_config_is_none(self) -> None:
        captured: dict[str, Any] = {}

        class FakePopen:

            def __init__(self, command: list[str], **kwargs: Any) -> None:
                captured.update(kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = ToolathlonServiceClient(ToolathlonServiceConfig(api_key=None, output_dir=Path(tmp_dir)))
            with (
                patch.dict(os.environ, {toolathlon_client.WS_CLIENT_API_KEY_ENV: 'stale-parent-secret'}),
                patch.object(toolathlon_client.subprocess, 'Popen', FakePopen),
            ):
                client._start_ws_client('job-1')

        self.assertNotIn(toolathlon_client.WS_CLIENT_API_KEY_ENV, captured['env'])

    def test_ws_client_reads_api_key_from_env(self) -> None:
        captured: dict[str, str] = {}

        async def fake_run_ws_proxy(server_url: str, llm_base_url: str, llm_api_key: str, job_id: str) -> None:
            captured.update(
                server_url=server_url,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                job_id=job_id,
            )

        argv = [
            'ws_client',
            '--server-url',
            'http://toolathlon.example:8081',
            '--llm-base-url',
            'http://localhost:8000/v1',
            '--job-id',
            'job-1',
        ]
        with (
            patch.dict(os.environ, {toolathlon_client.WS_CLIENT_API_KEY_ENV: 'env-secret'}),
            patch.object(toolathlon_ws_client, 'run_ws_proxy', fake_run_ws_proxy),
            patch.object(sys, 'argv', argv),
        ):
            toolathlon_ws_client.main()

        self.assertEqual(captured['llm_api_key'], 'env-secret')

    def test_stop_process_waits_after_kill(self) -> None:
        events = []

        class FakeProcess:
            pid = 4242

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                events.append('terminate')

            def kill(self) -> None:
                events.append('kill')

            def wait(self, timeout: Optional[float] = None) -> int:
                events.append('wait')
                if events.count('wait') == 1:
                    raise subprocess.TimeoutExpired(cmd='relay', timeout=timeout)
                return 0

        ToolathlonServiceClient(ToolathlonServiceConfig())._stop_process(FakeProcess())

        self.assertEqual(events, ['terminate', 'wait', 'kill', 'wait'])

    def test_stop_process_warns_when_killed_process_cannot_be_reaped(self) -> None:
        events = []

        class FakeProcess:
            pid = 4242

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                events.append('terminate')

            def kill(self) -> None:
                events.append('kill')

            def wait(self, timeout: Optional[float] = None) -> int:
                events.append('wait')
                raise subprocess.TimeoutExpired(cmd='relay', timeout=timeout)

        with self.assertLogs('evalscope', level='WARNING') as logs:
            ToolathlonServiceClient(ToolathlonServiceConfig())._stop_process(FakeProcess())

        self.assertEqual(events, ['terminate', 'wait', 'kill', 'wait'])
        self.assertIn('did not exit after SIGKILL', '\n'.join(logs.output))

    def test_ws_proxy_omits_empty_authorization_header(self) -> None:
        headers_seen = {}
        client_lifecycle = {'created': 0, 'closed': 0}

        class FakeResponse:
            status_code = 200

            def json(self) -> dict:
                return {'id': 'chatcmpl-mock'}

        class FakeAsyncClient:

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                client_lifecycle['created'] += 1

            async def __aenter__(self) -> 'FakeAsyncClient':
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                client_lifecycle['closed'] += 1
                return False

            async def post(self, url: str, json: dict, headers: dict) -> FakeResponse:
                headers_seen.update(headers)
                return FakeResponse()

        class FakeWebSocket:

            def __init__(self) -> None:
                self.sent_messages = []

            async def __aiter__(self):
                yield json.dumps({
                    'type': 'new_requests',
                    'requests': [
                        {
                            'request_id': 'request-1',
                            'messages': [],
                        },
                        {
                            'request_id': 'request-2',
                            'messages': [],
                        },
                    ]
                })
                while len(self.sent_messages) < 2:
                    await asyncio.sleep(0)
                yield json.dumps({'type': 'error', 'message': 'done'})

            async def send(self, message: str) -> None:
                self.sent_messages.append(message)

        class FakeConnect:

            async def __aenter__(self) -> FakeWebSocket:
                return FakeWebSocket()

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                return False

        with patch('websockets.connect', return_value=FakeConnect()):
            with patch('httpx.AsyncClient', FakeAsyncClient):
                with self.assertRaisesRegex(RuntimeError, 'done'):
                    asyncio.run(run_ws_proxy('http://toolathlon.example:8081', 'http://localhost:8000/v1', '', 'job-1'))

        self.assertNotIn('Authorization', headers_seen)
        self.assertEqual(client_lifecycle, {'created': 1, 'closed': 1})

    def test_ws_proxy_error_cancels_active_requests(self) -> None:
        request_started = asyncio.Event()
        request_cancelled = asyncio.Event()
        websocket_closed = asyncio.Event()
        client_closed = asyncio.Event()

        class FakeAsyncClient:

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> 'FakeAsyncClient':
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                client_closed.set()
                return False

            async def post(self, url: str, json: dict, headers: dict) -> None:
                request_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    request_cancelled.set()

        class FakeWebSocket:

            async def __aiter__(self):
                try:
                    yield json.dumps({
                        'type': 'new_requests',
                        'requests': [{
                            'request_id': 'request-1',
                            'messages': [],
                        }]
                    })
                    await request_started.wait()
                    yield json.dumps({'type': 'error', 'message': 'relay failed'})
                finally:
                    websocket_closed.set()

            async def send(self, message: str) -> None:
                return None

        class FakeConnect:

            async def __aenter__(self) -> FakeWebSocket:
                return FakeWebSocket()

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                return False

        with patch('websockets.connect', return_value=FakeConnect()):
            with patch('httpx.AsyncClient', FakeAsyncClient):
                with self.assertRaisesRegex(RuntimeError, 'relay failed'):
                    asyncio.run(
                        run_ws_proxy('http://toolathlon.example:8081', 'http://localhost:8000/v1', 'key', 'job-1')
                    )

        self.assertTrue(request_cancelled.is_set())
        self.assertTrue(websocket_closed.is_set())
        self.assertTrue(client_closed.is_set())

    def test_ws_proxy_heartbeat_timeout_cancels_all_tasks(self) -> None:
        request_started = asyncio.Event()
        request_cancelled = asyncio.Event()
        receive_cancelled = asyncio.Event()
        client_closed = asyncio.Event()

        class FakeAsyncClient:

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> 'FakeAsyncClient':
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                client_closed.set()
                return False

            async def post(self, url: str, json: dict, headers: dict) -> None:
                request_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    request_cancelled.set()

        class FakeWebSocket:

            async def __aiter__(self):
                try:
                    yield json.dumps({
                        'type': 'new_requests',
                        'requests': [{
                            'request_id': 'request-1',
                            'messages': [],
                        }]
                    })
                    await asyncio.Event().wait()
                finally:
                    receive_cancelled.set()

            async def send(self, message: str) -> None:
                if json.loads(message).get('type') == 'heartbeat':
                    await request_started.wait()

        class FakeConnect:

            async def __aenter__(self) -> FakeWebSocket:
                return FakeWebSocket()

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                return False

        with patch('websockets.connect', return_value=FakeConnect()):
            with patch('httpx.AsyncClient', FakeAsyncClient):
                with patch.object(toolathlon_client, 'WS_HEARTBEAT_INTERVAL_SECONDS', 0):
                    with patch.object(toolathlon_client, 'WS_HEARTBEAT_TIMEOUT_SECONDS', -1):
                        with self.assertRaisesRegex(TimeoutError, 'heartbeat timed out'):
                            asyncio.run(
                                run_ws_proxy(
                                    'http://toolathlon.example:8081',
                                    'http://localhost:8000/v1',
                                    'key',
                                    'job-1',
                                )
                            )

        self.assertTrue(request_cancelled.is_set())
        self.assertTrue(receive_cancelled.is_set())
        self.assertTrue(client_closed.is_set())

    def test_ws_proxy_request_send_error_propagates(self) -> None:
        receive_cancelled = asyncio.Event()

        class FakeResponse:
            status_code = 200

            def json(self) -> dict:
                return {'id': 'chatcmpl-mock'}

        class FakeAsyncClient:

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> 'FakeAsyncClient':
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                return False

            async def post(self, url: str, json: dict, headers: dict) -> FakeResponse:
                return FakeResponse()

        class FakeWebSocket:

            async def __aiter__(self):
                try:
                    yield json.dumps({
                        'type': 'new_requests',
                        'requests': [{
                            'request_id': 'request-1',
                            'messages': [],
                        }]
                    })
                    await asyncio.Event().wait()
                finally:
                    receive_cancelled.set()

            async def send(self, message: str) -> None:
                raise RuntimeError('websocket send failed')

        class FakeConnect:

            async def __aenter__(self) -> FakeWebSocket:
                return FakeWebSocket()

            async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
                return False

        with patch('websockets.connect', return_value=FakeConnect()):
            with patch('httpx.AsyncClient', FakeAsyncClient):
                with self.assertRaisesRegex(RuntimeError, 'websocket send failed'):
                    asyncio.run(
                        run_ws_proxy('http://toolathlon.example:8081', 'http://localhost:8000/v1', 'key', 'job-1')
                    )

        self.assertTrue(receive_cancelled.is_set())


def _make_task_archive(member_name: str = 'find-alita-paper/README.md') -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        content = b'mock task archive'
        info = tarfile.TarInfo(member_name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


if __name__ == '__main__':
    unittest.main()
