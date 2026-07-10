import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

from evalscope.benchmarks.toolathlon.client import ToolathlonServiceClient, ToolathlonServiceConfig, _extract_accuracy


class TestToolathlonClient(unittest.TestCase):

    def test_extract_accuracy_from_stats(self):
        self.assertEqual(_extract_accuracy({'passed': 3, 'total': 4}, []), 0.75)
        self.assertEqual(_extract_accuracy({'pass_rate': 0.5}, []), 0.5)
        self.assertEqual(_extract_accuracy({}, [{'pass': True}, {'pass': False}]), 0.5)

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
                with patch.object(ToolathlonServiceClient, '_start_ws_client', return_value=object()) as start_ws:
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


def _make_task_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        content = b'mock task archive'
        info = tarfile.TarInfo('find-alita-paper/README.md')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


if __name__ == '__main__':
    unittest.main()
