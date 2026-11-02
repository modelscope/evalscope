"""BFCL integration tests using the official data and a local scripted model API."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from collections import Counter
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip('bfcl_eval')

from bfcl_eval.utils import load_dataset_entry

from evalscope import TaskConfig, run_task
from evalscope.report import Report

# Correct actions for the first two official multi_turn_base cases. The dataset,
# function schemas, simulated tools, decoders, and scorer remain unmodified.
_CALLS: dict[str, list[list[tuple[str, dict[str, Any]]]]] = {
    'multi_turn_base_0': [
        [('cd', {'folder': 'document'}), ('mkdir', {'dir_name': 'temp'}),
         ('mv', {'source': 'final_report.pdf', 'destination': 'temp'})],
        [('cd', {'folder': 'temp'}), ('grep', {'file_name': 'final_report.pdf', 'pattern': 'budget analysis'})],
        [('sort', {'file_name': 'final_report.pdf'})],
        [('cd', {'folder': '..'}), ('mv', {'source': 'previous_report.pdf', 'destination': 'temp'}),
         ('cd', {'folder': 'temp'}), ('diff', {'file_name1': 'final_report.pdf', 'file_name2': 'previous_report.pdf'})],
    ],
    'multi_turn_base_1': [
        [('ls', {'a': True})],
        [('cd', {'folder': 'workspace'}), ('mv', {'source': 'log.txt', 'destination': 'archive'})],
        [('cd', {'folder': 'archive'}), ('grep', {'file_name': 'log.txt', 'pattern': 'Error'})],
        [('tail', {'file_name': 'log.txt', 'lines': 20})],
    ],
}
_REPEATS = 3
_SAMPLE_COUNT = len(_CALLS) * _REPEATS


class _ScriptedBFCLServer:

    def __init__(self) -> None:
        entries = load_dataset_entry('multi_turn_base', include_language_specific_hint=False)[:len(_CALLS)]
        assert [entry['id'] for entry in entries] == list(_CALLS)
        self.questions = {
            turn[0]['content']: (entry['id'], turn_index)
            for entry in entries
            for turn_index, turn in enumerate(entry['question'])
        }
        self.requests: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.barriers: dict[tuple[str, str], threading.Barrier] = {}
        self.lock = threading.Lock()
        self.active_requests = 0
        self.max_parallel_requests = 0
        scripted_server = self

        class Handler(BaseHTTPRequestHandler):

            def do_POST(self) -> None:
                try:
                    assert self.path == '/v1/chat/completions'
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    response = scripted_server.respond(body)
                    encoded = json.dumps(response).encode()
                    self.send_response(200)
                except Exception as error:
                    with scripted_server.lock:
                        scripted_server.errors.append(repr(error))
                    encoded = json.dumps({'error': {'message': str(error)}}).encode()
                    self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def api_url(self) -> str:
        return f'http://127.0.0.1:{self.httpd.server_port}/v1/chat/completions/'

    def respond(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body['messages']
        question_index, entry_id, turn_index = next(
            (index, *self.questions[message['content']])
            for index, message in reversed(list(enumerate(messages)))
            if message['role'] == 'user' and message.get('content') in self.questions
        )
        first_step = not any(message['role'] == 'assistant' for message in messages[question_index + 1:])
        with self.lock:
            self.requests.append(body)
            self.active_requests += 1
            self.max_parallel_requests = max(self.max_parallel_requests, self.active_requests)
            barrier = self.barriers.setdefault((body['model'], entry_id), threading.Barrier(_REPEATS))
        try:
            # The real handler creates its simulated API instances before its
            # first HTTP request. Hold all repeats here to exercise overlap.
            if turn_index == 0 and first_step:
                barrier.wait(timeout=30)
            calls = _CALLS[entry_id][turn_index] if first_step else []
            message: dict[str, Any] = {'role': 'assistant', 'content': 'Done.'}
            if body.get('tools'):
                # Explicit [] also covers OpenAI-compatible text-only replies.
                message['tool_calls'] = [
                    {
                        'id': f'call_{index}',
                        'type': 'function',
                        'function': {'name': name, 'arguments': json.dumps(arguments)},
                    }
                    for index, (name, arguments) in enumerate(calls)
                ]
                if calls:
                    message['content'] = None
            elif calls:
                message['content'] = '[' + ', '.join(
                    f'{name}(' + ', '.join(f'{key}={value!r}' for key, value in arguments.items()) + ')'
                    for name, arguments in calls
                ) + ']'
            return {
                'id': 'chatcmpl-bfcl-regression',
                'object': 'chat.completion',
                'created': 0,
                'model': body['model'],
                'choices': [{
                    'index': 0,
                    'message': message,
                    'finish_reason': 'tool_calls' if calls and body.get('tools') else 'stop',
                }],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            }
        finally:
            with self.lock:
                self.active_requests -= 1


@pytest.fixture
def bfcl_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_ScriptedBFCLServer]:
    # The adapter configures these variables; restore them after the test.
    monkeypatch.setenv('OPENAI_API_KEY', 'local-bfcl-test')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://127.0.0.1')
    server = _ScriptedBFCLServer()
    server.thread.start()
    try:
        yield server
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()
        server.thread.join(timeout=5)


def _task_config(server: _ScriptedBFCLServer, output: Path, model: str, is_fc_model: bool) -> TaskConfig:
    return TaskConfig(
        model=model,
        api_url=server.api_url,
        api_key='local-bfcl-test',
        eval_type='openai_api',
        datasets=['bfcl_v4'],
        dataset_args={
            'bfcl_v4': {
                'subset_list': ['multi_turn_base'],
                'extra_params': {'is_fc_model': is_fc_model},
            },
        },
        limit=len(_CALLS),
        repeats=_REPEATS,
        eval_batch_size=4,
        generation_config={'temperature': 0},
        analysis_report=False,
        enable_progress_tracker=False,
        work_dir=str(output),
        no_timestamp=True,
    )


def _cache_rows(output: Path, kind: str) -> list[dict[str, Any]]:
    files = list((output / kind).rglob('bfcl_v4_multi_turn_base.jsonl'))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]


def _assert_report(report: Report) -> None:
    subsets = [subset for metric in report.metrics for category in metric.categories for subset in category.subsets]
    subset = next(subset for subset in subsets if subset.name == 'multi_turn_base')
    assert subset.num == _SAMPLE_COUNT
    assert subset.score == pytest.approx(1.0)


def _assert_artifacts(output: Path, model: str) -> None:
    predictions = _cache_rows(output, 'predictions')
    reviews = _cache_rows(output, 'reviews')
    assert len(predictions) == len(reviews) == _SAMPLE_COUNT
    expected_ids = Counter({entry_id: _REPEATS for entry_id in _CALLS})
    assert Counter(row['metadata']['id'] for row in predictions) == expected_ids
    assert Counter(row['sample_score']['sample_metadata']['id'] for row in reviews) == expected_ids
    for review in reviews:
        assert review['sample_score']['score']['value'] == {'acc': 1}
        assert review['sample_score']['score']['metadata']['valid'] is True
        tools = [message for message in review['messages'] if message['role'] == 'tool']
        assert tools
        tool_errors = [
            message['content'] for message in tools
            if '"error"' in message['content'] or 'Error during execution:' in message['content']
        ]
        assert not tool_errors, (review['sample_score']['sample_metadata']['id'], tool_errors)
        assert any(message['role'] == 'assistant' and message.get('perf_metrics') for message in review['messages'])
    report_files = list((output / 'reports').rglob('bfcl_v4.json'))
    assert len(report_files) == 1
    report = Report.model_validate_json(report_files[0].read_text())
    assert report.model_name == model
    _assert_report(report)
    snapshot = yaml.safe_load((output / 'configs' / 'task_config.yaml').read_text())
    assert snapshot['evaluation_identity']['benchmarks']['bfcl_v4']['evaluation_version'] == 'v1.1'


@pytest.mark.parametrize('is_fc_model', [False, True], ids=['prompting', 'function-calling'])
def test_bfcl_official_multi_turn_pipeline_and_cached_review(
    tmp_path: Path, bfcl_server: _ScriptedBFCLServer, is_fc_model: bool
) -> None:
    """Parallel repeats, later models, and forced reviews each start from clean state."""
    model_prefix = f'bfcl-pipeline-{uuid.uuid4().hex}'
    configs = [
        _task_config(bfcl_server, tmp_path / str(index), f'{model_prefix}-{index}', is_fc_model)
        for index in range(2)
    ]
    # Use the public multi-task entry point so both model tasks share one process.
    reports = run_task(configs)
    assert not bfcl_server.errors
    assert bfcl_server.max_parallel_requests >= _REPEATS
    assert Counter(request['model'] for request in bfcl_server.requests) == {
        config.model: _SAMPLE_COUNT * 8 for config in configs
    }
    for index, (config, result) in enumerate(zip(configs, reports)):
        _assert_report(result['bfcl_v4'])
        output = Path(config.work_dir)
        _assert_artifacts(output, config.model)
        cached_predictions = _cache_rows(output, 'predictions')
        request_count = len(bfcl_server.requests)

        cached = config.model_copy(deep=True, update={'use_cache': str(output)})
        _assert_report(run_task(cached)['bfcl_v4'])
        assert len(bfcl_server.requests) == request_count

        if index == 0:
            # Simulate predictions/reviews saved before the scoring isolation
            # change. Normal resume must not silently reuse those reviews.
            snapshot_path = output / 'configs' / 'task_config.yaml'
            snapshot = yaml.safe_load(snapshot_path.read_text())
            identity = snapshot['evaluation_identity']['benchmarks']['bfcl_v4']
            identity.update(evaluation_version='v1.0', fingerprint='sha256:' + '0' * 64)
            snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
            with pytest.raises(ValueError, match='rerun_review=True'):
                run_task(cached.model_copy(deep=True))
            assert len(bfcl_server.requests) == request_count

        rerun = config.model_copy(deep=True, update={'use_cache': str(output), 'rerun_review': True})
        _assert_report(run_task(rerun)['bfcl_v4'])
        assert len(bfcl_server.requests) == request_count
        assert _cache_rows(output, 'predictions') == cached_predictions
        _assert_artifacts(output, config.model)
        if index == 0:
            snapshot = yaml.safe_load((output / 'configs' / 'task_config.yaml').read_text())
            source = snapshot['evaluation_identity']['benchmarks']['bfcl_v4']['cache_source']
            assert source['evaluation_version'] == 'v1.0'
            assert source['prediction_reused'] is True
    assert not bfcl_server.errors


def test_bfcl_cli_official_multi_turn_smoke(tmp_path: Path, bfcl_server: _ScriptedBFCLServer) -> None:
    """Exercise the real CLI in a fresh interpreter against the same local API."""
    model = f'bfcl-cli-{uuid.uuid4().hex}'
    config = _task_config(bfcl_server, tmp_path, model, True)
    command = [
        sys.executable, '-m', 'evalscope.cli.cli', 'eval',
        '--model', model, '--api-url', bfcl_server.api_url, '--api-key', 'local-bfcl-test',
        '--eval-type', 'openai_api', '--datasets', 'bfcl_v4',
        '--dataset-args', json.dumps(config.dataset_args),
        '--limit', str(len(_CALLS)), '--repeats', str(_REPEATS), '--eval-batch-size', '4',
        '--generation-config', 'temperature=0', '--work-dir', str(tmp_path), '--no-timestamp',
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not bfcl_server.errors
    assert len(bfcl_server.requests) == _SAMPLE_COUNT * 8
    assert {request['model'] for request in bfcl_server.requests} == {model}
    _assert_artifacts(tmp_path, model)
