import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evalscope import TaskConfig, run_task


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get('content-length', '0'))
        self.rfile.read(length)
        if self.path.endswith('/v1/eval/chat'):
            payload = {
                'reply': '余额 2000 元',
                'trace_id': 'tr-1',
                'steps': [{'type': 'tool', 'name': 'get_balance', 'arguments': {}, 'result': {'balance_cent': 200000}}],
                'state': {'balance_cent': 200000},
                'usage': {'total_tokens': 10},
            }
        else:
            payload = {'id': 'stub', 'model': 'stub-judge',
                       'choices': [{'message': {'content': json.dumps({'explanation': 'ok', 'criteria_met': True})}}]}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _task(task_id, balance_expect, must_pass=True):
    return {
        'id': task_id,
        'version': 1,
        'question': '查余额',
        'system': '客服',
        'messages': [],
        'rubric': [{
            'id': 'c1',
            'criterion': '余额正确',
            'must_pass': must_pass,
            'evidence_requirement': 'outcome',
        }],
        'checks': [{
            'id': 'j-1',
            'type': 'json_path',
            'path': 'state.json',
            'json_path': 'balance_cent',
            'expect': balance_expect,
        }],
        'exec': {
            'type': 'agent_api',
            'protocol': 'rest_steps',
            'agent_url_env': 'EVAL_AGENT_URL',
            'trace': {'mode': 'auto'},
            'session': {'id_from': 'sample_id'},
        },
        'source_refs': [],
        'origin': 'fact',
        'materials': [],
        'metadata': {},
    }


def test_project_eval_mock_run_writes_report(tmp_path, monkeypatch):
    (tmp_path / 'state.json').write_text(json.dumps({'balance_cent': 200000}))
    eval_set = tmp_path / 'eval_set.v1.jsonl'
    eval_set.write_text(
        '\n'.join(json.dumps(task) for task in [
            _task('ok', 200000),
            _task('bad', 1),
        ]),
        encoding='utf-8',
    )

    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv('EVAL_AGENT_URL', f'http://127.0.0.1:{port}')
    try:
        config = TaskConfig(
            model='agent',
            eval_type='mock_llm',
            datasets=['project_eval'],
            dataset_args={'project_eval': {
                'local_path': str(eval_set),
                'project_root': str(tmp_path),
            }},
            eval_batch_size=1,
            work_dir=str(tmp_path / 'outputs'),
            no_timestamp=True,
            judge={
                'strategy': 'llm',
                'models': {
                    'model_id': 'stub-judge',
                    'api_url': f'http://127.0.0.1:{port}/v1',
                    'api_key': 'EMPTY',
                    'eval_type': 'openai_api',
                },
            },
        )
        run_task(config)
    finally:
        server.shutdown()

    reports = list((tmp_path / 'outputs' / 'reports').glob('*/*.json'))
    assert reports, 'report was not written'
    text = reports[0].read_text(encoding='utf-8')
    assert 'task_pass' in text

    reviews = list((tmp_path / 'outputs' / 'reviews').rglob('*.jsonl'))
    assert reviews, 'review file was not written'
    review_text = '\n'.join(path.read_text(encoding='utf-8') for path in reviews)
    assert 'rubric_verdicts' in review_text
    assert 'runtime_checks' in review_text or 'checks' in review_text
