import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from evalscope import TaskConfig, run_task

DB_PATH = ''


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get('content-length', '0'))
        body = json.loads(self.rfile.read(length) or b'{}')
        if self.path.endswith('/v1/eval/chat'):
            session_id = body.get('session_id')
            message = str(body.get('message') or '')
            steps: list[dict[str, Any]] = []
            if '余额' in message:
                connection = sqlite3.connect(DB_PATH)
                connection.execute(
                    'insert into agent_audit_log(session_id, tool, confirmed) values (?, ?, ?)',
                    (session_id, 'get_balance', 0),
                )
                connection.commit()
                connection.close()
                steps = [{'type': 'tool', 'name': 'get_balance', 'arguments': {}, 'result': {'balance_cent': 200000}}]
            payload = {
                'reply': '当前余额 2000 元',
                'steps': steps,
                'state': {'session_id': session_id},
                'usage': {'total_tokens': 4},
            }
        else:
            payload = {
                'id': 'judge',
                'model': 'stub-judge',
                'choices': [{'message': {'content': json.dumps({'explanation': 'ok', 'criteria_met': True})}}],
            }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _task(task_id: str, question: str, expect_count: int) -> dict[str, Any]:
    return {
        'id': task_id,
        'version': 1,
        'question': question,
        'system': '',
        'messages': [],
        'rubric': [{
            'id': 'c1', 'criterion': '符合预期', 'weight': 1,
            'must_pass': True, 'evidence_requirement': 'outcome',
        }],
        'checks': [{
            'id': 'db-1', 'type': 'sqlite', 'dsn': 'agent.db',
            'query': f"select count(*) from agent_audit_log where session_id='{task_id}'",
            'expect': expect_count,
        }],
        'exec': {
            'type': 'agent_api', 'protocol': 'rest_steps',
            'agent_url_env': 'EVAL_AGENT_URL', 'trace': {'mode': 'auto'},
            'session': {'id_from': 'sample_id'},
        },
        'pass_threshold': 0.6,
        'source_refs': [], 'origin': 'fact', 'materials': [], 'metadata': {},
    }


def test_sqlite_checks_read_real_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    global DB_PATH
    DB_PATH = str(tmp_path / 'agent.db')
    connection = sqlite3.connect(DB_PATH)
    connection.execute('create table agent_audit_log(session_id text, tool text, confirmed integer)')
    connection.commit()
    connection.close()

    eval_set = tmp_path / 'eval_set.v1.jsonl'
    eval_set.write_text(
        '\n'.join(json.dumps(task) for task in [
            _task('t-balance', '查一下当前余额', 1),
            _task('t-silent', '只回复谢谢', 0),
        ]),
        encoding='utf-8',
    )

    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv('EVAL_AGENT_URL', f'http://127.0.0.1:{port}')
    try:
        run_task(TaskConfig(
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
        ))
    finally:
        server.shutdown()

    report = tmp_path / 'outputs' / 'reports' / 'agent' / 'project_eval.json'
    data = json.loads(report.read_text(encoding='utf-8'))
    metrics = {item['identity']['name']: item['score'] for item in data['metrics']}
    assert metrics['task_pass'] == 1.0

    connection = sqlite3.connect(DB_PATH)
    rows = connection.execute(
        "select tool from agent_audit_log where session_id='t-balance'"
    ).fetchall()
    connection.close()
    assert rows == [('get_balance',)]
