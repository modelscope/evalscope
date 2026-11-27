import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from evalscope.project_eval.checks import run_checks
from evalscope.project_eval.schema import EvalTask


def task(check):
    return EvalTask.model_validate({
        'id': 't1',
        'question': 'q',
        'rubric': [{'id': 'c1', 'criterion': 'ok'}],
        'checks': [check],
        'exec': {
            'type': 'agent_api',
            'protocol': 'rest_steps',
            'agent_url_env': 'EVAL_AGENT_URL',
            'trace': {'mode': 'auto'},
            'session': {'id_from': 'sample_id'},
        },
    })


def test_json_path_check(tmp_path):
    (tmp_path / 'state.json').write_text(json.dumps({'balance_cent': 200000}))
    result = run_checks(task({
        'id': 'j-1',
        'type': 'json_path',
        'path': 'state.json',
        'json_path': 'balance_cent',
        'expect': 200000,
    }), tmp_path)[0]
    assert result.ok is True
    assert result.evidence['value'] == 200000


def test_sqlite_check_reads_final_state(tmp_path):
    db = tmp_path / 'agent.db'
    con = sqlite3.connect(db)
    con.execute('create table agent_audit_log(session_id text, tool text)')
    con.execute("insert into agent_audit_log values ('t1', 'get_balance')")
    con.commit()
    con.close()
    result = run_checks(task({
        'id': 'db-1',
        'type': 'sqlite',
        'dsn': 'agent.db',
        'query': "select tool from agent_audit_log where session_id='t1'",
        'expect': 'get_balance',
    }), tmp_path)[0]
    assert result.ok is True
    assert result.evidence['rows'] == [['get_balance']]


def test_path_escape_and_sensitive_are_rejected(tmp_path):
    for check in [
        {'id': 'f-1', 'type': 'file_exists', 'path': '../etc/passwd'},
        {'id': 'f-2', 'type': 'file_exists', 'path': '.env'},
    ]:
        result = run_checks(task(check), tmp_path)[0]
        assert result.ok is False
        assert result.error


def test_http_prefix_is_enforced(tmp_path):
    result = run_checks(task({
        'id': 'h-1',
        'type': 'http',
        'method': 'GET',
        'url': 'https://example.com',
        'expect_status': 200,
    }), tmp_path)[0]
    assert result.ok is False
    assert 'not allowed' in result.error


def test_mutating_http_method_is_rejected():
    with pytest.raises(ValidationError):
        task({
            'id': 'h-2',
            'type': 'http',
            'method': 'POST',
            'url': 'http://127.0.0.1:1/state',
        })


def test_symlink_is_rejected(tmp_path):
    target = tmp_path / 'target.txt'
    target.write_text('done', encoding='utf-8')
    link = tmp_path / 'link.txt'
    link.symlink_to(target)
    result = run_checks(task({'id': 'f-link', 'type': 'file_exists', 'path': 'link.txt'}), tmp_path)[0]
    assert result.ok is False
    assert 'symlink' in result.error


def test_file_content_check(tmp_path):
    (tmp_path / 'out.txt').write_text('done', encoding='utf-8')
    result = run_checks(task({
        'id': 'f-3',
        'type': 'file_content',
        'path': 'out.txt',
        'contains': 'done',
    }), tmp_path)[0]
    assert result.ok is True


class _StateHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({'audit': [{'tool': 'get_balance'}], 'balance_cent': 200000}).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_http_json_check_reads_state_endpoint(tmp_path):
    server = ThreadingHTTPServer(('127.0.0.1', 0), _StateHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        result = run_checks(task({
            'id': 'hj-1',
            'type': 'http_json',
            'url': f'http://127.0.0.1:{port}/v1/eval/state?session_id=t1',
            'json_path': 'audit.0.tool',
            'expect': 'get_balance',
        }), tmp_path)[0]
    finally:
        server.shutdown()
    assert result.ok is True
    assert result.evidence['value'] == 'get_balance'


def test_sqlite_mismatch_reports_actual_value(tmp_path):
    db = tmp_path / 'agent.db'
    con = sqlite3.connect(db)
    con.execute('create table agent_audit_log(session_id text, tool text)')
    con.execute("insert into agent_audit_log values ('t1', 'get_balance')")
    con.commit()
    con.close()
    result = run_checks(task({
        'id': 'db-x', 'type': 'sqlite', 'dsn': 'agent.db',
        'query': "select tool from agent_audit_log where session_id='t1'",
        'expect': 'refund_recharge',
    }), tmp_path)[0]
    assert result.ok is False
    assert result.evidence['rows'] == [['get_balance']]


def test_sensitive_directory_is_rejected(tmp_path):
    (tmp_path / '.ssh').mkdir()
    (tmp_path / '.ssh' / 'config').write_text('Host *', encoding='utf-8')
    result = run_checks(task({'id': 's-1', 'type': 'file_exists',
                              'path': '.ssh/config'}), tmp_path)[0]
    assert result.ok is False
    assert 'sensitive' in result.error


def test_missing_file_is_not_ok(tmp_path):
    result = run_checks(task({'id': 'f-9', 'type': 'file_exists',
                              'path': 'missing.txt'}), tmp_path)[0]
    assert result.ok is False
