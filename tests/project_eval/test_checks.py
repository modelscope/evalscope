import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evalscope.project_eval.checks import run_checks
from evalscope.project_eval.schema import EvalTask


def task(check: dict[str, Any]) -> EvalTask:
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


def test_json_path_check(tmp_path: Path) -> None:
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


def test_sqlite_check_reads_final_state(tmp_path: Path) -> None:
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


def test_path_escape_and_sensitive_are_rejected(tmp_path: Path) -> None:
    for check in [
        {'id': 'f-1', 'type': 'file_exists', 'path': '../etc/passwd'},
        {'id': 'f-2', 'type': 'file_exists', 'path': '.env'},
    ]:
        result = run_checks(task(check), tmp_path)[0]
        assert result.ok is False
        assert result.error


def test_http_prefix_is_enforced(tmp_path: Path) -> None:
    result = run_checks(task({
        'id': 'h-1',
        'type': 'http',
        'method': 'GET',
        'url': 'https://example.com',
        'expect_status': 200,
    }), tmp_path)[0]
    assert result.ok is False
    assert 'not allowed' in result.error


@pytest.mark.parametrize(
    'url',
    [
        'http://127.0.0.1.evil.example/state',
        'http://localhost.evil.example/state',
        'http://127.0.0.1.nip.io/state',
        'http://127.0.0.1:9000@evil.example/state',
        'http://localhost:9000@evil.example/state',
    ],
)
def test_http_prefix_cannot_be_bypassed_by_host_suffix(
    url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        'evalscope.project_eval.checks.requests.get',
        lambda target, **_: calls.append(target) or None,
    )
    result = run_checks(task({
        'id': 'h-bypass',
        'type': 'http_json',
        'url': url,
        'json_path': 'ok',
        'expect': True,
    }), tmp_path)[0]
    assert result.ok is False
    assert 'not allowed' in result.error
    assert calls == []


def test_empty_http_prefix_is_rejected(tmp_path: Path) -> None:
    result = run_checks(
        task({
            'id': 'h-empty',
            'type': 'http',
            'method': 'GET',
            'url': 'http://127.0.0.1:1/state',
            'expect_status': 200,
        }),
        tmp_path,
        allowed_http_prefixes=('',),
    )[0]
    assert result.ok is False
    assert 'not allowed' in result.error


def test_http_prefix_path_must_match_a_segment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        'evalscope.project_eval.checks.requests.get',
        lambda target, **_: calls.append(target) or None,
    )
    result = run_checks(
        task({
            'id': 'h-path',
            'type': 'http_json',
            'url': 'https://state.example/v1evil',
            'json_path': 'ok',
            'expect': True,
        }),
        tmp_path,
        allowed_http_prefixes=('https://state.example/v1',),
    )[0]
    assert result.ok is False
    assert calls == []


class _RemoteStateResponse:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {'ok': True}

    def raise_for_status(self) -> None:
        return None


def test_remote_state_api_needs_explicit_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        'evalscope.project_eval.checks.requests.get',
        lambda url, **_: calls.append(url) or _RemoteStateResponse(),
    )
    remote_task = task({
        'id': 'remote-1',
        'type': 'http_json',
        'url': 'http://state.example/v1/state',
        'json_path': 'ok',
        'expect': True,
    })

    rejected = run_checks(remote_task, tmp_path)[0]
    assert rejected.ok is False
    assert 'not allowed' in rejected.error
    assert calls == []

    accepted = run_checks(
        remote_task,
        tmp_path,
        allowed_http_prefixes=('http://state.example',),
    )[0]
    assert accepted.ok is True
    assert calls == ['http://state.example/v1/state']


def test_mutating_http_method_is_rejected() -> None:
    with pytest.raises(ValidationError):
        task({
            'id': 'h-2',
            'type': 'http',
            'method': 'POST',
            'url': 'http://127.0.0.1:1/state',
        })


def test_non_http_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        task({
            'id': 'h-3',
            'type': 'http',
            'method': 'GET',
            'url': 'file:///etc/passwd',
        })


def test_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / 'target.txt'
    target.write_text('done', encoding='utf-8')
    link = tmp_path / 'link.txt'
    link.symlink_to(target)
    result = run_checks(task({'id': 'f-link', 'type': 'file_exists', 'path': 'link.txt'}), tmp_path)[0]
    assert result.ok is False
    assert 'symlink' in result.error


def test_file_content_check(tmp_path: Path) -> None:
    (tmp_path / 'out.txt').write_text('done', encoding='utf-8')
    result = run_checks(task({
        'id': 'f-3',
        'type': 'file_content',
        'path': 'out.txt',
        'contains': 'done',
    }), tmp_path)[0]
    assert result.ok is True


class _StateHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({'audit': [{'tool': 'get_balance'}], 'balance_cent': 200000}).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


def test_http_json_check_reads_state_endpoint(tmp_path: Path) -> None:
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


def test_sqlite_mismatch_reports_actual_value(tmp_path: Path) -> None:
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


def test_sensitive_directory_is_rejected(tmp_path: Path) -> None:
    (tmp_path / '.ssh').mkdir()
    (tmp_path / '.ssh' / 'config').write_text('Host *', encoding='utf-8')
    result = run_checks(task({'id': 's-1', 'type': 'file_exists',
                              'path': '.ssh/config'}), tmp_path)[0]
    assert result.ok is False
    assert 'sensitive' in result.error


def test_missing_file_is_not_ok(tmp_path: Path) -> None:
    result = run_checks(task({'id': 'f-9', 'type': 'file_exists',
                              'path': 'missing.txt'}), tmp_path)[0]
    assert result.ok is False
