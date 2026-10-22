import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from jinja2 import Environment, FileSystemLoader

from evalscope.report import ReportRef
from evalscope.service.api_models import ActiveTaskInfo
from evalscope.service.app import create_app
from evalscope.service.report_meta_cache import (
    build_report_meta_cached,
    clear_report_meta_cache,
    list_etag,
    report_ref_fingerprint,
)
from evalscope.service.utils import log
from evalscope.service.utils.process import list_active_tasks, register_process, unregister_process
from evalscope.utils.uri_utils import sanitize_endpoint, validate_http_endpoint


@pytest.mark.parametrize('url, expected', [
    ('http://user:password@Example.COM:8000/v1?token=secret#x', 'example.com:8000'),
    ('http://[2001:db8::1]:8000/v1', '[2001:db8::1]:8000'),
    ('http://[2001:db8::1:8000]/v1', '[2001:db8::1:8000]'),
    ('http://host:0', 'host:0'), ('host:8080/path', 'host:8080'),
    ('https://host:bad/path', None), ('http://host:65536/', None),
    ('http://[invalid]/', None), ('http://some host/', None), (None, None), ('', None),
])
def test_endpoint_display(url: str | None, expected: str | None) -> None:
    assert sanitize_endpoint(url) == expected


@pytest.mark.parametrize('url', ['http://host:9000/v1?token=x', 'https://[::1]:8000/v1', 'https://host/v1'])
def test_valid_transport_url(url: str) -> None:
    validate_http_endpoint(url)


@pytest.mark.parametrize('url', ['', 'javascript:alert(1)', 'http://host:bad', 'http://user:secret@host', 'http://some host'])
def test_invalid_transport_url_without_input_echo(url: str) -> None:
    with pytest.raises(ValueError) as error:
        validate_http_endpoint(url)
    assert 'secret' not in str(error.value)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from evalscope.service.blueprints import tasks
    monkeypatch.setattr(log, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(tasks, 'OUTPUT_DIR', str(tmp_path))
    yield create_app(outputs=str(tmp_path)).test_client()
    for task in list_active_tasks():
        unregister_process(task['task_id'])


def test_summary_is_whitelisted_and_lifecycle_is_authoritative(client, tmp_path: Path) -> None:
    config = SimpleNamespace(model='m', api_url='http://host:9000/v1?token=SECRET', api_key='SECRET',
                             datasets=['general_qa'], limit=2, eval_batch_size=1, repeats=1,
                             generation_config={'max_tokens': 8, 'extra_body': {'token': 'SECRET'}})
    info = log.build_task_info('eval_safe', 'eval', config, 'server-A')
    manifest = log.read_task_manifest(str(tmp_path / 'eval_safe'))
    assert 'SECRET' not in json.dumps(manifest)
    assert manifest['config']['max_tokens'] == 8
    register_process('eval_safe', Mock(), info)
    progress = tmp_path / 'eval_safe' / 'progress.json'
    progress.write_text('{"percent":100,"status":"completed"}')
    response = client.get('/api/v1/tasks/eval_safe')
    assert response.status_code == 200
    assert response.json['status'] == 'finalizing'
    assert response.json['report_url'] is None
    unregister_process('eval_safe')
    log.finish_task('eval_safe', 'error')
    assert client.get('/api/v1/tasks/eval_safe').json['status'] == 'error'
    log.finish_task('eval_safe', 'stopped')
    assert client.get('/api/v1/tasks/eval_safe').json['status'] == 'stopped'


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), '50'])
def test_invalid_progress_is_unknown(client, tmp_path: Path, value: object) -> None:
    register_process('perf_p', Mock(), ActiveTaskInfo(task_id='perf_p', kind='perf'))
    folder = tmp_path / 'perf_p' / 'perf'
    folder.mkdir(parents=True)
    (folder / 'progress.json').write_text(json.dumps({'percent': value}))
    task = client.get('/api/v1/tasks/active').json['tasks'][0]
    assert task['percent'] is None
    assert task['status'] == 'preparing'


def test_old_metadata_and_disappeared_process_are_not_fabricated(client, tmp_path: Path) -> None:
    log.write_task_manifest('eval_old', {'kind': 'eval', 'alias': 'old', 'api_url': 'https://user:SECRET@host:8000?key=SECRET'})
    task = client.get('/api/v1/tasks/eval_old').json
    assert task['status'] == 'unknown'
    assert task['config'] is None
    assert task['endpoint'] == 'host:8000'
    assert 'SECRET' not in json.dumps(task)
    assert client.get('/api/v1/tasks/missing').status_code == 404


def test_task_path_safety(client, tmp_path: Path) -> None:
    for task_id in ('.', '..', '../escape', 'a\\b'):
        with pytest.raises(ValueError):
            log.write_task_manifest(task_id, {})
    outside = tmp_path.parent / 'outside'
    outside.mkdir(exist_ok=True)
    (tmp_path / 'link').symlink_to(outside, target_is_directory=True)
    assert client.get('/api/v1/tasks/link').status_code == 400
    local = tmp_path / 'local'
    local.mkdir()
    (local / 'perf').symlink_to(outside, target_is_directory=True)
    assert client.get('/api/v1/tasks/local').status_code == 400
    (local / 'task_info.json').symlink_to(outside / 'task_info.json')
    assert log.read_task_manifest(str(local)) is None
    with pytest.raises(ValueError):
        log.write_task_manifest('local', {})


def test_manifest_changes_metadata_cache_and_etag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_report_meta_cache()
    monkeypatch.setattr(log, 'OUTPUT_DIR', str(tmp_path))
    ref = ReportRef(run_id='run', model_id='model')
    folder = tmp_path / 'run' / 'reports' / 'model'
    folder.mkdir(parents=True)
    (folder / 'report.json').write_text('{}')
    def compute(_ref: ReportRef, root: str) -> dict:
        return {'alias': (log.read_task_manifest(str(Path(root) / 'run')) or {}).get('alias')}
    fingerprints = []
    for alias in ('A', 'B'):
        log.write_task_manifest('run', {'alias': alias})
        fingerprint = report_ref_fingerprint(str(tmp_path), ref)
        fingerprints.append(fingerprint)
        assert build_report_meta_cached(str(tmp_path), ref, fingerprint, compute)['alias'] == alias
    assert fingerprints[0] != fingerprints[1]
    assert list_etag([(ref.key, fingerprints[0])], []) != list_etag([(ref.key, fingerprints[1])], [])


@pytest.mark.parametrize('api', ['openai', 'openai_responses'])
def test_query_survives_endpoint_resolution(api: str) -> None:
    from evalscope.perf.arguments import Arguments
    args = Arguments(model='m', api=api, url='http://host:8000/v1?token=test')
    path = 'responses' if api == 'openai_responses' else 'chat/completions'
    assert args.url == f'http://host:8000/v1/{path}?token=test'
    assert args.apply_chat_template == (api == 'openai')


@pytest.mark.parametrize('scope', ['eval', 'perf'])
def test_bad_request_is_400_without_echoing_config(client, scope: str) -> None:
    headers = {'EvalScope-Task-Id': 'task_safe'}
    assert client.post(f'/api/v1/{scope}/invoke', json=['SECRET'], headers=headers).status_code == 400
    payload = {'model': 'm', 'datasets': ['general_qa'], 'url': 'http://u:SECRET@host', 'api_url': 'http://u:SECRET@host'}
    response = client.post(f'/api/v1/{scope}/invoke', json=payload, headers=headers)
    assert response.status_code == 400
    assert b'SECRET' not in response.data
    assert client.post(f'/api/v1/{scope}/stop?task_id=..').status_code == 400
    assert client.get(f'/api/v1/{scope}/progress?task_id=..').status_code == 400


def test_unrebuildable_old_html_is_download_only(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.service.blueprints import perf
    monkeypatch.setattr(perf, 'OUTPUT_DIR', str(tmp_path))
    folder = tmp_path / 'old_perf' / 'perf'
    folder.mkdir(parents=True)
    (folder / 'perf_report.html').write_text('<script>window.oldReportExecuted=true</script>')
    response = client.get('/api/v1/perf/report?task_id=old_perf')
    assert response.status_code == 200
    assert response.headers['Content-Disposition'].startswith('attachment;')


def test_render_marker_only_identifies_the_generated_header(tmp_path: Path) -> None:
    from evalscope.perf.utils.report.generate_report import PERF_HTML_MARKER, is_current_perf_report

    report = tmp_path / 'perf_report.html'
    assert not is_current_perf_report(str(report))
    report.write_text(f'<!DOCTYPE html>\n{PERF_HTML_MARKER}\n<html></html>', encoding='utf-8')
    assert is_current_perf_report(str(report))
    report.write_text(f'<html><body>{PERF_HTML_MARKER}</body></html>', encoding='utf-8')
    assert not is_current_perf_report(str(report))


def test_html_identity_text_is_escaped_without_escaping_charts() -> None:
    root = Path(__file__).parents[2] / 'evalscope' / 'report' / 'template'
    template = Environment(loader=FileSystemLoader(root), autoescape=False).get_template('perf_report.html.j2')
    value = '<script>window.aliasExecuted=true</script>'
    html = template.render(model=value, alias=value, api_type='openai', dataset='qa', endpoint='host:8000',
                           basic_info={'Alias': value}, default_lang='en', generated_at='',
                           latency_tabs=[{'label': 'Latency', 'chart': '<div id="trusted-chart"></div>'}])
    assert value not in html
    assert '&lt;script&gt;window.aliasExecuted=true&lt;/script&gt;' in html
    assert '<div id="trusted-chart"></div>' in html
