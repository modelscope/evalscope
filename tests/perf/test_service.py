from flask import Flask

from evalscope.perf import PerfConfig
from evalscope.service.blueprints import perf as perf_blueprint


def _app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(perf_blueprint.bp_perf)
    return app


def _request() -> dict:
    return {
        'target': {
            'model': 'fake',
            'base_url': 'http://127.0.0.1:8000/v1',
            'protocol': 'openai_chat',
        },
        'workload': {
            'name': 'prompt',
            'prompt': 'hello'
        },
        'suite': {
            'loads': [
                {
                    'mode': 'closed_loop',
                    'concurrency': 2,
                    'request_count': 3
                },
            ]
        },
    }


def test_service_validates_and_passes_typed_config(monkeypatch, tmp_path) -> None:
    received = []

    def fake_run(func, config, task_id):
        received.append(config)
        return {'run_id': config.output.run_id}

    monkeypatch.setattr(perf_blueprint, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(perf_blueprint, 'create_log_file', lambda *args: None)
    monkeypatch.setattr(perf_blueprint, 'run_in_subprocess', fake_run)
    response = _app().test_client().post(
        '/api/v1/perf/invoke',
        json=_request(),
        headers={'EvalScope-Task-Id': 'service-test'},
    )
    assert response.status_code == 200
    assert response.get_json()['result']['run_id'] == 'perf'
    assert len(received) == 1
    assert isinstance(received[0], PerfConfig)
    assert received[0].output.root == str(tmp_path / 'service-test')


def test_service_returns_400_for_invalid_typed_config() -> None:
    data = _request()
    data['suite']['loads'][0]['concurrency'] = 0
    response = _app().test_client().post(
        '/api/v1/perf/invoke',
        json=data,
        headers={'EvalScope-Task-Id': 'service-test'},
    )
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Invalid performance configuration'


def test_service_reads_suite_level_progress(monkeypatch, tmp_path) -> None:
    progress_dir = tmp_path / 'service-test' / 'perf'
    progress_dir.mkdir(parents=True)
    (progress_dir / 'progress.json').write_text('{"completed":3,"failed":1,"dropped":2}', encoding='utf-8')
    monkeypatch.setattr(perf_blueprint, 'OUTPUT_DIR', str(tmp_path))
    response = _app().test_client().get('/api/v1/perf/progress?task_id=service-test')
    assert response.status_code == 200
    assert response.get_json() == {'completed': 3, 'failed': 1, 'dropped': 2}
