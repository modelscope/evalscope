# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contract tests for JSON responses consumed by the bundled Web UI."""
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from pydantic import ValidationError

flask = pytest.importorskip('flask')

from evalscope.api.metric import JudgeSummary
from evalscope.constants import ScoreStatus
from evalscope.report import Report
from evalscope.service.api_models import (
    AnalysisResponse,
    BenchmarksResponse,
    ConfigResponse,
    DataFrameResponse,
    DeletePerfRunResponse,
    DeleteReportResponse,
    EvalInvokeResponse,
    ListPerfRunsResponse,
    ListReportsResponse,
    LoadReportResponse,
    LogResponse,
    PerfDetailResponse,
    PerfRequestsResponse,
    PerfRunsListResponse,
    PredictionsResponse,
    ProgressResponse,
    TaskStatusResponse,
    WebApiContracts,
)
from evalscope.service.responses import json_response

ROOT = Path(__file__).parents[2]

# Every generic JSON transport in the SPA must name a generated response model,
# and every corresponding Flask success path must validate that model.
CONTRACT_REGISTRY = (
    ('ConfigResponse', 'evalscope/web/src/contexts/ReportsContext.tsx', 'evalscope/service/app.py'),
    ('BenchmarksResponse', 'evalscope/web/src/api/eval.ts', 'evalscope/service/blueprints/eval.py'),
    ('EvalInvokeResponse', 'evalscope/web/src/api/task.ts', 'evalscope/service/blueprints/eval.py'),
    ('ProgressResponse', 'evalscope/web/src/api/task.ts', 'evalscope/service/blueprints/eval.py'),
    ('LogResponse', 'evalscope/web/src/api/task.ts', 'evalscope/service/blueprints/eval.py'),
    ('TaskStatusResponse', 'evalscope/web/src/api/task.ts', 'evalscope/service/blueprints/eval.py'),
    ('EvalInvokeResponse', 'evalscope/web/src/api/task.ts', 'evalscope/service/blueprints/perf.py'),
    ('ProgressResponse', 'evalscope/web/src/api/task.ts', 'evalscope/service/blueprints/perf.py'),
    ('LogResponse', 'evalscope/web/src/api/task.ts', 'evalscope/service/blueprints/perf.py'),
    ('TaskStatusResponse', 'evalscope/web/src/api/task.ts', 'evalscope/service/blueprints/perf.py'),
    ('ListReportsResponse', 'evalscope/web/src/api/reports.ts', 'evalscope/service/blueprints/reports.py'),
    ('DeleteReportResponse', 'evalscope/web/src/api/reports.ts', 'evalscope/service/blueprints/reports.py'),
    ('LoadReportResponse', 'evalscope/web/src/api/reports.ts', 'evalscope/service/blueprints/reports.py'),
    ('DataFrameResponse', 'evalscope/web/src/api/reports.ts', 'evalscope/service/blueprints/reports.py'),
    ('PredictionsResponse', 'evalscope/web/src/api/reports.ts', 'evalscope/service/blueprints/reports.py'),
    ('AnalysisResponse', 'evalscope/web/src/api/reports.ts', 'evalscope/service/blueprints/reports.py'),
    ('ListPerfRunsResponse', 'evalscope/web/src/api/perf.ts', 'evalscope/service/blueprints/perf.py'),
    ('DeletePerfRunResponse', 'evalscope/web/src/api/perf.ts', 'evalscope/service/blueprints/perf.py'),
    ('PerfDetailResponse', 'evalscope/web/src/api/perf.ts', 'evalscope/service/blueprints/perf.py'),
    ('PerfRunsListResponse', 'evalscope/web/src/api/perf.ts', 'evalscope/service/blueprints/perf.py'),
    ('PerfRequestsResponse', 'evalscope/web/src/api/perf.ts', 'evalscope/service/blueprints/perf.py'),
)

NON_JSON_ENDPOINTS = {
    'evalscope/service/blueprints/eval.py': ("@bp_eval.route('/report'", 'send_file(report_file'),
    'evalscope/service/blueprints/perf.py': (
        "@bp_perf.route('/report'",
        "@bp_perf.route('/chart'",
        "@bp_perf.route('/compare/chart'",
        "@bp_perf.route('/history/report'",
    ),
    'evalscope/service/blueprints/reports.py': (
        "@bp_reports.route('/media/file'",
        "@bp_reports.route('/charts/<chart_type>'",
        "@bp_reports.route('/runs/<run_id>/models/<model_id>/charts/<chart_type>'",
        "@bp_reports.route('/runs/<run_id>/html'",
    ),
}

RESPONSE_MODELS = (
    ConfigResponse,
    BenchmarksResponse,
    EvalInvokeResponse,
    ProgressResponse,
    LogResponse,
    TaskStatusResponse,
    ListReportsResponse,
    DeleteReportResponse,
    LoadReportResponse,
    DataFrameResponse,
    PredictionsResponse,
    AnalysisResponse,
    ListPerfRunsResponse,
    DeletePerfRunResponse,
    PerfDetailResponse,
    PerfRunsListResponse,
    PerfRequestsResponse,
)


@pytest.fixture
def client(tmp_path):
    from evalscope.service.app import create_app

    app = create_app(outputs=str(tmp_path))
    app.config['TESTING'] = True
    return app.test_client()


def test_config_and_idle_task_endpoints_use_response_contracts(client, tmp_path) -> None:
    config = client.get('/api/v1/config')
    assert config.status_code == 200
    assert config.get_json() == {'outputs_root': str(tmp_path)}

    for scope in ('eval', 'perf'):
        progress = client.get(f'/api/v1/{scope}/progress', query_string={'task_id': 'missing-task'})
        assert progress.status_code == 200
        assert progress.get_json() == {'percent': 0.0}

        module = f'evalscope.service.blueprints.{scope}'
        with mock.patch(f'{module}.get_log_content', return_value={
            'text': 'line',
            'head_line': 1,
            'tail_line': 1,
            'total_lines': 1,
        }):
            log = client.get(f'/api/v1/{scope}/log', query_string={'task_id': 'task-1'})
        assert log.status_code == 200
        assert log.get_json()['text'] == 'line'

        with mock.patch(f'{module}.stop_process', return_value=True):
            stopped = client.post(f'/api/v1/{scope}/stop', query_string={'task_id': 'task-1'}, json={})
        assert stopped.status_code == 200
        assert stopped.get_json() == {'status': 'stopped', 'task_id': 'task-1'}


def test_eval_and_perf_invoke_success_responses_are_validated(client) -> None:
    task_config = SimpleNamespace(model='model', datasets=['demo'], work_dir='')
    with mock.patch('evalscope.service.blueprints.eval._build_task_config', return_value=task_config), \
            mock.patch('evalscope.service.blueprints.eval.create_log_file'), \
            mock.patch('evalscope.service.blueprints.eval.run_in_subprocess', return_value={'demo': {'score': 1}}), \
            mock.patch('evalscope.service.blueprints.eval._build_result_table', return_value='table'):
        response = client.post(
            '/api/v1/eval/invoke',
            headers={'EvalScope-Task-Id': 'task-1'},
            json={'model': 'model', 'datasets': ['demo'], 'api_url': 'http://example.test'},
        )
    assert response.status_code == 200
    assert response.get_json()['status'] == 'completed'

    perf_args = SimpleNamespace(model='model', url='http://example.test', api='openai')
    with mock.patch('evalscope.service.blueprints.perf.PerfArguments.from_dict', return_value=perf_args), \
            mock.patch('evalscope.service.blueprints.perf.create_log_file'), \
            mock.patch('evalscope.service.blueprints.perf.run_in_subprocess', return_value={'rps': 1}), \
            mock.patch('evalscope.service.blueprints.perf._build_perf_table', return_value='table'):
        response = client.post(
            '/api/v1/perf/invoke',
            headers={'EvalScope-Task-Id': 'task-2'},
            json={'model': 'model', 'url': 'http://example.test'},
        )
    assert response.status_code == 200
    assert response.get_json()['status'] == 'completed'


def test_benchmark_route_accepts_missing_translation(client) -> None:
    entry = {
        'name': 'demo',
        'pretty_name': 'Demo',
        'tags': [],
        'category': 'llm',
        'subset_list': [],
        'total_samples': 0,
        'few_shot_num': 0,
        'dataset_id': '',
        'metrics': [],
        'meta': {'nullable': None},
        'description': {'en': {'full': '', 'sections': {}}},
    }
    with mock.patch('evalscope.service.blueprints.eval.build_benchmark_entry', return_value=entry):
        client.application.config['SUPPORTED_BENCHMARKS'] = {'text': ['demo'], 'multimodal': []}
        response = client.get('/api/v1/eval/benchmarks')
    assert response.status_code == 200
    assert response.get_json()['text'][0]['description']['en']['full'] == ''


def test_report_table_and_analysis_routes_validate_dynamic_data(client) -> None:
    import pandas as pd

    report = Report(analysis='analysis')
    bundle = ([report], [report.dataset_name], {})
    with mock.patch('evalscope.service.blueprints.reports.load_report_bundle', return_value=bundle), \
            mock.patch(
                'evalscope.service.blueprints.reports.get_acc_report_df',
                return_value=pd.DataFrame([{'score': float('nan'), 'metadata': {'nullable': None}}]),
            ):
        table = client.get('/api/v1/reports/runs/run/models/model/table')
    assert table.status_code == 200
    assert table.get_json()['data'] == [{'metadata': {'nullable': None}, 'score': None}]

    with mock.patch('evalscope.service.blueprints.reports.load_report_bundle', return_value=bundle):
        analysis = client.get(
            '/api/v1/reports/runs/run/models/model/analysis',
            query_string={'dataset_name': report.dataset_name},
        )
    assert analysis.status_code == 200
    assert analysis.get_json() == {'analysis': 'analysis'}


def test_json_response_validates_and_serializes_aliases() -> None:
    app = flask.Flask(__name__)
    payload = {
        'predictions': [{
            'Index': '0',
            'Input': 'question',
            'Metadata': {'nullable': None},
            'Generated': 'answer',
            'Gold': 'answer',
            'Pred': '*Same as Generated*',
            'Score': {'value': {'accuracy': 1.0}},
            'NScore': None,
        }]
    }

    with app.app_context():
        response = json_response(PredictionsResponse, payload, status=201)

    assert response.status_code == 201
    body = response.get_json()
    assert body['predictions'][0]['Index'] == '0'
    assert body['predictions'][0]['NScore'] is None
    assert body['predictions'][0]['Metadata']['nullable'] is None


def test_json_response_rejects_invalid_success_shape() -> None:
    app = flask.Flask(__name__)
    with app.app_context(), pytest.raises(ValidationError):
        json_response(ConfigResponse, {'outputs_root': './outputs', 'unexpected': True})


def test_real_single_sample_report_fixture_matches_contract() -> None:
    fixture = ROOT / 'evalscope/web/src/test/fixtures/report-real-single-sample.json'
    response = LoadReportResponse.model_validate(json.loads(fixture.read_text(encoding='utf-8')))

    report = response.report_list[0]
    assert report.perf_metrics is not None
    assert report.perf_metrics.summary.latency.std is None
    assert report.perf_metrics.summary.usage.total_tokens_count == 114


def test_report_perf_contract_supports_coverage_and_incomplete_runs(client) -> None:
    fixture = ROOT / 'evalscope/web/src/test/fixtures/report-real-single-sample.json'
    payload = json.loads(fixture.read_text(encoding='utf-8'))
    payload['report_list'][0]['perf_metrics']['coverage'] = {
        'requests_with_metrics': 1,
        'total_requests': 1,
    }
    response = LoadReportResponse.model_validate(payload)
    assert response.report_list[0].perf_metrics.coverage.requests_with_metrics == 1

    report = Report(perf_metrics={'coverage': {'requests_with_metrics': 0, 'total_requests': 3}})
    with mock.patch(
        'evalscope.service.blueprints.reports.load_report_bundle',
        return_value=([report], [report.dataset_name], {}),
    ):
        route_response = client.get('/api/v1/reports/runs/run/models/model')

    assert route_response.status_code == 200
    assert route_response.get_json()['report_list'][0]['perf_metrics'] == {
        'coverage': {'requests_with_metrics': 0, 'total_requests': 3},
    }


def test_regular_and_judge_reports_preserve_nullable_contract() -> None:
    regular = Report().to_dict()
    regular_model = LoadReportResponse.model_validate({
        'report_list': [regular],
        'datasets': [regular['dataset_name']],
        'task_config': {},
    })
    assert regular_model.report_list[0].judge_summary is None

    judged = Report(
        judge_summary=JudgeSummary(
            status=ScoreStatus.SUCCESS,
            scored=1,
            total=1,
            coverage=1.0,
            judge_models=['judge'],
            valid_observations=1,
            total_observations=1,
        )
    ).to_dict()
    judged_model = LoadReportResponse.model_validate({
        'report_list': [judged],
        'datasets': [judged['dataset_name']],
        'task_config': {},
    })
    assert judged_model.report_list[0].judge_summary is not None
    assert judged_model.report_list[0].judge_summary.error is None


def test_v1_report_migrates_before_response_validation() -> None:
    report = Report.from_dict({
        'dataset_name': 'conll2003',
        'score': 0.8,
        'metrics': [{
            'name': 'mean_f1_score',
            'score': 0.8,
            'categories': [],
        }],
    })

    response = LoadReportResponse.model_validate({
        'report_list': [report.to_dict()],
        'datasets': ['conll2003'],
        'task_config': {'metadata': {'nullable': None}},
    })

    assert response.report_list[0].schema_version == 2
    assert response.report_list[0].metrics[0].identity.name == 'f1'
    assert response.report_list[0].judge_summary is None


def test_prediction_contract_supports_messages_trace_and_missing_optional_fields() -> None:
    response = PredictionsResponse.model_validate({
        'predictions': [{
            'Index': 'sample-1',
            'Input': 'question',
            'Metadata': {},
            'Generated': 'answer',
            'Gold': 'answer',
            'Pred': '*Same as Generated*',
            'Score': {
                'value': {},
                'status': 'excluded',
                'judge_summary': {
                    'status': 'excluded',
                    'scored': 0,
                    'total': 1,
                    'coverage': 0.0,
                    'error': None,
                },
                'metadata': {
                    'judge_attempts': [{
                        'status': 'parse_error',
                        'case_id': 'case-1',
                        'judge_id': 'judge-1',
                    }],
                    'judge_skipped': True,
                },
            },
            'NScore': None,
            'Messages': [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'question'}]},
                {'role': 'assistant', 'content': 'answer'},
            ],
            'AgentTrace': {
                'max_steps': 1,
                'events': [{
                    'step': 0,
                    'timestamp': 1.0,
                    'type': 'model_generate',
                    'payload': {'nullable': None},
                }],
            },
        }]
    })

    row = response.predictions[0]
    assert row.normalized_score is None
    assert row.messages[1].perf_metrics is None
    assert row.agent_trace.events[0].payload['nullable'] is None


def test_dynamic_dataframe_nan_serializes_as_json_null() -> None:
    app = flask.Flask(__name__)
    with app.app_context():
        response = json_response(DataFrameResponse, {'columns': ['score'], 'data': [{'score': float('nan')}]})
    assert response.get_json() == {'columns': ['score'], 'data': [{'score': None}]}


def test_benchmark_contract_allows_missing_translation() -> None:
    response = BenchmarksResponse.model_validate({
        'text': [{
            'name': 'demo',
            'pretty_name': 'Demo',
            'tags': [],
            'category': 'llm',
            'subset_list': [],
            'total_samples': 0,
            'few_shot_num': 0,
            'dataset_id': '',
            'metrics': [],
            'meta': {'nullable': None},
            'description': {'en': {'full': '', 'sections': {}}},
        }]
    })
    assert response.text[0].description.zh is None


def test_frontend_json_endpoints_are_registered_with_generated_models() -> None:
    definitions = WebApiContracts.model_json_schema(mode='serialization', by_alias=True)['$defs']
    for model in RESPONSE_MODELS:
        assert model.__name__ in definitions

    frontend_models = set()
    for _, frontend_path, _ in CONTRACT_REGISTRY:
        frontend = (ROOT / frontend_path).read_text(encoding='utf-8')
        frontend_models.update(re.findall(r'api(?:Post|Delete)?Validated<(\w+)>', frontend))

    registered_models = {model_name for model_name, _, _ in CONTRACT_REGISTRY}
    assert frontend_models == registered_models

    for model_name, frontend_path, backend_path in CONTRACT_REGISTRY:
        frontend = (ROOT / frontend_path).read_text(encoding='utf-8')
        backend = (ROOT / backend_path).read_text(encoding='utf-8')
        assert f'<{model_name}>' in frontend
        assert re.search(rf'json_response\(\s*{model_name}\b', backend)


def test_non_json_endpoints_are_explicitly_excluded() -> None:
    for source_path, route_markers in NON_JSON_ENDPOINTS.items():
        source = (ROOT / source_path).read_text(encoding='utf-8')
        for marker in route_markers:
            assert marker in source
