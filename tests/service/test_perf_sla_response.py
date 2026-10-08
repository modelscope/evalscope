from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip('flask')

from evalscope.perf.sla.sla_models import SLAProbe, SLASelection
from evalscope.perf.sla.sla_run import SLAResult
from evalscope.service.app import create_app


def test_sla_invoke_preserves_legacy_result_and_exact_count_table(tmp_path) -> None:
    app = create_app(outputs=str(tmp_path))
    app.config['TESTING'] = True
    result = SLAResult(
        {'parallel_4': {'metrics': {}, 'percentiles': {}}},
        probes=[
            SLAProbe(
                value=4,
                averaged_result={'metrics': {}, 'percentiles': {}},
                total_requests=24,
                succeeded_requests=23,
                success_rate=23 / 24 * 100,
                request_gate_passed=False,
                valid=False,
                reasons=['run 3: 7/8 requests succeeded'],
            )
        ],
        selections=[
            SLASelection(
                criteria={'avg_ttft': '<=50ms'},
                mode='constraint',
                selected_value=None,
                status='none',
                reason='No tested pressure satisfied the SLA',
                assumption='SLA satisfaction is monotonic as pressure increases',
            )
        ],
    )
    perf_args = SimpleNamespace(model='mock', url='http://example.test', api='openai')
    with mock.patch('evalscope.service.blueprints.perf.PerfArguments.from_dict', return_value=perf_args), \
            mock.patch('evalscope.service.blueprints.perf.create_log_file'), \
            mock.patch('evalscope.service.blueprints.perf.run_in_subprocess', return_value=result):
        response = app.test_client().post(
            '/api/v1/perf/invoke',
            headers={'EvalScope-Task-Id': 'sla-task'},
            json={'model': 'mock', 'url': 'http://example.test'},
        )
    assert response.status_code == 200
    body = response.get_json()
    assert body['result'] == {'parallel_4': {'metrics': {}, 'percentiles': {}}}
    assert '23/24' in body['table']
    assert 'best_observed' not in body['table']
    assert 'none' in body['table']
