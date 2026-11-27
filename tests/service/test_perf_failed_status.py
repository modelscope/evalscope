from unittest import mock

import pytest

pytest.importorskip('flask')

from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.service.app import create_app


def _run(total: int, succeeded: int) -> dict:
    metrics = {'Total Requests': total, 'Success Requests': succeeded, 'Failed Requests': total - succeeded}
    return {'metrics': metrics, 'percentiles': {}}


@pytest.mark.parametrize(
    'result, status',
    [
        ({'parallel_1_number_4': _run(4, 0), 'parallel_2_number_4': _run(4, 0)}, 'error'),
        ({'parallel_1_number_4': _run(4, 0), 'parallel_2_number_4': _run(4, 1)}, 'completed'),
    ],
)
def test_invoke_status_reflects_request_outcomes(tmp_path, result, status) -> None:
    app = create_app(outputs=str(tmp_path))
    app.config['TESTING'] = True
    perf_args = PerfArguments(model='mock', url='http://example.test', api='openai')
    module = 'evalscope.service.blueprints.perf'
    with mock.patch(f'{module}.PerfArguments.from_dict', return_value=perf_args), \
            mock.patch(f'{module}.create_log_file'), \
            mock.patch(f'{module}.run_in_subprocess', return_value=result), \
            mock.patch(f'{module}._build_perf_table', return_value=''), \
            mock.patch(f'{module}.finish_task') as finish:
        response = app.test_client().post(
            '/api/v1/perf/invoke',
            headers={'EvalScope-Task-Id': 'task-1'},
            json={'model': 'mock', 'url': 'http://example.test'},
        )
    assert response.get_json()['status'] == status
    finish.assert_called_once_with('task-1', status)
