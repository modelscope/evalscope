from tabulate import tabulate

from evalscope.perf.domain.result import PerfSuiteResult
from evalscope.utils.logger import get_logger

logger = get_logger()


def print_suite(result: PerfSuiteResult) -> None:
    """Render one typed suite result to the console."""
    rows = []
    for run in result.runs:
        load = run.run_spec.load
        load_value = getattr(load, 'concurrency', getattr(load, 'request_rate', '-'))
        rows.append({
            'Load': run.run_spec.load_id,
            'Value': load_value,
            'Total': run.summary.total,
            'Succeeded': run.summary.succeeded,
            'Failed': run.summary.failed,
            'Dropped': run.summary.dropped,
            'RPS': f'{run.summary.request_throughput:.2f}',
            'Avg latency (s)': f'{run.summary.averages.get("latency", 0):.3f}',
            'Success': f'{run.summary.success_rate:.1f}%',
        })
    logger.info('\nPerformance suite summary:\n' + tabulate(rows, headers='keys', tablefmt='simple_grid'))
