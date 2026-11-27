# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end runs of the example plugin in ``examples/plugins`` against its local toy endpoint."""

import json
import threading
from pathlib import Path
from typing import Iterator

import pytest

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.utils.io_utils import yaml_to_dict

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / 'examples' / 'plugins' / 'evalscope_example_plugin'
PLUGIN = 'example-under-test'


@pytest.fixture
def example_plugin(fake_plugins, monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the example importable from the source tree and declare it as an installed plugin."""
    monkeypatch.syspath_prepend(str(EXAMPLE_ROOT))
    fake_plugins.declare(PLUGIN, 'evalscope_example_plugin', distribution='evalscope-example-plugin', version='0.1.0')
    for name in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        monkeypatch.delenv(name, raising=False)
    return PLUGIN


@pytest.fixture
def arith_endpoint(example_plugin: str) -> Iterator[str]:
    import importlib.util
    from http.server import ThreadingHTTPServer

    # Load the server file on its own: importing it through the package would run the plugin's
    # registrations before ``load_plugins`` does, hiding them from the run record.
    spec = importlib.util.spec_from_file_location('_example_arith_server', EXAMPLE_ROOT / 'evalscope_example_plugin' / 'server.py')
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)

    server = ThreadingHTTPServer(('127.0.0.1', 0), server_module.ArithHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}/v1'
    finally:
        server.shutdown()
        server.server_close()


def test_example_plugin_runs_an_evaluation(example_plugin: str, arith_endpoint: str, tmp_path: Path) -> None:
    from evalscope.run import run_task

    result = run_task(
        TaskConfig(
            plugins=[example_plugin],
            eval_type='example_arith_api',
            model='arith-demo',
            api_url=arith_endpoint,
            datasets=['example_arith'],
            limit=5,
            work_dir=str(tmp_path),
            no_timestamp=True,
        )
    )

    report = result['example_arith']
    assert report.metrics[0].name.startswith('example_numeric_match')
    assert report.metrics[0].score == 1.0
    predictions = (tmp_path / 'predictions' / 'arith-demo' / 'example_arith_default.jsonl').read_text().splitlines()
    first = json.loads(predictions[0])
    assert json.loads(first['model_output']['metadata']['raw_response'])['text'].startswith('The answer is')
    snapshot = yaml_to_dict(str(tmp_path / 'configs' / 'task_config.yaml'))
    assert snapshot['plugin_packages'][0]['distribution'] == 'evalscope-example-plugin'
    assert snapshot['plugin_packages'][0]['version'] == '0.1.0'


def test_example_plugin_runs_a_perf_benchmark(example_plugin: str, arith_endpoint: str, tmp_path: Path) -> None:
    from evalscope.perf.main import run_perf_benchmark

    results = run_perf_benchmark(
        PerfArguments(
            plugins=[example_plugin],
            api='example_arith',
            dataset='example_arith_prompts',
            model='arith-demo',
            url=f'{arith_endpoint}/arith',
            parallel=[1],
            number=[4],
            outputs_dir=str(tmp_path),
            no_test_connection=True,
        )
    )

    [run] = results.values()
    assert run['metrics'].succeed_requests == 4
    [args_file] = list(tmp_path.rglob('benchmark_args.json'))
    record = json.loads(args_file.read_text())
    assert record['plugin_packages'][0]['registrations']['perf_api'] == ['example_arith']
