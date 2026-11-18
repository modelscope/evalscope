"""Offline contracts for both harness corpora and legacy scenario compatibility."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evalscope.constants import HubType
from evalscope.perf.arguments import Arguments, add_argument
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.scenarios import AgentXHarnessScenario, AgentXScenario, parse_scenario, run_scenario_benchmark
from evalscope.perf.scenarios import agentx as mvp
from evalscope.perf.scenarios import agentx_harness as harness

_NAMES = [('agentx_harness', 'full'), ('agentx_harness_256k', '256k')]


def _args(scenario: Any = 'agentx_harness', **overrides: Any) -> Arguments:
    return Arguments(**{
        'scenario': scenario,
        'model': 'test-model',
        'tokenizer_path': '/tokenizer',
        'url': 'http://localhost:8000/proxy/v1/chat/completions',
        **overrides,
    })


def _dataset(source: str = HubType.HUGGINGFACE, verified: bool = True) -> mvp.AgentXDatasetProvenance:
    return mvp.AgentXDatasetProvenance(
        source=source, dataset_id='test-corpus', trace_file='/data/traces.jsonl', sha256='digest', verified=verified,
    )


@pytest.mark.parametrize('name, variant', _NAMES)
def test_selection_and_defaults(name: str, variant: str) -> None:
    expected = AgentXHarnessScenario(name=name)
    for value in [name, {'name': name}, json.dumps({'name': name}), expected]:
        args = _args(value)
        assert isinstance(args.scenario, AgentXHarnessScenario)
        assert args.scenario == expected
        assert args.scenario.variant == variant
        assert args.scenario.tokenizer_trust_remote_code is False
        assert args.scenario.effective_trace_limit is None
        assert harness._resolve_duration(args, args.scenario) == 3600
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = Arguments.from_args(parser.parse_args([
        '--scenario', name, '--model', 'test-model', '--tokenizer-path', '/tokenizer',
        '--url', 'http://localhost:8000/v1/chat/completions', '--parallel', '1', '2',
    ]))
    assert args.scenario == expected
    assert args.parallel == [1, 2]


def test_installed_runner_isolation() -> None:
    if importlib.util.find_spec('agentx_cli') is not None:
        harness._require_harness()
    elif importlib.util.find_spec('aiperf') is not None:
        mvp._require_aiperf()
    else:
        pytest.skip('Install an isolated AgentX runner to verify its distribution')


def test_legacy_defaults_and_imports_are_preserved() -> None:
    from evalscope.perf.scenarios import run_agentx_benchmark

    for value in ['agentx', {}, '{"mode":"smoke"}', AgentXScenario()]:
        args = _args(value)
        assert type(args.scenario) is AgentXScenario
        assert args.scenario.variant == '256k'
    assert parse_scenario('agentx') == mvp.parse_agentx_scenario('agentx')
    assert run_agentx_benchmark is mvp.run_agentx_benchmark
    assert Arguments(model='test-model').scenario is None
    assert 'discriminator' in json.dumps(Arguments.model_json_schema())


@pytest.mark.parametrize('scenario', [
    'agentx-v1', 'agentx-256k', 'unknown', '[]', 'null', 42,
    {'name': 'agentx_harness', 'variant': '256k'},
    {'name': 'agentx_harness_256k', 'variant': 'full'},
    {'name': 'agentx_harness', 'trace_limit': 2},
    {'name': 'agentx_harness', 'unknown': True},
])
def test_invalid_scenarios_are_rejected(scenario: Any) -> None:
    with pytest.raises(ValueError):
        _args(scenario)


@pytest.mark.parametrize('options', [
    {'duration': 899}, {'duration': float('nan')}, {'duration': float('inf')}, {'number': 1},
    {'open_loop': True}, {'multi_turn': True}, {'sla_auto_tune': True}, {'rate': 2},
    {'warmup_num': 1}, {'dataset': 'random'}, {'dataset_args': {'foo': 1}},
    {'api': 'openai_responses'}, {'tokenizer_path': None}, {'parallel': 0},
    {'exclude_warmup_overlap': True},
])
def test_invalid_perf_controls(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _args(**options)


@pytest.mark.parametrize('field, value', [
    ('trace_limit', 0), ('max_context_length', -1), ('num_gpus', 0),
    ('request_timeout_seconds', 0), ('request_timeout_seconds', float('inf')),
    ('benchmark_grace_period', -1), ('benchmark_grace_period', float('nan')),
])
def test_invalid_scenario_controls(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        _args({'name': 'agentx_harness', 'mode': 'smoke', field: value})


@pytest.mark.parametrize('name, variant', _NAMES)
def test_commands_and_optional_controls(name: str, variant: str, tmp_path: Path) -> None:
    args = _args({
        'name': name, 'request_timeout_seconds': 90.5, 'benchmark_grace_period': 0,
        'tokenizer_revision': 'reviewed-revision', 'max_context_length': 1048576,
    })
    command, _ = harness._build_command(args, args.scenario, _dataset(), 8, 900, 42, tmp_path)
    assert command[:4] == [sys.executable, '-m', 'agentx_cli', 'profile']
    expected = {
        '--scenario': 'agentx', '--url': 'http://localhost:8000',
        '--custom-endpoint': '/proxy/v1/chat/completions', '--endpoint-type': 'chat',
        '--concurrency': '8', '--benchmark-duration': '900', '--random-seed': '42',
        '--extra-inputs': 'ignore_eos:true', '--cache-bust': 'first_turn_prefix',
        '--system-idle-gap-cap-seconds': '10', '--trajectory-start-min-ratio': '0.25',
        '--trajectory-start-max-ratio': '0.75', '--public-dataset': mvp._DATASETS[variant]['public_dataset'],
        '--benchmark-grace-period': '0.0', '--request-timeout-seconds': '90.5',
        '--tokenizer-revision': 'reviewed-revision', '--max-context-length': '1048576',
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value, flag
    assert '--streaming' in command and '--use-server-token-count' in command
    assert '--unsafe-override' not in command
    assert '--tokenizer-trust-remote-code' not in command
    assert '--warmup-request-count' not in command
    assert '--num-dataset-entries' not in command
    # Exercise the installed CLI parser when testing the real optional dependency.
    if importlib.util.find_spec('agentx_cli') is not None:
        from agentx_cli import app

        app.parse_args(command[3:])


def test_smoke_local_loader_and_secret_redaction(tmp_path: Path) -> None:
    args = _args(
        {'name': 'agentx_harness_256k', 'mode': 'smoke', 'tokenizer_trust_remote_code': True},
        duration=3, api_key='secret-key', headers={'X-Custom': 'secret-header'},
    )
    command, _ = harness._build_command(args, args.scenario, _dataset(HubType.LOCAL), 1, 3, 42, tmp_path)
    assert harness._resolve_duration(_args(args.scenario), args.scenario) == 60
    assert '--unsafe-override' in command
    assert '--tokenizer-trust-remote-code' in command
    assert command[command.index('--hf-weka-dataset') + 1] == '/data'
    assert command[command.index('--num-dataset-entries') + 1] == '4'
    assert '--input-file' not in command
    redacted = ' '.join(harness._redact_command(command))
    assert 'secret-key' not in redacted and 'secret-header' not in redacted


@pytest.mark.parametrize('url', [
    'file:///v1/chat/completions', 'http://u:p@localhost/v1/chat/completions',
    'http://localhost/v1/chat/completions?key=secret', 'http://localhost/v1/chat/completions#fragment',
    'http://localhost/v1/completions',
])
def test_invalid_url_rejected_before_running(url: str) -> None:
    with pytest.raises(ValueError):
        _args(url=url)


@pytest.mark.parametrize('name, package, version', [
    ('agentx', 'aiperf', '0.12.0'), ('agentx_harness', 'agentx-harness', '1.0.13'),
])
@pytest.mark.parametrize('state', ['missing', 'wrong', 'mixed', 'correct'])
def test_dependency_isolation(
    name: str, package: str, version: str, state: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {} if state == 'missing' else {package: 'wrong' if state == 'wrong' else version}
    if state == 'mixed':
        versions = {'aiperf': '0.12.0', 'agentx-harness': '1.0.13'}

    def installed(distribution: str) -> str:
        if distribution not in versions:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    monkeypatch.setattr(importlib.metadata, 'version', installed)
    require = mvp._require_aiperf if name == 'agentx' else harness._require_harness
    if state == 'correct':
        require()
        return
    with pytest.raises(RuntimeError):
        run_scenario_benchmark(_args(name), '/unused')


@pytest.mark.parametrize('name, variant', _NAMES)
def test_default_dataset_identity(
    name: str, variant: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = []
    monkeypatch.setattr(harness, '_verify_huggingface_revision', verified.append)
    args = _args(name)
    dataset = harness._resolve_dataset(args, args.scenario)
    assert dataset.source == HubType.HUGGINGFACE
    assert dataset.dataset_id == mvp._DATASETS[variant]['huggingface_id']
    assert verified == [mvp._DATASETS[variant]]


def test_public_alias_verifies_actual_head(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    spec = mvp._DATASETS['full']
    info = SimpleNamespace(sha=spec['huggingface_revision'], siblings=[
        SimpleNamespace(rfilename='traces.jsonl', lfs=SimpleNamespace(sha256=spec['sha256'])),
    ])
    calls = []

    def dataset_info(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return info

    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(dataset_info=dataset_info))
    harness._verify_huggingface_revision(spec)
    assert calls[0]['revision'] == 'main'
    info.sha = 'moved-head'
    with pytest.raises(RuntimeError, match='revision changed'):
        harness._verify_huggingface_revision(spec)
    info.sha = spec['huggingface_revision']
    info.siblings[0].lfs.sha256 = 'wrong-hash'
    with pytest.raises(RuntimeError, match='hash changed'):
        harness._verify_huggingface_revision(spec)


def test_local_dataset_hash_and_no_mvp_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / 'traces.jsonl').write_text('{}\n', encoding='utf-8')
    calls = []

    def download(*args: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(harness, 'download_dataset_snapshot', download)
    args = _args({'name': 'agentx_harness', 'mode': 'smoke'}, dataset_path=str(tmp_path))
    dataset = harness._resolve_dataset(args, args.scenario)
    assert dataset.verified is False and dataset.trace_dir == ''
    args = _args(dataset_path=str(tmp_path))
    with pytest.raises(ValueError, match='hash mismatch'):
        harness._resolve_dataset(args, args.scenario)
    args = _args({'name': 'agentx_harness', 'mode': 'smoke'}, data_source='modelscope')
    harness._resolve_dataset(args, args.scenario)
    assert calls[-1]['revision'] == mvp._DATASETS['full']['modelscope_revision']
    assert calls[-1]['allow_file_pattern'] == 'traces.jsonl'


@pytest.mark.parametrize('mode, status, native_valid, verified', [
    ('benchmark', 'completed', True, True), ('benchmark', 'completed', False, True),
    ('benchmark', 'completed', True, False), ('smoke', 'completed', True, True),
    ('benchmark', 'failed', True, True), ('benchmark', 'cancelled', True, True),
])
def test_native_submission_validity(
    mode: str, status: str, native_valid: bool, verified: bool, tmp_path: Path,
) -> None:
    result = harness._normalize_summary(
        {'metadata': {'submission_valid': native_valid, 'submission_invalid_reasons': ['unsafe_override']},
         'output_token_throughput': {'unit': 'tokens/sec', 'avg': 80}},
        status, AgentXHarnessScenario(mode=mode, num_gpus=8), _dataset(HubType.LOCAL, verified),
        8, 900, 42, tmp_path, None, [],
    )
    assert result.submission_valid is (native_valid and verified and mode == 'benchmark' and status == 'completed')
    assert result.mirror_revalidated is False
    assert result.aiperf_submission_valid is native_valid
    assert result.derived_metrics['output_token_throughput_per_gpu'].avg == 10
    assert result.schema_version == '2.0' and result.agentx_harness_version == '1.0.13'


@pytest.mark.parametrize('name, variant', _NAMES)
def test_entrypoint_sweep_with_real_subprocess(
    name: str, variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = []
    real_build = harness._build_command
    monkeypatch.setattr(harness, '_require_harness', lambda: None)
    monkeypatch.setattr(harness, '_resolve_dataset', lambda *args: _dataset())

    def stub_build(*args: Any) -> tuple[list[str], dict[str, str]]:
        command, env = real_build(*args)
        built.append(command)
        path = args[-1] / harness._AIPERF_SUMMARY
        data = {'metadata': {'submission_valid': True}, 'request_count': {'unit': 'requests', 'avg': 5}}
        code = f'import pathlib; pathlib.Path({str(path)!r}).write_text({json.dumps(data)!r})'
        return [sys.executable, '-c', code], env

    monkeypatch.setattr(harness, '_build_command', stub_build)
    args = _args(name, parallel=[1, 2], sleep_interval=0, outputs_dir=str(tmp_path), no_timestamp=True)
    results = run_perf_benchmark(args)
    assert list(results) == ['parallel_1', 'parallel_2']
    for result in results.values():
        assert result['status'] == 'completed' and result['submission_valid'] is True
        assert result['seed'] == 42 and result['duration'] == 3600
        assert result['scenario']['variant'] == variant
        assert Path(result['raw_summary_file']).is_file()
    assert len(built) == 2
    sweep = Path(results['parallel_1']['raw_artifact_dir']).parent.parent / 'agentx_sweep_summary.json'
    assert json.loads(sweep.read_text()) == results
    with pytest.raises(FileExistsError):
        harness.run_agentx_harness_benchmark(args, str(sweep.parent.parent))


@pytest.mark.parametrize('code', [
    'pass', 'import sys; sys.exit(7)', "p.write_text('not-json')", "p.write_text('[]')",
    "p.write_text('{\"metadata\": [1]}')", "p.write_text('{\"request_count\": {\"unit\": \"requests\", \"avg\": []}}')",
])
def test_missing_and_malformed_exports_fail(code: str, tmp_path: Path) -> None:
    artifacts = tmp_path / 'harness'
    artifacts.mkdir()
    preamble = f'import pathlib; p=pathlib.Path({str(artifacts / harness._AIPERF_SUMMARY)!r}); '
    result = harness._run_one_agentx(
        [sys.executable, '-c', preamble + code], os.environ.copy(), AgentXHarnessScenario(),
        _dataset(), 1, 900, 42, tmp_path, artifacts,
    )
    assert result.status == 'failed' and result.submission_valid is False
    assert result.error_summary


def test_spawn_failure_writes_failed_summary(tmp_path: Path) -> None:
    result = harness._run_one_agentx(
        ['/nonexistent-agentx-executable'], {}, AgentXHarnessScenario(), _dataset(),
        1, 900, 42, tmp_path, tmp_path / 'harness',
    )
    assert result.status == 'failed' and result.error_summary


@pytest.mark.skipif(os.name != 'posix', reason='Harness uses POSIX process groups')
@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_interrupt_cleans_process_group(sig: int, tmp_path: Path) -> None:
    previous_handler = signal.getsignal(signal.SIGTERM)
    child_pid = tmp_path / 'child.pid'
    worker_code = (
        'import os,signal,time; '
        'signal.signal(signal.SIGINT,signal.SIG_IGN); '
        'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        f'open({str(child_pid)!r},"w").write(str(os.getpid())); '
        'time.sleep(60)'
    )
    code = (
        'import os,signal,subprocess,sys,time,pathlib; '
        f'subprocess.Popen([sys.executable,"-c",{worker_code!r}]); '
        f'path=pathlib.Path({str(child_pid)!r}); '
        '\nwhile not path.exists() or not path.read_text(): time.sleep(0.01)\n'
        f'os.kill(os.getppid(), {int(sig)}); time.sleep(60)'
    )
    result = harness._run_one_agentx(
        [sys.executable, '-c', code], os.environ.copy(), AgentXHarnessScenario(), _dataset(),
        1, 900, 42, tmp_path, tmp_path / 'harness',
    )
    assert result.status == 'cancelled' and result.submission_valid is False
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    # A surviving worker would still be sleeping, not a reaped or zombie process.
    pid = int(child_pid.read_text())
    stat = Path(f'/proc/{pid}/stat')
    for _ in range(100):
        try:
            state = stat.read_text().split()[2]
        except FileNotFoundError:
            break
        if state == 'Z':
            break
        time.sleep(0.01)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail('Harness worker survived process-group shutdown')


@pytest.mark.skipif(importlib.util.find_spec('agentx_cli') is None, reason='Install the isolated agentx_harness extra')
@pytest.mark.parametrize('name, variant', _NAMES)
def test_native_harness_with_local_streaming_server(
    name: str, variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('AIPERF_DATASET_CONFIGURATION_TIMEOUT', '30')
    monkeypatch.setenv('AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT', '30')
    requests = []

    class MockOpenAI(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            payload = json.dumps({'object': 'list', 'data': [{'id': 'mock-model', 'object': 'model'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, body))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            base = {
                'id': 'mock-completion', 'object': 'chat.completion.chunk',
                'created': int(time.time()), 'model': 'mock-model',
            }
            chunks = [
                {**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'ok '},
                                     'finish_reason': None}]},
                {**base, 'choices': [{'index': 0, 'delta': {'content': 'done'}, 'finish_reason': None}]},
                {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}],
                 'usage': {'prompt_tokens': 64, 'completion_tokens': 2, 'total_tokens': 66}},
            ]
            try:
                for chunk in chunks:
                    self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
                    self.wfile.flush()
                    time.sleep(0.005)
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    tokenizer_dir = tmp_path / 'tokenizer'
    tokenizer = Tokenizer(models.WordLevel({
        '[UNK]': 0, 'ok': 1, 'done': 2, **{f'word{i}': i + 3 for i in range(1024)},
    }, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token='[UNK]',
        chat_template='{% for message in messages %}{{ message["content"] }} {% endfor %}',
    ).save_pretrained(tokenizer_dir)
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    traces = [
        {'id': f'fixture-{i}', 'models': ['mock-model'], 'block_size': 64, 'hash_id_scope': 'local', 'requests': [
            {'type': 'n', 't': 0.0, 'model': 'mock-model', 'in': 64, 'out': 2, 'hash_ids': [i * 10 + 1]},
            {'type': 'n', 't': 0.05, 'model': 'mock-model', 'in': 128, 'out': 2,
             'hash_ids': [i * 10 + 1, i * 10 + 2]},
        ]}
        for i in range(4)
    ]
    (dataset / 'traces.jsonl').write_text(''.join(json.dumps(trace) + '\n' for trace in traces), encoding='utf-8')
    with ThreadingHTTPServer(('127.0.0.1', 0), MockOpenAI) as server:
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            results = run_perf_benchmark(Arguments(
                scenario={'name': name, 'mode': 'smoke', 'benchmark_grace_period': 2},
                model='mock-model', tokenizer_path=str(tokenizer_dir), dataset_path=str(dataset),
                url=f'http://127.0.0.1:{server.server_port}/proxy/v1/chat/completions',
                parallel=[1], duration=3, outputs_dir=str(tmp_path / 'results'), no_timestamp=True,
            ))
        finally:
            server.shutdown()
            thread.join(timeout=5)
    summary = results['parallel_1']
    assert summary['status'] == 'completed' and summary['submission_valid'] is False
    assert summary['scenario']['variant'] == variant
    assert summary['metrics']['request_count']['avg'] > 0
    assert requests
    assert all(path == '/proxy/v1/chat/completions' for path, _ in requests)
    assert all(body.get('stream') is True and body.get('ignore_eos') is True for _, body in requests)
