"""Offline full-corpus harness contracts; MVP regressions remain in test_agentx.py."""

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
from evalscope.perf.scenarios import AgentXHarnessScenario, AgentXScenario
from evalscope.perf.scenarios import agentx as mvp
from evalscope.perf.scenarios import agentx_harness as harness


def _args(**options: Any) -> Arguments:
    return Arguments(**{
        'scenario': 'agentx_harness', 'model': 'mock-model', 'tokenizer_path': '/tokenizer',
        'url': 'http://localhost:8000/proxy/v1/chat/completions', **options,
    })


def _dataset() -> mvp.AgentXDatasetProvenance:
    return mvp.AgentXDatasetProvenance(
        source=HubType.LOCAL, dataset_id='fixture', trace_file='/data/traces.jsonl', sha256='digest', verified=True,
    )


def _run(tmp_path: Path, code: str, env: dict[str, str] | None = None) -> harness.AgentXHarnessRunSummary:
    artifacts = tmp_path / 'harness'
    artifacts.mkdir()
    prefix = f'import pathlib; p=pathlib.Path({str(artifacts / harness._AIPERF_SUMMARY)!r}); '
    return harness._run_one_agentx(
        [sys.executable, '-c', prefix + code], env if env is not None else os.environ.copy(), AgentXHarnessScenario(),
        _dataset(), 1, 900, 42, tmp_path, artifacts,
    )


def test_scenario_selection_and_cli() -> None:
    expected = AgentXHarnessScenario()
    for value in ['agentx_harness', {'name': 'agentx_harness'}, '{"name":"agentx_harness"}', expected]:
        assert _args(scenario=value).scenario == expected
    assert harness._resolve_duration(_args(), expected) == 3600
    assert expected.tokenizer_trust_remote_code is False
    for value in ['agentx', {}, '{"mode":"smoke"}']:
        assert isinstance(_args(scenario=value).scenario, AgentXScenario)
    parser = argparse.ArgumentParser()
    add_argument(parser)
    parsed = Arguments.from_args(parser.parse_args([
        '--scenario', 'agentx_harness', '--model', 'mock-model', '--tokenizer-path', '/tokenizer',
    ]))
    assert parsed.scenario == expected


@pytest.mark.parametrize('options', [
    {'scenario': 'agentx_harness_256k'}, {'scenario': {'name': 'agentx_harness', 'variant': '256k'}},
    {'scenario': {'name': 'agentx_harness', 'trace_limit': 2}}, {'scenario': '[]'},
    {'duration': float('nan')}, {'duration': float('inf')}, {'duration': 899}, {'number': 1},
    {'open_loop': True}, {'multi_turn': True}, {'rate': 2}, {'warmup_num': 1}, {'tokenizer_path': None},
    {'url': 'http://u:p@localhost/v1/chat/completions'}, {'url': 'http://localhost/v1/chat/completions?key=secret'},
])
def test_invalid_configuration(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _args(**options)


def test_installed_runner_isolation() -> None:
    if importlib.util.find_spec('agentx_cli') is not None:
        harness._require_harness()
    elif importlib.util.find_spec('aiperf') is not None:
        mvp._require_aiperf()
    else:
        pytest.skip('No optional runner installed')


@pytest.mark.parametrize('versions', [{}, {'aiperf': 'wrong', 'agentx-harness': 'wrong'},
                                      {'aiperf': '0.12.0', 'agentx-harness': '1.0.13'}])
def test_dependency_guards(versions: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    def installed(name: str) -> str:
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, 'version', installed)
    for runner in [mvp._require_aiperf, harness._require_harness]:
        with pytest.raises(RuntimeError):
            runner()


def test_command_contract(tmp_path: Path) -> None:
    args = _args(scenario={'name': 'agentx_harness', 'benchmark_grace_period': 0, 'request_timeout_seconds': 90.5})
    public = _dataset().model_copy(update={'source': HubType.HUGGINGFACE})
    command, _ = harness._build_command(args, args.scenario, public, 8, 900, 42, tmp_path)
    assert command[:4] == [sys.executable, '-m', 'evalscope.perf.scenarios._agentx_harness_cli', 'profile']
    for flag, value in {
        '--scenario': 'agentx', '--public-dataset': mvp._DATASETS['full']['public_dataset'],
        '--benchmark-grace-period': '0.0', '--request-timeout-seconds': '90.5',
    }.items():
        assert command[command.index(flag) + 1] == value
    assert '--unsafe-override' not in command
    assert '--tokenizer-trust-remote-code' not in command
    if importlib.util.find_spec('agentx_cli') is not None:
        from agentx_cli import app
        app.parse_args(command[3:])
    args = _args(
        scenario={'name': 'agentx_harness', 'mode': 'smoke'}, api_key='agentx-api-key-sentinel-8ce93',
        headers={'X-Custom': 'agentx-header-sentinel-7bd41'},
    )
    command, env = harness._build_command(args, args.scenario, _dataset(), 1, 60, 42, tmp_path)
    assert command[command.index('--hf-weka-dataset') + 1] == '/data'
    assert command[command.index('--num-dataset-entries') + 1] == '4'
    assert '--unsafe-override' in command and '--config' not in command
    for secret in ['agentx-api-key-sentinel-8ce93', 'agentx-header-sentinel-7bd41']:
        assert secret not in ' '.join(command)
    assert env['EVALSCOPE_AGENTX_API_KEY'] == 'agentx-api-key-sentinel-8ce93'
    assert json.loads(env['EVALSCOPE_AGENTX_HEADERS']) == {'X-Custom': 'agentx-header-sentinel-7bd41'}
    assert not list(tmp_path.iterdir())
    for secret in ['invalid\ncredential', 'invalid\rcredential']:
        with pytest.raises(ValueError, match='CR or LF'):
            args = _args(api_key=secret)
            harness._build_command(args, args.scenario, public, 1, 900, 42, tmp_path)


@pytest.mark.skipif(importlib.util.find_spec('agentx_cli') is None, reason='Requires the isolated harness extra')
@pytest.mark.parametrize('secret', ['123', 'true', 'ms', 'review-sentinel${', '${MISSING_AGENTX_TEST_ENV}', 'quote"\\\\'])
@pytest.mark.parametrize('source', [HubType.HUGGINGFACE, HubType.LOCAL])
def test_native_credentials_preserve_defaults(
    secret: str, source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentx_cli import app
    from aiperf.config.flags.resolver import resolve_config

    from evalscope.perf.scenarios import _agentx_harness_cli

    args = _args(api_key=secret, headers={'X-Custom': secret})
    dataset = _dataset().model_copy(update={'source': source})
    command, env = harness._build_command(args, args.scenario, dataset, 8, 900, 42, tmp_path)
    parse = app.parse_args
    expected = resolve_config(parse(command[3:])[1].arguments['cli_config']).model_dump()
    captured = []

    def check_config(**kwargs: Any) -> None:
        config = resolve_config(kwargs['cli_config'])
        assert config.benchmark.endpoint.api_key == secret
        assert config.benchmark.endpoint.headers['X-Custom'] == secret
        config.benchmark.endpoint.api_key = None
        config.benchmark.endpoint.headers = {}
        assert config.model_dump() == expected
        captured.append(config)

    def parse_args(argv: list[str]) -> Any:
        _, bound, unused = parse(argv)
        return check_config, bound, unused

    monkeypatch.setattr('agentx_cli.app', SimpleNamespace(parse_args=parse_args))
    monkeypatch.setenv('EVALSCOPE_AGENTX_API_KEY', env['EVALSCOPE_AGENTX_API_KEY'])
    monkeypatch.setenv('EVALSCOPE_AGENTX_HEADERS', env['EVALSCOPE_AGENTX_HEADERS'])
    _agentx_harness_cli.main(command[3:])
    assert captured


def test_corpus_provenance(tmp_path: Path) -> None:
    spec = mvp._DATASETS['full']
    public = harness._resolve_dataset(_args(), AgentXHarnessScenario())
    assert public.dataset_id == spec['huggingface_id']
    assert public.revision is None and public.sha256 == '' and public.verified is False
    result = harness._normalize_summary(
        {'metadata': {'submission_valid': True}}, 'completed', AgentXHarnessScenario(),
        public, 1, 900, 42, tmp_path, None, [],
    )
    assert result.aiperf_submission_valid is True and result.submission_valid is False
    (tmp_path / 'traces.jsonl').write_text('{}\n', encoding='utf-8')
    args = _args(dataset_path=str(tmp_path))
    with pytest.raises(ValueError, match='hash mismatch'):
        harness._resolve_dataset(args, args.scenario)
    args = _args(dataset_path=str(tmp_path), scenario={'name': 'agentx_harness', 'mode': 'smoke'})
    assert harness._resolve_dataset(args, args.scenario).verified is False


@pytest.mark.parametrize('mode,status,valid', [
    ('benchmark', 'completed', True), ('benchmark', 'completed', False),
    ('smoke', 'completed', True), ('benchmark', 'cancelled', True), ('benchmark', 'failed', True),
])
def test_native_validity(mode: str, status: str, valid: bool, tmp_path: Path) -> None:
    result = harness._normalize_summary(
        {'metadata': {'submission_valid': valid, 'submission_invalid_reasons': ['unsafe_override']}},
        status, AgentXHarnessScenario(mode=mode), _dataset(), 1, 900, 42, tmp_path, None, [],
    )
    assert result.submission_valid is (valid and mode == 'benchmark' and status == 'completed')
    assert result.aiperf_submission_valid is valid and result.mirror_revalidated is False


@pytest.mark.parametrize('code', [
    'pass', 'import sys; sys.exit(7)', "p.write_text('not-json')", "p.write_text('[]')",
    "p.write_text('{\"metadata\": [1]}')", "p.write_text('{\"request_count\": {\"unit\": \"requests\", \"avg\": []}}')",
])
def test_failed_exports(code: str, tmp_path: Path) -> None:
    result = _run(tmp_path, code)
    assert result.status == 'failed' and not result.submission_valid and result.error_summary


@pytest.mark.parametrize('secret', ['review-startup-sentinel${', '123', 'true', 'ms', 'quote"\\\\'])
def test_failure_diagnostics_and_metric_redaction(secret: str, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    env = {**os.environ, 'EVALSCOPE_AGENTX_API_KEY': secret,
           'EVALSCOPE_AGENTX_HEADERS': json.dumps({'X-Custom': secret})}
    result = _run(tmp_path, (
        'import os,json,sys; secret=os.environ["EVALSCOPE_AGENTX_API_KEY"]; print("startup: "+secret); '
        'p.write_text(json.dumps({"input_config":{"endpoint":{"api_key":secret,"headers":{"X-Custom":secret}}},'
        '"request_latency":{"unit":"ms","avg":1},"error_summary":[{"message":"error: "+secret}]})); sys.exit(7)'
    ), env)
    assert result.status == 'failed'
    assert result.metrics['request_latency'].unit == 'ms'
    assert result.error_summary == [{'message': 'error: <redacted>'}]
    assert 'startup: <redacted>' in capsys.readouterr().out
    assert 'startup: <redacted>' in (tmp_path / 'harness.log').read_text()
    raw = json.loads(Path(result.raw_summary_file).read_text())
    assert raw['input_config']['endpoint'] == {'api_key': '<redacted>', 'headers': {'X-Custom': '<redacted>'}}
    assert raw['request_latency']['unit'] == 'ms'


@pytest.mark.parametrize('sig', [None, signal.SIGINT, signal.SIGTERM])
def test_partial_export_is_not_retained(sig: int | None, tmp_path: Path) -> None:
    env = {**os.environ, 'EVALSCOPE_AGENTX_API_KEY': 'partial-export-sentinel-9bf42'}
    code = 'import os,time; p.write_text(\'{"api_key":"\'+os.environ["EVALSCOPE_AGENTX_API_KEY"]); '
    if sig is not None:
        code += f'os.kill(os.getppid(),{int(sig)}); time.sleep(60)'
    result = _run(tmp_path, code, env)
    assert result.status == ('cancelled' if sig is not None else 'failed') and result.raw_summary_file is None
    assert not list((tmp_path / 'harness').iterdir())


def test_redaction_write_failure_discards_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError('write failed')

    monkeypatch.setattr(harness, '_write_json', fail_write)
    result = _run(tmp_path, 'p.write_text(\'{"input_config":{"endpoint":{"api_key":"write-failure-sentinel"}}}\')')
    assert result.status == 'failed' and result.raw_summary_file is None
    assert not list((tmp_path / 'harness').iterdir())


@pytest.mark.parametrize('suffix', ['.log', '.txt', '.jsonl'])
@pytest.mark.parametrize('failure', ['read', 'write', 'decode'])
def test_diagnostic_sanitization_failure_discards_file(
    suffix: str, failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = 'diagnostic-failure-sentinel'
    bad = tmp_path / f'bad{suffix}'
    good = tmp_path / f'good{suffix}'
    payload = json.dumps({'error': {'message': secret}}) + '\n' if suffix == '.jsonl' else secret
    bad.write_bytes(b'\xff' if failure == 'decode' else payload.encode())
    good.write_text(payload)
    original = Path.open

    def open_file(path: Path, mode: str = 'r', *args: Any, **kwargs: Any) -> Any:
        if (failure == 'read' and path == bad) or (
            failure == 'write' and path == bad.with_suffix(f'{suffix}.tmp')
        ):
            raise OSError('diagnostic sanitization failed')
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_file)
    harness._sanitize_artifacts(tmp_path, [secret])
    assert not bad.exists() and not bad.with_suffix(f'{suffix}.tmp').exists()
    assert secret not in good.read_text() and '<redacted>' in good.read_text()


def test_diagnostic_deletion_failure_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / 'bad.log'
    bad.write_bytes(b'\xff')
    original = Path.unlink

    def unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == bad:
            raise PermissionError('unable to discard unsafe diagnostic')
        original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', unlink)
    with pytest.raises(PermissionError, match='unable to discard unsafe diagnostic'):
        harness._sanitize_artifacts(tmp_path, ['sentinel'])


def test_partial_jsonl_is_not_retained(tmp_path: Path) -> None:
    path = tmp_path / 'profile_export.jsonl'
    path.write_text('{"error":{"message":"credential-sentinel"}}\n{"error":')
    harness._sanitize_artifacts(tmp_path, ['credential-sentinel'])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_process_group_shutdown(sig: int, tmp_path: Path) -> None:
    handler = signal.getsignal(signal.SIGTERM)
    child_pid = tmp_path / 'child.pid'
    worker = (
        'import os,signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); '
        'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        f'open({str(child_pid)!r},"w").write(str(os.getpid())); time.sleep(60)'
    )
    code = (
        f'import os,signal,subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{worker!r}]); '
        f'path=pathlib.Path({str(child_pid)!r}); '
        '\nwhile not path.exists() or not path.read_text(): time.sleep(0.01)\n'
        f'os.kill(os.getppid(),{int(sig)}); time.sleep(60)'
    )
    result = _run(tmp_path, code)
    assert result.status == 'cancelled' and signal.getsignal(signal.SIGTERM) == handler
    pid = int(child_pid.read_text())
    for _ in range(100):
        try:
            if Path(f'/proc/{pid}/stat').read_text().split()[2] == 'Z':
                break
        except FileNotFoundError:
            break
        time.sleep(0.01)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail('Orphan worker survived shutdown')


@pytest.mark.skipif(importlib.util.find_spec('agentx_cli') is None, reason='Requires the isolated harness extra')
@pytest.mark.parametrize('echo_errors', [False, True])
def test_native_sweep(echo_errors: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('AIPERF_DATASET_CONFIGURATION_TIMEOUT', '30')
    requests = []
    api_key = 'agentx-api-key-sentinel-8ce93' + ('q7V9b3M5' * 32 if echo_errors else '')
    header = 'agentx-header-sentinel-7bd41'

    class Server(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"mock-model"}]}')

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, body, self.headers.get('Authorization'), self.headers.get('X-Custom')))
            if echo_errors and body.get('max_completion_tokens', body.get('max_tokens')) != 1 and len(requests) % 20 == 0:
                self.send_response(401)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f'invalid credential {api_key} header {header}'.encode())
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            base = {'id': 'mock', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'mock-model'}
            try:
                for content, finish in [('ok done', None), ('', 'length')]:
                    chunk = {**base, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': finish}]}
                    if finish:
                        chunk['usage'] = {'prompt_tokens': 64, 'completion_tokens': 2, 'total_tokens': 66}
                    self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
                    self.wfile.flush()
                    time.sleep(0.005)
                self.wfile.write(b'data: [DONE]\n\n')
            except (BrokenPipeError, ConnectionResetError):
                pass

    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, **{f'word{i}': i + 1 for i in range(1024)}}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token='[UNK]',
        chat_template='{% for message in messages %}{{ message["content"] }} {% endfor %}',
    ).save_pretrained(tmp_path / 'tokenizer')
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'traces.jsonl').write_text(''.join(json.dumps({
        'id': f'fixture-{i}', 'models': ['mock-model'], 'block_size': 64, 'hash_id_scope': 'local',
        'requests': [{'type': 'n', 't': 0.0, 'model': 'mock-model', 'in': 64, 'out': 2, 'hash_ids': [i]}],
    }) + '\n' for i in range(4)), encoding='utf-8')
    with ThreadingHTTPServer(('127.0.0.1', 0), Server) as server:
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        args = _args(
            scenario={'name': 'agentx_harness', 'mode': 'smoke', 'benchmark_grace_period': 2}, parallel=[1, 2],
            duration=3, tokenizer_path=str(tmp_path / 'tokenizer'), dataset_path=str(dataset), sleep_interval=0,
            url=f'http://127.0.0.1:{server.server_port}/proxy/v1/chat/completions', outputs_dir=str(tmp_path / 'results'),
            api_key=api_key, headers={'X-Custom': header},
        )
        try:
            results = run_perf_benchmark(args)
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert list(results) == ['parallel_1', 'parallel_2']
    for summary in results.values():
        assert summary['status'] == 'completed' and summary['submission_valid'] is False
        assert summary['schema_version'] == '2.0' and summary['seed'] == 42
        assert summary['metrics']['request_count']['avg'] > 0
    assert requests and all(path == '/proxy/v1/chat/completions' for path, *_ in requests)
    assert all(body.get('stream') is True and body.get('ignore_eos') is True for _, body, *_ in requests)
    assert all(auth == f'Bearer {api_key}' and value == header for _, _, auth, value in requests)
    output = capsys.readouterr().out
    for secret in [api_key, header, 'q7V9b3M5']:
        assert secret not in output
    sweep = Path(results['parallel_1']['raw_artifact_dir']).parents[1] / 'agentx_sweep_summary.json'
    assert json.loads(sweep.read_text()) == results
    for summary in results.values():
        assert Path(summary['raw_artifact_dir']).stat().st_mode & 0o077 == 0
        for artifact in Path(summary['raw_artifact_dir']).rglob('*'):
            if artifact.suffix in {'.json', '.jsonl', '.log', '.txt'}:
                content = artifact.read_text()
                for secret in [api_key, header, 'q7V9b3M5']:
                    assert secret not in content
        if echo_errors:
            records = list(Path(summary['raw_artifact_dir']).glob('profile_export.jsonl'))
            assert records and any(json.loads(line).get('error') for line in records[0].read_text().splitlines())
            assert summary['error_summary']
    assert 'agentx-api-key-sentinel-8ce93' not in sweep.read_text()
    assert 'agentx-header-sentinel-7bd41' not in sweep.read_text()
    with pytest.raises(FileExistsError):
        harness.run_agentx_harness_benchmark(args, str(sweep.parent.parent))
