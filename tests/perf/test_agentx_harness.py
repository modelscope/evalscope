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


def _run(tmp_path: Path, code: str) -> harness.AgentXHarnessRunSummary:
    artifacts = tmp_path / 'harness'
    artifacts.mkdir()
    prefix = f'import pathlib; p=pathlib.Path({str(artifacts / harness._AIPERF_SUMMARY)!r}); '
    return harness._run_one_agentx(
        [sys.executable, '-c', prefix + code], os.environ.copy(), AgentXHarnessScenario(),
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
    assert command[:4] == [sys.executable, '-m', 'agentx_cli', 'profile']
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
    args = _args(scenario={'name': 'agentx_harness', 'mode': 'smoke'}, api_key='secret', headers={'X-Custom': 'private'})
    command, _ = harness._build_command(args, args.scenario, _dataset(), 1, 60, 42, tmp_path)
    assert command[command.index('--hf-weka-dataset') + 1] == '/data'
    assert command[command.index('--num-dataset-entries') + 1] == '4'
    assert '--unsafe-override' in command
    assert not any(secret in ' '.join(harness._redact_command(command)) for secret in ['secret', 'private'])


def test_corpus_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    spec = mvp._DATASETS['full']
    info = SimpleNamespace(sha=spec['huggingface_revision'], siblings=[
        SimpleNamespace(rfilename='traces.jsonl', lfs=SimpleNamespace(sha256=spec['sha256'])),
    ])

    def dataset_info(*args: Any, **kwargs: Any) -> Any:
        assert kwargs['revision'] == 'main'
        return info

    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(dataset_info=dataset_info))
    assert harness._resolve_dataset(_args(), AgentXHarnessScenario()).dataset_id == spec['huggingface_id']
    for changed in [SimpleNamespace(sha='moved', siblings=info.siblings), SimpleNamespace(sha=info.sha, siblings=[])]:
        monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(dataset_info=lambda *a, **kw: changed))
        with pytest.raises(RuntimeError):
            harness._verify_huggingface_revision(spec)
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
def test_native_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('AIPERF_DATASET_CONFIGURATION_TIMEOUT', '30')
    requests = []

    class Server(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"mock-model"}]}')

        def do_POST(self) -> None:
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
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
    assert requests and all(path == '/proxy/v1/chat/completions' for path, _ in requests)
    assert all(body.get('stream') is True and body.get('ignore_eos') is True for _, body in requests)
    sweep = Path(results['parallel_1']['raw_artifact_dir']).parents[1] / 'agentx_sweep_summary.json'
    assert json.loads(sweep.read_text()) == results
    with pytest.raises(FileExistsError):
        harness.run_agentx_harness_benchmark(args, str(sweep.parent.parent))
