"""AgentX Harness serving-performance scenarios, separate from the legacy MVP runner.

The pinned harness owns official scenario locks, warmup, trajectory replay, and
native metrics. EvalScope owns configuration, provenance, process supervision,
and summaries. The harness and upstream AIPerf must not share an environment.
"""

import importlib.metadata
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalscope.api.dataset import download_dataset_snapshot
from evalscope.constants import HubType
from evalscope.utils import get_secret_value
from evalscope.utils.logger import get_logger

from .agentx import (
    _DATASETS,
    AgentXDatasetProvenance,
    AgentXMetric,
    _derive_per_gpu_metrics,
    _print_summary,
    _sha256,
    _write_json,
    validate_agentx_arguments,
)

if TYPE_CHECKING:
    from evalscope.perf.arguments import Arguments

logger = get_logger()

_HARNESS_VERSION = '1.0.13'
_DEFAULT_SEED = 42
_DEFAULT_DURATION = 3600.0
_SMOKE_DURATION = 60.0
_SMOKE_TRACE_LIMIT = 4
_AIPERF_SUMMARY = 'profile_export_aiperf.json'


class AgentXHarnessScenario(BaseModel):
    """Configure the official harness; the scenario name selects its corpus."""

    model_config = ConfigDict(extra='forbid')

    name: Literal['agentx_harness', 'agentx_harness_256k'] = 'agentx_harness'
    variant: Literal['256k', 'full'] = 'full'
    mode: Literal['benchmark', 'smoke'] = 'benchmark'
    max_context_length: Optional[int] = Field(default=None, gt=0)
    trace_limit: Optional[int] = Field(default=None, gt=0)
    num_gpus: Optional[int] = Field(default=None, gt=0)
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    hardware: Optional[str] = None
    tokenizer_trust_remote_code: bool = False
    tokenizer_revision: Optional[str] = None
    request_timeout_seconds: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    benchmark_grace_period: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode='before')
    @classmethod
    def _select_corpus(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            name = value.get('name', 'agentx_harness')
            expected = '256k' if name == 'agentx_harness_256k' else 'full'
            if 'variant' in value and value['variant'] != expected:
                raise ValueError(f'{name} requires variant={expected}; select the other scenario name instead.')
            value['variant'] = expected
        return value

    @model_validator(mode='after')
    def _validate_mode(self) -> 'AgentXHarnessScenario':
        if self.mode == 'benchmark' and self.trace_limit is not None:
            raise ValueError('trace_limit is only supported when mode is smoke')
        return self

    @property
    def effective_trace_limit(self) -> Optional[int]:
        """Return the resolved smoke trace limit, or no benchmark limit."""
        return (self.trace_limit or _SMOKE_TRACE_LIMIT) if self.mode == 'smoke' else None


class AgentXHarnessRunSummary(BaseModel):
    """EvalScope summary alongside untouched native harness artifacts."""

    schema_version: str = '2.0'
    agentx_harness_version: str = _HARNESS_VERSION
    harness_scenario: Literal['agentx'] = 'agentx'
    status: Literal['completed', 'failed', 'cancelled']
    scenario: AgentXHarnessScenario
    concurrency: int
    duration: float
    seed: int
    aiperf_version: Optional[str] = None
    aiperf_schema_version: Optional[str] = None
    dataset: AgentXDatasetProvenance
    submission_valid: bool
    aiperf_submission_valid: Optional[bool] = None
    aiperf_submission_invalid_reasons: List[str] = Field(default_factory=list)
    mirror_revalidated: bool = False
    metrics: Dict[str, AgentXMetric] = Field(default_factory=dict)
    derived_metrics: Dict[str, AgentXMetric] = Field(default_factory=dict)
    error_summary: List[Dict[str, Any]] = Field(default_factory=list)
    raw_artifact_dir: str
    raw_summary_file: Optional[str] = None
    command: List[str] = Field(default_factory=list)


def validate_agentx_harness_arguments(args: 'Arguments') -> None:
    """Reject Perf controls incompatible with the official harness."""
    validate_agentx_arguments(args)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError('AgentX Harness requires a finite positive --duration.')
    _split_chat_url(args.url)


def run_agentx_harness_benchmark(args: 'Arguments', output_path: str) -> Dict[str, Dict[str, Any]]:
    """Run one harness process per concurrency point and retain native artifacts."""
    _require_harness()
    scenario = args.scenario
    assert isinstance(scenario, AgentXHarnessScenario)
    dataset = _resolve_dataset(args, scenario)
    seed = args.seed if args.seed is not None else _DEFAULT_SEED
    duration = _resolve_duration(args, scenario)
    base_path = Path(output_path) / scenario.name
    results: Dict[str, Dict[str, Any]] = {}
    failed_run: Optional[AgentXHarnessRunSummary] = None
    for index, concurrency in enumerate(args.parallel):
        run_path = base_path / f'parallel_{concurrency}'
        artifacts_path = run_path / 'harness'
        # Never accept a stale native export as the result of a new run.
        artifacts_path.mkdir(parents=True, exist_ok=False)
        command, env = _build_command(args, scenario, dataset, int(concurrency), duration, seed, artifacts_path)
        summary = _run_one_agentx(
            command, env, scenario, dataset, int(concurrency), duration, seed, run_path, artifacts_path
        )
        results[f'parallel_{concurrency}'] = summary.model_dump(mode='json')
        _write_json(run_path / 'agentx_summary.json', results[f'parallel_{concurrency}'])
        if summary.status != 'completed':
            failed_run = summary
            break
        if index < len(args.parallel) - 1:
            logger.info(f'Sleeping for {args.sleep_interval} seconds before the next AgentX Harness run...')
            time.sleep(args.sleep_interval)
    _write_json(base_path / 'agentx_sweep_summary.json', results)
    _print_summary(results)
    if failed_run is not None:
        raise RuntimeError(f'AgentX Harness run {failed_run.status}; inspect {failed_run.raw_artifact_dir}')
    return results


def _require_harness() -> None:
    if os.name != 'posix' or not (sys.version_info.major == 3 and 11 <= sys.version_info.minor < 14):
        raise RuntimeError('AgentX Harness requires POSIX and Python 3.11-3.13.')
    try:
        version = importlib.metadata.version('agentx-harness')
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            'Install `evalscope[agentx_harness]` in a separate environment from `evalscope[agentx]`.'
        ) from error
    if version != _HARNESS_VERSION:
        raise RuntimeError(f'AgentX Harness requires agentx-harness=={_HARNESS_VERSION}, found {version}.')
    try:
        importlib.metadata.version('aiperf')
    except importlib.metadata.PackageNotFoundError:
        return
    raise RuntimeError('Conflicting aiperf distribution installed; rebuild with only `evalscope[agentx_harness]`.')


def _resolve_dataset(args: 'Arguments', scenario: AgentXHarnessScenario) -> AgentXDatasetProvenance:
    spec = _DATASETS[scenario.variant]
    source = HubType.LOCAL if args.dataset_path else (args.data_source or HubType.HUGGINGFACE)
    if source not in {HubType.MODELSCOPE, HubType.HUGGINGFACE, HubType.LOCAL}:
        raise ValueError(f'Unsupported AgentX Harness data source: {source}')
    if source == HubType.HUGGINGFACE:
        _verify_huggingface_revision(spec)
        return AgentXDatasetProvenance(
            source=source,
            dataset_id=spec['huggingface_id'],
            revision=spec['huggingface_revision'],
            trace_file='',
            sha256=spec['sha256'],
            verified=True,
        )
    if source == HubType.MODELSCOPE:
        root = download_dataset_snapshot(
            spec['modelscope_id'],
            data_source=source,
            revision=spec['modelscope_revision'],
            allow_file_pattern='traces.jsonl',
        )
        dataset_id = spec['modelscope_id']
        revision = spec['modelscope_revision']
    else:
        root = download_dataset_snapshot(args.dataset_path, data_source=source)
        dataset_id = os.path.realpath(args.dataset_path)
        revision = None
    trace_file = Path(root) / 'traces.jsonl'
    if not trace_file.is_file():
        raise FileNotFoundError(f'AgentX Harness dataset must contain traces.jsonl: {root}')
    digest = _sha256(trace_file)
    verified = digest == spec['sha256']
    if scenario.mode == 'benchmark' and not verified:
        raise ValueError(f'AgentX Harness benchmark dataset hash mismatch for {trace_file}: {digest}')
    return AgentXDatasetProvenance(
        source=source,
        dataset_id=dataset_id,
        revision=revision,
        trace_file=str(trace_file),
        sha256=digest,
        verified=verified,
    )


def _verify_huggingface_revision(spec: Dict[str, str]) -> None:
    from huggingface_hub import HfApi

    try:
        # Public aliases load the current head, not an arbitrary historic revision.
        info = HfApi().dataset_info(spec['huggingface_id'], revision='main', files_metadata=True)
    except Exception as error:
        raise RuntimeError('Unable to verify the AgentX Harness public dataset revision.') from error
    if info.sha != spec['huggingface_revision']:
        raise RuntimeError(f'AgentX Harness public dataset revision changed: expected {spec["huggingface_revision"]}.')
    trace = next((file for file in info.siblings if file.rfilename == 'traces.jsonl'), None)
    digest = trace.lfs.sha256 if trace is not None and trace.lfs is not None else None
    if digest != spec['sha256']:
        raise RuntimeError(f'AgentX Harness traces.jsonl hash changed: expected {spec["sha256"]}.')


def _resolve_duration(args: 'Arguments', scenario: AgentXHarnessScenario) -> float:
    if args.duration is not None:
        return float(args.duration)
    return _SMOKE_DURATION if scenario.mode == 'smoke' else _DEFAULT_DURATION


def _build_command(
    args: 'Arguments',
    scenario: AgentXHarnessScenario,
    dataset: AgentXDatasetProvenance,
    concurrency: int,
    duration: float,
    seed: int,
    artifacts_path: Path,
) -> tuple[List[str], Dict[str, str]]:
    base_url, endpoint = _split_chat_url(args.url)
    command = [
        sys.executable,
        '-m',
        'agentx_cli',
        'profile',
        '--scenario',
        'agentx',
        '--url',
        base_url,
        '--custom-endpoint',
        endpoint,
        '--endpoint-type',
        'chat',
        '--model',
        args.model,
        '--concurrency',
        str(concurrency),
        '--benchmark-duration',
        str(duration),
        '--random-seed',
        str(seed),
        '--use-server-token-count',
        '--streaming',
        '--extra-inputs',
        'ignore_eos:true',
        '--cache-bust',
        'first_turn_prefix',
        '--system-idle-gap-cap-seconds',
        '10',
        '--trajectory-start-min-ratio',
        '0.25',
        '--trajectory-start-max-ratio',
        '0.75',
        '--artifact-dir',
        str(artifacts_path),
        '--ui',
        'none',
        '--tokenizer',
        args.tokenizer_path,
    ]
    if scenario.tokenizer_trust_remote_code:
        command.append('--tokenizer-trust-remote-code')
    for flag, value in (
        ('--tokenizer-revision', scenario.tokenizer_revision),
        ('--request-timeout-seconds', scenario.request_timeout_seconds),
        ('--benchmark-grace-period', scenario.benchmark_grace_period),
        ('--max-context-length', scenario.max_context_length),
        ('--num-dataset-entries', scenario.effective_trace_limit),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    if dataset.source == HubType.HUGGINGFACE:
        command.extend(['--public-dataset', _DATASETS[scenario.variant]['public_dataset']])
    else:
        # The harness accepts a directory of JSONL rows through its datasets loader.
        command.extend(['--hf-weka-dataset', str(Path(dataset.trace_file).parent)])
    if scenario.mode == 'smoke' or dataset.source != HubType.HUGGINGFACE:
        command.append('--unsafe-override')
    api_key = get_secret_value(args.api_key)
    if api_key:
        command.extend(['--api-key', str(api_key)])
    for key, value in args.headers.items():
        raw_value = str(get_secret_value(value))
        if api_key and key.lower() == 'authorization' and raw_value == f'Bearer {api_key}':
            continue
        command.extend(['--header', f'{key}:{raw_value}'])
    return command, os.environ.copy()


def _run_one_agentx(
    command: List[str],
    env: Dict[str, str],
    scenario: AgentXHarnessScenario,
    dataset: AgentXDatasetProvenance,
    concurrency: int,
    duration: float,
    seed: int,
    run_path: Path,
    artifacts_path: Path,
) -> AgentXHarnessRunSummary:
    status: Literal['completed', 'failed', 'cancelled'] = 'failed'
    process: Optional[subprocess.Popen[str]] = None
    fallback_error: Optional[str] = None
    previous_term_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_term_handler = signal.signal(signal.SIGTERM, _handle_interrupt)
    try:
        with (run_path / 'harness.log').open('w', encoding='utf-8') as log_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
            return_code = process.wait()
        status = 'completed' if return_code == 0 else 'failed'
        if return_code != 0:
            fallback_error = f'AgentX Harness exited with return code {return_code}.'
    except OSError as error:
        fallback_error = f'Unable to run AgentX Harness: {error}'
    except KeyboardInterrupt:
        status = 'cancelled'
    finally:
        if process is not None:
            if process.poll() is None:
                _stop_process_group(process)
            if process.stdout is not None:
                process.stdout.close()
        if previous_term_handler is not None:
            signal.signal(signal.SIGTERM, previous_term_handler)

    raw_summary = artifacts_path / _AIPERF_SUMMARY
    raw_data: Dict[str, Any] = {}
    try:
        raw_data = json.loads(raw_summary.read_text(encoding='utf-8'))
        if not isinstance(raw_data, dict):
            raise ValueError('summary must be a JSON object')
    except (OSError, ValueError) as error:
        raw_data = {}
        if status != 'cancelled':
            status = 'failed'
            fallback_error = fallback_error or f'Unable to read AgentX Harness {_AIPERF_SUMMARY}: {error}'
    if raw_data.get('was_cancelled'):
        status = 'cancelled'
    try:
        summary = _normalize_summary(
            raw_data,
            status,
            scenario,
            dataset,
            concurrency,
            duration,
            seed,
            artifacts_path,
            raw_summary if raw_summary.is_file() else None,
            _redact_command(command),
        )
    except (ValueError, TypeError) as error:
        summary = _normalize_summary(
            {},
            'cancelled' if status == 'cancelled' else 'failed',
            scenario,
            dataset,
            concurrency,
            duration,
            seed,
            artifacts_path,
            raw_summary if raw_summary.is_file() else None,
            _redact_command(command),
        )
        fallback_error = f'Invalid AgentX Harness summary: {error}'
    if fallback_error and not summary.error_summary:
        summary.error_summary = [{'message': fallback_error}]
    return summary


def _normalize_summary(
    raw_data: Dict[str, Any],
    status: Literal['completed', 'failed', 'cancelled'],
    scenario: AgentXHarnessScenario,
    dataset: AgentXDatasetProvenance,
    concurrency: int,
    duration: float,
    seed: int,
    artifacts_path: Path,
    raw_summary: Optional[Path],
    command: List[str],
) -> AgentXHarnessRunSummary:
    metadata = raw_data.get('metadata') or {}
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be a JSON object')
    metrics = {
        name: AgentXMetric.model_validate(value)
        for name, value in raw_data.items()
        if isinstance(value, dict) and isinstance(value.get('unit'), str)
    }
    raw_valid = metadata.get('submission_valid')
    # Native validation is authoritative; checksum-verified mirrors are not promoted.
    submission_valid = raw_valid is True and dataset.verified and scenario.mode == 'benchmark' and status == 'completed'
    return AgentXHarnessRunSummary(
        status=status,
        scenario=scenario,
        concurrency=concurrency,
        duration=duration,
        seed=seed,
        aiperf_version=raw_data.get('aiperf_version'),
        aiperf_schema_version=raw_data.get('schema_version'),
        dataset=dataset,
        submission_valid=submission_valid,
        aiperf_submission_valid=raw_valid if isinstance(raw_valid, bool) else None,
        aiperf_submission_invalid_reasons=metadata.get('submission_invalid_reasons') or [],
        metrics=metrics,
        derived_metrics=_derive_per_gpu_metrics(metrics, scenario.num_gpus),
        error_summary=raw_data.get('error_summary') or [],
        raw_artifact_dir=str(artifacts_path),
        raw_summary_file=str(raw_summary) if raw_summary else None,
        command=command,
    )


def _handle_interrupt(signum: int, frame: Optional[FrameType]) -> None:
    raise KeyboardInterrupt


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    """Stop the launcher and its workers, escalating only when necessary."""
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # The launcher has exited; kill any orphan workers, including ones ignoring SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return


def _split_chat_url(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {'http', 'https'}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError('AgentX Harness requires an absolute HTTP(S) chat URL without credentials, query or fragment.')
    expected = '/v1/chat/completions'
    if not parsed.path.rstrip('/').endswith(expected):
        raise ValueError(f'AgentX Harness requires a URL ending with {expected}.')
    return urlunsplit((parsed.scheme, parsed.netloc, '', '', '')), parsed.path.rstrip('/')


def _redact_command(command: List[str]) -> List[str]:
    redacted = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append('<redacted>')
            redact_next = False
        else:
            redacted.append(re.sub(r'\$\{EVALSCOPE_AGENTX_[^}]+\}', '<redacted>', part))
            redact_next = part in {'--api-key', '--header'}
    return redacted
