# SPDX-License-Identifier: Apache-2.0
"""Build, execute, and harvest AIPerf ``inferencex-agentx-mvp`` runs."""

import json
import os
import re
import shutil
import subprocess  # nosec B404 - invoking the user-resolved aiperf executable
from typing import Any, Dict, List, Optional, Tuple

from evalscope.perf.agentx.config import AGENTX_VARIANTS

PER_RUN_EXPORT = 'profile_export_aiperf.json'
AGGREGATE_EXPORT = os.path.join('aggregate', 'profile_export_aiperf_aggregate.json')

# Flags the scenario hard-locks; writing them out keeps the constructed
# command self-documenting (AIPerf would auto-fill identical values).
_SCENARIO_LOCKED_FLAGS = [
    '--endpoint-type', 'chat',
    '--use-server-token-count',
    '--streaming',
    '--extra-inputs', 'ignore_eos:true',
    '--cache-bust', 'first_turn_prefix',
    '--system-idle-gap-cap-seconds', '10',
    '--trajectory-start-min-ratio', '0.0',
    '--trajectory-start-max-ratio', '1.0',
    '--ui', 'simple',
]


def build_aiperf_command(
    aiperf_executable: str,
    url: str,
    model: str,
    variant: str = 'full',
    parallel: int = 8,
    duration: int = 3600,
    api_key: Optional[str] = None,
    tokenizer: Optional[str] = None,
    max_context_length: Optional[int] = None,
    num_dataset_entries: Optional[int] = None,
    random_seed: Optional[int] = None,
    artifact_dir: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Construct the ``aiperf profile`` command line.

    Only the URL's scheme://host[:port] is forwarded: AIPerf appends the
    endpoint path itself based on ``--endpoint-type chat``.
    """
    cmd = [aiperf_executable, 'profile', '--scenario', 'inferencex-agentx-mvp']
    cmd += ['--url', _strip_endpoint_path(url)]
    cmd += ['--model', model]
    cmd += ['--public-dataset', AGENTX_VARIANTS[variant]]
    cmd += ['--concurrency', str(parallel)]
    cmd += ['--benchmark-duration', str(duration)]
    if api_key:
        cmd += ['--api-key', api_key]
    if tokenizer:
        cmd += ['--tokenizer', tokenizer]
    if max_context_length is not None:
        cmd += ['--max-context-length', str(max_context_length)]
    if num_dataset_entries is not None:
        cmd += ['--num-dataset-entries', str(num_dataset_entries)]
    if random_seed is not None:
        cmd += ['--random-seed', str(random_seed)]
    if artifact_dir:
        cmd += ['--artifact-dir', artifact_dir]
    cmd += _SCENARIO_LOCKED_FLAGS
    if extra_args:
        cmd += list(extra_args)
    return cmd


def _strip_endpoint_path(url: str) -> str:
    m = re.match(r'^(https?://[^/]+)', url)
    if not m:
        raise ValueError(f'--url must be an http(s) URL, got {url!r}')
    return m.group(1)


def resolve_aiperf_executable(explicit_path: Optional[str] = None) -> Tuple[str, List[str]]:
    """Return ``(executable, prefix_args)`` for launching AIPerf.

    Resolution order: explicit path → ``aiperf`` on PATH → ``uvx aiperf``.
    Raises ``FileNotFoundError`` when none is available (AIPerf is an
    optional dependency by design).
    """
    if explicit_path:
        if not (os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK)):
            raise FileNotFoundError(
                f'--agentx-aiperf-path {explicit_path!r} is not an executable file'
            )
        return explicit_path, []
    found = shutil.which('aiperf')
    if found:
        return found, []
    if shutil.which('uvx'):
        return 'uvx', ['aiperf']
    raise FileNotFoundError(
        'AIPerf executable not found. Install it with `pip install aiperf` '
        '(>= 0.13) or pass --agentx-aiperf-path.'
    )


def query_aiperf_version(executable: str, prefix_args: Optional[List[str]] = None) -> Optional[str]:
    """Best-effort version probe; ``None`` when the command fails."""
    try:
        out = subprocess.run(  # nosec B603 - fixed executable, no shell
            [executable, *(prefix_args or []), '--version'],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or '') + (out.stderr or '')
    m = re.search(r'(\d+\.\d+\.\d+(?:[+.-]\S+)?)', text)
    return m.group(1) if m else (text.strip() or None)


def parse_profile_export(path: str) -> Dict[str, Any]:
    """Load a ``profile_export_aiperf.json`` (per-run or aggregate layout).

    The per-run file has metrics top-level next to ``metadata``; the
    aggregate file nests them under ``metrics``. Returns a dict with
    ``metadata`` and ``metrics`` keys regardless of layout.
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Unexpected profile export layout in {path}')
    if 'metrics' in data and isinstance(data['metrics'], dict):
        metadata = data.get('metadata') or {}
        metrics = data['metrics']
    else:
        metadata = data.get('metadata') or {}
        metrics = {k: v for k, v in data.items() if k != 'metadata'}
    return {'metadata': metadata, 'metrics': metrics}


class AIPerfRunner:
    """Execute one AgentX run and retain its raw artifacts."""

    def __init__(self, args, env: Optional[Dict[str, str]] = None):
        """``args`` is an ``evalscope.perf.arguments.Arguments`` instance."""
        self.perf_args = args
        self.env = env
        executable, prefix = resolve_aiperf_executable(getattr(args, 'agentx_aiperf_path', None))
        self.executable = executable
        self.executable_prefix = prefix
        self.aiperf_version = query_aiperf_version(executable, prefix)

    def build_command(self, artifact_dir: str) -> List[str]:
        cmd = build_aiperf_command(
            self.executable,
            url=self.perf_args.url,
            model=self.perf_args.model,
            variant=self.perf_args.agentx_variant,
            parallel=_first(self.perf_args.parallel),
            duration=_first(self.perf_args.duration),
            api_key=self.perf_args.api_key,
            tokenizer=self.perf_args.agentx_tokenizer or self.perf_args.tokenizer_path,
            max_context_length=self.perf_args.agentx_max_context_length,
            num_dataset_entries=self.perf_args.agentx_num_dataset_entries,
            random_seed=self.perf_args.agentx_random_seed,
            artifact_dir=artifact_dir,
            extra_args=self.perf_args.agentx_extra_args,
        )
        return self.executable_prefix + cmd

    def run(self, artifact_dir: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        """Launch AIPerf, blocking until completion. Raw output stays under
        ``artifact_dir`` (AIPerf writes its exports there)."""
        cmd = self.build_command(artifact_dir)
        merged_env = {**os.environ, **(env or self.env or {})}
        return subprocess.run(  # nosec B603 - constructed command, no shell
            cmd,
            cwd=artifact_dir,
            env=merged_env,
            capture_output=True,
            text=True,
        )


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value
