# SPDX-License-Identifier: Apache-2.0
"""Argument surface and validation for the AgentX scenario."""

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Aliases understood by AIPerf >= 0.13 (see aiperf/src/aiperf/plugin/plugins.yaml).
# ``full`` is the date-pinned 062126 corpus (393 traces, current default);
# ``256k`` is its filtered sibling for servers capped at ~256k context.
AGENTX_VARIANTS = {
    'full': 'semianalysis_cc_traces_weka_062126',
    '256k': 'semianalysis_cc_traces_weka_062126_256k',
}

# HF repos behind the aliases, recorded in results for provenance.
AGENTX_DATASET_REPOS = {
    'full': 'semianalysisai/cc-traces-weka-062126',
    '256k': 'semianalysisai/cc-traces-weka-062126-256k',
}

# Scenario locks documented by AIPerf; passing any of these alongside
# --scenario is either rejected by AIPerf or silently overrides the lock,
# so we reject them up front with a clear message.
_CONFLICTING_AIPERF_FLAGS = (
    '--fixed-schedule',
    '--request-rate',
    '--ignore-trace-delays',
)


class AgentxArguments(BaseModel):
    """AgentX-specific arguments (``--agentx-*`` CLI flags).

    These complement the standard Perf arguments (``--url``, ``--model``,
    ``--parallel``, ``--duration``, ...) which map onto their AIPerf
    counterparts in :mod:`evalscope.perf.agentx.runner`.
    """

    agentx_variant: str = 'full'
    """AgentX corpus variant: ``full`` (393 traces) or ``256k`` (filtered)."""

    agentx_aiperf_path: Optional[str] = None
    """Explicit path to the AIPerf executable. When omitted, the runner
    resolves ``aiperf`` from PATH (or ``uvx aiperf`` as a fallback)."""

    agentx_max_context_length: Optional[int] = None
    """Drops traces whose peak input+output exceeds this length (passed to
    AIPerf ``--max-context-length``)."""

    agentx_num_dataset_entries: Optional[int] = None
    """Smoke-test only: cap the number of eligible traces. Marking a run
    non-canonical is *not* automatic (AIPerf still stamps valid), so this
    wrapper flags it explicitly."""

    agentx_tokenizer: Optional[str] = None
    """Tokenizer for ISL/OSL accounting; defaults to the ``--model`` value."""

    agentx_random_seed: Optional[int] = None
    """Deterministic trajectory sampling seed (AIPerf ``--random-seed``)."""

    agentx_extra_args: List[str] = Field(default_factory=list)
    """Escape hatch: raw extra flags forwarded verbatim to ``aiperf profile``."""

    @field_validator('agentx_variant')
    @classmethod
    def _check_variant(cls, v: str) -> str:
        if v not in AGENTX_VARIANTS:
            raise ValueError(f'Unknown agentx variant {v!r}; expected one of {sorted(AGENTX_VARIANTS)}')
        return v

    @field_validator('agentx_extra_args')
    @classmethod
    def _check_extra_args(cls, v: List[str]) -> List[str]:
        for item in v:
            for flag in _CONFLICTING_AIPERF_FLAGS:
                if item == flag or item.startswith(flag + '='):
                    raise ValueError(
                        f'{flag} conflicts with the inferencex-agentx-mvp scenario locks; '
                        'the scenario auto-fills the agentic-replay scheduler'
                    )
        return v


def validate_agentx_arguments(
    url: str,
    model: str,
    parallel,
    duration,
    tokenizer_path: Optional[str] = None,
    max_context_length: Optional[int] = None,
) -> None:
    """Fail fast on wrapper-level inconsistencies AIPerf would only report
    mid-run (or not at all). Raises ``ValueError`` with an actionable message."""
    if not model:
        raise ValueError('AgentX scenario requires --model (the served model name)')
    if not url:
        raise ValueError('AgentX scenario requires --url pointing at the OpenAI-compatible server')
    if not _is_http_url(url):
        raise ValueError(f'--url must be an http(s) URL, got {url!r}')
    if isinstance(parallel, (list, tuple)):
        if len(parallel) != 1:
            raise ValueError('AgentX concurrency maps to AIPerf --concurrency; comma-list sweeps are not supported')
        parallel = parallel[0]
    if not isinstance(parallel, int) or parallel < 1:
        raise ValueError(f'--parallel must be a positive integer (session trees), got {parallel!r}')
    if isinstance(duration, (list, tuple)):
        if len(duration) != 1:
            raise ValueError('AgentX duration maps to AIPerf --benchmark-duration; sweeps are not supported')
        duration = duration[0]
    if not isinstance(duration, int) or duration < 1:
        raise ValueError(f'--duration must be a positive integer (seconds), got {duration!r}')
    if max_context_length is not None and max_context_length < 1:
        raise ValueError(f'--agentx-max-context-length must be positive, got {max_context_length!r}')


_URL_RE = re.compile(r'^https?://')


def _is_http_url(url: str) -> bool:
    return bool(_URL_RE.match(url or ''))
