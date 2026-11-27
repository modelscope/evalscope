"""Analysis entry points: per-sample analysis and provider resolution."""

from functools import lru_cache
from typing import AbstractSet, Dict, Iterable, List, Optional, Tuple

from evalscope.api.analysis import (
    FailureCase,
    ProviderSignals,
    SampleRef,
    TraceSignalProvider,
    TraceView,
)
from evalscope.api.analysis.config import TraceAnalysisConfig
from evalscope.api.evaluator.cache import ReviewResult
from evalscope.utils.logger import get_logger

from .attribution import apply_attribution, resolve_outcome
from .detectors import CompiledRules, DetectionContext, run_detectors
from .reader import build_view

logger = get_logger()

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def resolve_provider(benchmark: str) -> Optional[TraceSignalProvider]:
    """Load the benchmark's optional provider once; failures fall back to generic analysis."""
    try:
        import evalscope.benchmarks  # noqa: F401 - installs the lazy benchmark resolvers
        from evalscope.api.registry import BENCHMARK_REGISTRY, get_trace_signal_provider

        if benchmark in BENCHMARK_REGISTRY:
            provider_cls = get_trace_signal_provider(benchmark)
            return provider_cls() if provider_cls is not None else None
    except Exception as exc:  # noqa: BLE001 - optional providers must not stop generic analysis
        logger.warning(f'Trace signal provider for {benchmark!r} unavailable: {type(exc).__name__}: {exc}')
    return None


def _collect_signals(
    record: ReviewResult, view: TraceView, config: TraceAnalysisConfig
) -> Tuple[ProviderSignals, List[str]]:
    if not config.use_providers:
        return ProviderSignals(), []
    provider = resolve_provider(view.ref.benchmark)
    if provider is None:
        return ProviderSignals(), []
    try:
        signals = provider.collect(record, view)
    except Exception as exc:  # noqa: BLE001 - a provider bug must not stop the analysis
        logger.warning(f'Trace provider for {view.ref.benchmark} failed on {view.ref.key()}: {exc}')
        return ProviderSignals(), [f'provider_failed: {type(exc).__name__}']
    return signals, list(signals.notes)


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------


def analyze_sample(
    record: ReviewResult,
    ref: SampleRef,
    config: TraceAnalysisConfig,
    rules: Optional[CompiledRules] = None,
    metric_directions: Optional[Dict[str, str]] = None,
    correctness_metrics: AbstractSet[str] = frozenset(),
) -> FailureCase:
    """Analyze one review row; every sample gets a case, including successful ones."""
    rules = rules or CompiledRules.from_config(config)
    view = build_view(record, ref)
    signals, notes = _collect_signals(record, view, config)

    findings = run_detectors(DetectionContext(view=view, signals=signals, config=config, rules=rules))
    calls = {call.position: call.call_id for call in view.tool_calls}
    for finding in findings:
        finding.call_ids = [calls[p] for p in finding.positions if calls.get(p)]
    apply_attribution(findings, rules)
    outcome = resolve_outcome(view, signals, config, findings, metric_directions, correctness_metrics)
    return FailureCase(
        labels=signals.labels,
        outcome=outcome,
        score_value=view.score_value,
        score_status=view.score_status,
        findings=findings,
        tool_call_count=len(view.tool_calls),
        notes=notes,
    )


def analyze_records(
    records: Iterable[Tuple[SampleRef, ReviewResult]],
    config: Optional[TraceAnalysisConfig] = None,
    metric_directions: Optional[Dict[str, str]] = None,
    correctness_metrics: AbstractSet[str] = frozenset(),
) -> List[FailureCase]:
    """Analyze review rows with one compiled rule set per batch."""
    config = config or TraceAnalysisConfig()
    rules = CompiledRules.from_config(config)
    return [
        analyze_sample(record, ref, config, rules, metric_directions, correctness_metrics) for ref, record in records
    ]


__all__ = ['analyze_records', 'analyze_sample', 'resolve_provider']
