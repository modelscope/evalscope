"""Analysis entry points: per-sample analysis and provider resolution."""

from threading import Lock
from typing import Dict, Iterable, List, Optional, Tuple

from evalscope.api.analysis import (
    INFORMATIONAL_CATEGORIES,
    Confidence,
    FailureCase,
    Finding,
    FindingCategory,
    ProviderSignals,
    SamplePrecursors,
    SampleRef,
    TraceSignalProvider,
    TraceView,
)
from evalscope.api.analysis.config import TraceAnalysisConfig
from evalscope.api.evaluator.cache import ReviewResult
from evalscope.utils.logger import get_logger

from .attribution import apply_attribution, failure_mode, primary_attribution, resolve_outcome
from .detectors import CompiledRules, DetectionContext, run_detectors
from .reader import build_view

logger = get_logger()

_RISK_FAILURE_CATEGORIES = frozenset({FindingCategory.ATTACK_GOAL_ACTION, FindingCategory.RISKY_COMMAND})

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

_PROVIDERS: Dict[str, Optional[TraceSignalProvider]] = {}
_PROVIDER_LOCK = Lock()


def resolve_provider(benchmark: str) -> Optional[TraceSignalProvider]:
    """Return the provider of ``benchmark``, importing only that benchmark's adapter module.

    Any failure (unknown benchmark, missing optional dependency, construction error) is logged and cached as
    ``None`` so that the generic analysis continues.
    """
    with _PROVIDER_LOCK:
        if benchmark not in _PROVIDERS:
            provider: Optional[TraceSignalProvider] = None
            try:
                import evalscope.benchmarks  # noqa: F401 - installs the lazy benchmark resolvers
                from evalscope.api.registry import BENCHMARK_REGISTRY, get_trace_signal_provider

                if benchmark in BENCHMARK_REGISTRY:
                    provider_cls = get_trace_signal_provider(benchmark)
                    provider = provider_cls() if provider_cls is not None else None
            except Exception as exc:  # noqa: BLE001 - provider loading must never stop the analysis
                logger.warning(f'Trace signal provider for {benchmark!r} unavailable: {type(exc).__name__}: {exc}')
            _PROVIDERS[benchmark] = provider
        return _PROVIDERS[benchmark]


def clear_provider_cache() -> None:
    """Forget resolved providers (used by tests)."""
    with _PROVIDER_LOCK:
        _PROVIDERS.clear()


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
    view.providers.append(provider.name or type(provider).__name__)
    return signals, list(signals.notes)


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------


def analyze_sample(
    record: ReviewResult,
    ref: SampleRef,
    config: TraceAnalysisConfig,
    rules: Optional[CompiledRules] = None,
) -> FailureCase:
    """Analyze one review row; every sample gets a case, including successful ones."""
    rules = rules or CompiledRules.from_config(config)
    view = build_view(record, ref)
    signals, notes = _collect_signals(record, view, config)
    view.labels.update(signals.labels)

    findings = run_detectors(DetectionContext(view=view, signals=signals, config=config, rules=rules))
    calls = {call.position: call.call_id for call in view.tool_calls}
    for finding in findings:
        finding.call_ids = [calls[p] for p in finding.positions if calls.get(p)]
    apply_attribution(findings, rules)
    outcome = resolve_outcome(view, signals, config, findings)
    if outcome.task_success is False and all(f.category in INFORMATIONAL_CATEGORIES for f in findings):
        # A failed benchmark result alone does not establish abnormal termination or its cause.
        outcome_finding = Finding(
            ref=ref,
            category=FindingCategory.OUTCOME,
            subtype='benchmark_task_failed',
            confidence=Confidence.LOW,
            detector='outcome@2',
            data={'source': outcome.source},
        )
        apply_attribution([outcome_finding], rules)
        findings.append(outcome_finding)

    tainted = sum(f.category == FindingCategory.TAINTED_SIDE_EFFECT for f in findings)
    precursors = SamplePrecursors(
        side_effect_calls=sum(f.category == FindingCategory.SIDE_EFFECT for f in findings),
        tainted_side_effect_calls=tainted,
        has_tainted_side_effect=tainted > 0,
        attack_goal_actions=(
            sum(f.category == FindingCategory.ATTACK_GOAL_ACTION for f in findings)
            if signals.attack_goal_calls is not None
            else None
        ),
    )
    failed = (
        outcome.task_success is False
        or outcome.safety_violation is True
        or (view.score_status or 'success').lower() == 'excluded'
        or any(f.category in _RISK_FAILURE_CATEGORIES and f.confidence == Confidence.HIGH for f in findings)
    )
    primary, key_finding = primary_attribution(findings)
    return FailureCase(
        ref=ref,
        labels=view.labels,
        task_excerpt=(view.user_text or '')[:500],
        outcome=outcome,
        score_value=view.score_value,
        score_status=view.score_status,
        failed=failed,
        primary_attribution=primary,
        failure_mode=failure_mode(findings, primary) if failed else None,
        key_position=key_finding.first_position if key_finding is not None else None,
        findings=findings,
        precursors=precursors,
        tool_call_count=len(view.tool_calls),
        providers=view.providers,
        notes=notes,
    )


def analyze_records(
    records: Iterable[Tuple[SampleRef, ReviewResult]],
    config: Optional[TraceAnalysisConfig] = None,
) -> List[FailureCase]:
    """Analyze in-memory review rows (e.g. from an experiment driver that already loaded them)."""
    config = config or TraceAnalysisConfig()
    rules = CompiledRules.from_config(config)
    return [analyze_sample(record, ref, config, rules) for ref, record in records]


__all__ = ['analyze_records', 'analyze_sample', 'clear_provider_cache', 'resolve_provider']
