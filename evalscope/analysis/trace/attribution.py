"""Per-finding attribution and sample outcomes."""

import re
from typing import AbstractSet, Dict, List, Optional, Pattern, Tuple

from evalscope.api.analysis import (
    INFORMATIONAL_CATEGORIES,
    Attribution,
    Confidence,
    Finding,
    FindingCategory,
    Outcome,
    ProviderSignals,
    TraceView,
)
from evalscope.api.analysis.config import TraceAnalysisConfig

from .detectors import CompiledRules

_BAD_ARGUMENT = re.compile(
    r'\b(?:ValidationError|validation errors? for|missing (?:\d+ )?required (?:argument|field)'
    r'|unexpected keyword argument|invalid (?:value|argument|id))\b',
    re.IGNORECASE,
)
_TRACEBACK = re.compile(r'Traceback \(most recent call last\)')
_UNKNOWN_TOOL_SUBTYPES = {
    'provider:invalid_function',
    'text_pattern:not_found_tool',
    'text_pattern:empty_function_name',
}
_PARSE_SUBTYPES = {'argument_parse_error', 'structured:parsing', 'model_output_parse_error'}
_MODEL_TERMINATIONS = {'max_steps_exceeded', 'nudges_exhausted', 'parse_error_exhausted', 'empty_answer'}


_USABLE_STATUSES = frozenset({'success', 'fallback', 'degraded'})


def _match_any(patterns: Dict[str, Pattern[str]], text: str) -> Optional[str]:
    for pattern_id, pattern in patterns.items():
        if pattern.search(text):
            return pattern_id
    return None


def attribute(finding: Finding, rules: CompiledRules) -> Tuple[Attribution, str, Optional[Confidence]]:
    """Return ``(attribution, rule_id, confidence override)`` for one finding; first matching rule wins."""
    category, subtype, text = finding.category, finding.subtype, finding.text or ''

    if category in INFORMATIONAL_CATEGORIES:
        return Attribution.MODEL, 'info', None
    if category == FindingCategory.TOOL_FAILURE:
        return _attribute_tool_failure(subtype, text, rules)
    if category == FindingCategory.TERMINATION:
        if subtype == 'content_filter':
            return Attribution.GUARD, 'guard.content_filter', None
        if subtype in _MODEL_TERMINATIONS:
            return Attribution.MODEL, 'model.behavior', None
        if subtype == 'context_overflow':
            return Attribution.MODEL, 'model.context_overflow', Confidence.LOW
        if subtype.startswith('score_'):
            return Attribution.UNKNOWN, 'score.excluded', None
        return Attribution.UNKNOWN, 'fallback', None
    if category == FindingCategory.REFUSAL:
        return Attribution.MODEL, 'model.refusal', None
    if category in (
        FindingCategory.REPETITION,
        FindingCategory.TAINTED_SIDE_EFFECT,
        FindingCategory.ATTACK_GOAL_ACTION,
        FindingCategory.RISKY_COMMAND,
    ):
        return Attribution.MODEL, 'model.behavior', None
    return Attribution.UNKNOWN, 'fallback', None


def _attribute_tool_failure(
    subtype: str, text: str, rules: CompiledRules
) -> Tuple[Attribution, str, Optional[Confidence]]:
    if subtype == 'structured:timeout':
        return Attribution.ENVIRONMENT, 'env.timeout', None
    env_rule = _match_any(rules.environment, text)
    if env_rule is not None:
        return Attribution.ENVIRONMENT, f'env.{env_rule}', None
    if subtype in ('structured:permission', 'structured:approval'):
        return Attribution.GUARD, 'guard.permission', None
    guard_rule = _match_any(rules.guard, text)
    if guard_rule is not None:
        return Attribution.GUARD, f'guard.{guard_rule}', None
    if subtype == 'command_nonzero_exit':
        return Attribution.MODEL, 'model.command_error', Confidence.LOW
    if subtype in _PARSE_SUBTYPES:
        return Attribution.MODEL, 'model.parse', None
    if subtype in _UNKNOWN_TOOL_SUBTYPES or (subtype == 'structured:unknown' and 'is not registered' in text):
        return Attribution.MODEL, 'model.unknown_tool', None
    if subtype == 'provider:formatter_error':
        return Attribution.ENVIRONMENT, 'env.harness', None
    if _BAD_ARGUMENT.search(text):
        return Attribution.MODEL, 'model.bad_arguments', Confidence.MEDIUM
    if _TRACEBACK.search(text):
        return Attribution.ENVIRONMENT, 'env.tool_internal', Confidence.LOW
    return Attribution.UNKNOWN, 'fallback', None


def apply_attribution(findings: List[Finding], rules: CompiledRules) -> None:
    """Set attribution, rule id and (possibly lowered) confidence on every finding in place."""
    for finding in findings:
        attribution, rule_id, confidence = attribute(finding, rules)
        finding.attribution = attribution
        finding.rule_id = rule_id
        if confidence is not None:
            finding.confidence = confidence


def resolve_outcome(
    view: TraceView,
    signals: ProviderSignals,
    config: TraceAnalysisConfig,
    findings: List[Finding],
    metric_directions: Optional[Dict[str, str]] = None,
    correctness_metrics: AbstractSet[str] = frozenset(),
) -> Outcome:
    """Use provider outcomes, explicit thresholds, then the main score's binary fallback."""
    refused = any(f.category == FindingCategory.REFUSAL for f in findings)
    if signals.outcome is not None and view.score_status in _USABLE_STATUSES:
        outcome = signals.outcome.model_copy(update={'source': 'provider'})
        if outcome.refused is None:
            outcome.refused = refused or None
        if outcome.task_success is not None:
            return outcome
    else:
        outcome = Outcome(refused=True if refused else None)
    if (view.score_status or 'success').lower() not in _USABLE_STATUSES or not view.score_value:
        return outcome
    main = view.main_score_name or next(iter(view.score_value))
    directions = metric_directions or {}
    explicit_direction = config.outcome_directions.get(view.ref.benchmark)
    thresholds = {
        name: limit
        for name, limit in (config.outcome_thresholds.get(view.ref.benchmark) or {}).items()
        if name in view.score_value
    }
    metric = main if main in thresholds else next(iter(thresholds)) if len(thresholds) == 1 else main
    direction = explicit_direction or directions.get(metric, 'higher_is_better')
    if direction == 'none':
        return outcome
    if metric in thresholds:
        value, threshold = view.score_value[metric], thresholds[metric]
        outcome.task_success = value <= threshold if direction == 'lower_is_better' else value >= threshold
        outcome.source = 'threshold'
    elif (
        signals.outcome is None
        and not thresholds
        and (
            main in correctness_metrics
            or (main not in directions and main in {'accuracy', 'acc', 'exact_match', 'em', 'is_correct'})
        )
    ):
        value = view.score_value.get(main)
        if value in (0.0, 1.0):
            outcome.task_success = value == (0.0 if direction == 'lower_is_better' else 1.0)
            outcome.source = 'binary_score'
    return outcome


__all__ = ['apply_attribution', 'attribute', 'resolve_outcome']
