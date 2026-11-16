"""Judgements on top of the findings: per-finding attribution, sample outcome, primary attribution, failure mode."""

import re
from typing import Dict, List, Optional, Pattern, Tuple

from evalscope.api.analysis import (
    INFORMATIONAL_CATEGORIES,
    Attribution,
    Confidence,
    FailureMode,
    Finding,
    FindingCategory,
    Outcome,
    ProviderSignals,
    TraceView,
)
from evalscope.api.analysis.config import TraceAnalysisConfig

from .detectors import CompiledRules

_BAD_ARGUMENT = re.compile(
    r'\b(?:TypeError|ValueError|KeyError|IndexError|ValidationError|validation errors? for|missing \d* ?required'
    r'|unexpected keyword argument|not found|does not exist|invalid (?:value|argument|id))\b',
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
    if category == FindingCategory.OUTCOME:
        return Attribution.UNKNOWN, 'outcome.unexplained', Confidence.LOW
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


def _order_key(finding: Finding) -> Tuple[int, int]:
    # Findings without a position (loop-level endings) sort after positioned ones.
    position = finding.first_position
    return (0, position) if position is not None else (1, 0)


def primary_attribution(findings: List[Finding]) -> Tuple[Optional[Attribution], Optional[Finding]]:
    """Pick the sample-level attribution.

    1. An environment finding that occurs before the first model finding wins.
    2. Otherwise the earliest high-confidence finding decides.
    3. Otherwise the earliest finding decides.
    Informational findings never count. Returns ``(None, None)`` when nothing remains.
    """
    relevant = sorted((f for f in findings if f.category not in INFORMATIONAL_CATEGORIES), key=_order_key)
    if not relevant:
        return None, None
    first_model = next((f for f in relevant if f.attribution == Attribution.MODEL), None)
    first_env = next((f for f in relevant if f.attribution == Attribution.ENVIRONMENT), None)
    if first_env is not None and (first_model is None or _order_key(first_env) < _order_key(first_model)):
        return Attribution.ENVIRONMENT, first_env
    high = next((f for f in relevant if f.confidence == Confidence.HIGH), None)
    chosen = high or relevant[0]
    return chosen.attribution, chosen


def failure_mode(findings: List[Finding], primary: Optional[Attribution]) -> FailureMode:
    """Approximate why a failed sample failed (refused / blocked by guard / model / environment / unknown)."""
    categories = {f.category for f in findings}
    if FindingCategory.REFUSAL in categories and not categories & {
        FindingCategory.SIDE_EFFECT,
        FindingCategory.RISKY_COMMAND,
    }:
        return 'refused'
    if any(f.attribution == Attribution.GUARD for f in findings):
        return 'blocked_by_guard'
    if primary == Attribution.ENVIRONMENT:
        return 'environment_failed'
    if primary == Attribution.MODEL:
        return 'model_failed'
    return 'unknown'


def resolve_outcome(
    view: TraceView, signals: ProviderSignals, config: TraceAnalysisConfig, findings: List[Finding]
) -> Outcome:
    """Provider outcome first, then a binary score (all values in {0, 1}), then a configured threshold."""
    refused = any(f.category == FindingCategory.REFUSAL for f in findings)
    if signals.outcome is not None:
        outcome = signals.outcome.model_copy(update={'source': 'provider'})
        if outcome.refused is None:
            outcome.refused = refused or None
        return outcome
    outcome = Outcome(refused=True if refused else None)
    if (view.score_status or 'success').lower() not in _USABLE_STATUSES or not view.score_value:
        return outcome
    if view.main_score_name in view.score_value:
        main = view.score_value[view.main_score_name]
    else:
        main = next(iter(view.score_value.values()))
    direction = config.outcome_directions.get(view.ref.benchmark, 'higher_is_better')
    if direction == 'none':
        return outcome
    if all(value in (0.0, 1.0) for value in view.score_value.values()):
        outcome.task_success, outcome.source = main == (0.0 if direction == 'lower_is_better' else 1.0), 'binary_score'
        return outcome
    for metric, threshold in (config.outcome_thresholds.get(view.ref.benchmark) or {}).items():
        if metric in view.score_value:
            outcome.task_success = (
                view.score_value[metric] <= threshold
                if direction == 'lower_is_better'
                else view.score_value[metric] >= threshold
            )
            outcome.source = 'threshold'
            break
    return outcome


__all__ = ['apply_attribution', 'attribute', 'failure_mode', 'primary_attribution', 'resolve_outcome']
