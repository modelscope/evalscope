"""Metric identity normalization and v1 migration.

New producers must pass canonical names. The permissive rules in this module are reserved for
built-in adapter output and historical reports; they are deliberately not used by the v2
resolver.
"""

import re
from typing import Callable, Dict, Match, NamedTuple, Optional, Pattern, Tuple

from evalscope.api.metric.semantics import MetricIdentity, Scalar
from evalscope.metrics.semantics.legacy import LEGACY_METRIC_ALIASES

_EXACT_ALIASES = {name: alias.canonical_name for name, alias in LEGACY_METRIC_ALIASES.items()}

# Producer-side aliases are intentionally narrow. These spellings are exact synonyms; mappings
# that reinterpret an ambiguous score remain exclusive to v1 report migration.
_SAFE_PRODUCER_ALIASES = {
    'acc': 'accuracy',
    'bertscore': 'bert_score',
    'f1_score': 'f1',
    'em': 'exact_match',
}

_AGGREGATION_ALIASES = {
    'avg': 'mean',
    'average': 'mean',
    'macro': 'macro_mean',
    'micro': 'micro_mean',
    'weighted': 'weighted_mean',
    '': 'identity',
}

_DYNAMIC_K = re.compile(r'^(?P<name>.+?)_(?P<kind>pass|vote)@(?P<k>\d+)$')
_DYNAMIC_HAT_K = re.compile(r'^(?P<name>.+?)_pass\^(?P<k>\d+)$')
_BLEU_N = re.compile(r'^(?:mean_)?[Bb]leu[-_](?P<ngram>\d+)$')
_ROUGE_VARIANT = re.compile(r'^(?:mean_)?Rouge-(?P<variant>[12L])-(?P<statistic>[RPF])$')
_THRESHOLD_ACC = re.compile(r'^(?:mean_)?ACC@(?P<threshold>\d+(?:\.\d+)?)$')
_SCOPE_METRIC = re.compile(r'^(?P<scope>[^/]+)/(?P<name>[^/]+)$')
_K_AGGREGATION = re.compile(r'^(?P<kind>avg|mean|pass|max|vote)@(?P<k>\d+)$')
_SNAKE_BOUNDARY = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_NON_NAME = re.compile(r'[^a-z0-9]+')


def _snake_case(value: str) -> str:
    value = _SNAKE_BOUNDARY.sub('_', value).lower()
    return _NON_NAME.sub('_', value).strip('_')


def _canonical_overlap_name(name: str, dimensions: Dict[str, Scalar]) -> Optional[str]:
    """Return the canonical identity for an unambiguous BLEU or ROUGE spelling."""
    bleu = _BLEU_N.fullmatch(name)
    if bleu:
        dimensions.setdefault('ngram', int(bleu.group('ngram')))
        return 'bleu'

    rouge = _ROUGE_VARIANT.fullmatch(name)
    if rouge:
        variant = rouge.group('variant')
        if variant == 'L':
            dimensions.setdefault('variant', 'l')
        else:
            dimensions.setdefault('ngram', int(variant))
        dimensions.setdefault(
            'statistic',
            {
                'R': 'recall',
                'P': 'precision',
                'F': 'f1',
            }[rouge.group('statistic')],
        )
        return 'rouge'

    return None


def canonicalize_producer_identity(
    metric_name: str,
    aggregation: Optional[str],
    dimensions: Optional[Dict[str, Scalar]] = None,
) -> MetricIdentity:
    """Canonicalize producer syntax without inferring what a metric measures.

    New producers must express structural axes through ``aggregation`` and ``dimensions``.
    Ambiguous or empty names are kept reportable under ``legacy_metric`` with their original
    spelling, so resolving them can only produce diagnostic semantics.
    """
    original_name = metric_name
    raw_aggregation = aggregation or 'identity'
    canonical_aggregation = _AGGREGATION_ALIASES.get(raw_aggregation, _snake_case(raw_aggregation))
    identity_dimensions = dict(dimensions or {})
    canonical_name = _canonical_overlap_name(metric_name, identity_dimensions)
    if canonical_name is None:
        snake_name = _snake_case(metric_name)
        canonical_name = _SAFE_PRODUCER_ALIASES.get(snake_name, snake_name)

    try:
        MetricIdentity(name=canonical_name, aggregation='identity')
    except ValueError:
        canonical_name = 'legacy_metric'
        identity_dimensions['original_name'] = original_name

    return MetricIdentity(
        name=canonical_name,
        aggregation=canonical_aggregation,
        dimensions=identity_dimensions,
    )


class _BenchmarkRule(NamedTuple):
    """How one benchmark's historical metric-name shape maps onto a v2 identity.

    ``pattern`` is the single source of truth for both jobs this knowledge is needed for:
    :func:`migrate_legacy_identity` uses its match groups to rewrite the identity, and
    :func:`is_known_dynamic_legacy_name` uses the same pattern to decide whether a non-catalogued
    spelling belongs to a supported family. Declaring it once is what keeps the two from drifting.
    """

    pattern: Pattern[str]
    """Full-match pattern over the raw metric name."""

    apply: Callable[[Match[str], Dict[str, Scalar], str], Tuple[str, str]]
    """``(match, dimensions, aggregation) -> (raw_name, aggregation)``, mutating ``dimensions``."""

    before_scope_split: bool = True
    """Whether the rule runs before the generic ``scope/metric`` split.

    ``hallusion_bench`` runs after it: a name such as ``scope/Overall_aAcc`` must have its scope
    stripped first, otherwise the level would absorb the scope prefix.
    """


def _scoped_suffix_rule(suffix: str, canonical_name: str) -> _BenchmarkRule:
    """Build the ``{scope}_{suffix}`` rule shared by longmemeval and locomo.

    Both benchmarks report one metric per question type plus two roll-ups, spelled as a prefix on
    the metric name. ``overall`` is a plain mean over all questions, ``task_averaged`` is a macro
    mean over the question types, and anything else names a single question type.

    Args:
        suffix: Metric suffix the benchmark uses (``acc`` / ``f1``).
        canonical_name: Canonical metric name the suffix stands for.

    Returns:
        The rule for that benchmark.
    """

    def apply(match: Match[str], dimensions: Dict[str, Scalar], aggregation: str) -> Tuple[str, str]:
        scope = match.group('scope')
        if scope == 'overall':
            dimensions.setdefault('scope', 'overall')
            return canonical_name, 'mean'
        if scope == 'task_averaged':
            dimensions.setdefault('scope', 'question_types')
            return canonical_name, 'macro_mean'
        dimensions.setdefault('question_type', _snake_case(scope))
        return canonical_name, 'mean'

    return _BenchmarkRule(pattern=re.compile(rf'(?P<scope>.+)_{suffix}'), apply=apply)


def _apply_overlap_aggregation(match: Match[str], dimensions: Dict[str, Scalar], aggregation: str) -> Tuple[str, str]:
    """General-QA/VQA: recover the aggregation, keeping the name for ``_canonical_base_name``.

    Historical reports stored post-aggregation overlap metric names without the ``mean_`` prefix,
    so the aggregation has to come from the benchmark contract instead of the spelling.
    """
    return match.string, 'mean' if aggregation == 'identity' else aggregation


def _apply_language_suffix(match: Match[str], dimensions: Dict[str, Scalar], aggregation: str) -> Tuple[str, str]:
    """OmniDocBench: the legacy TSV evaluator reports every metric once per language.

    Language is an axis of the same metric, so the ``_EN`` / ``_CH`` suffix becomes a dimension and
    the remaining stem is left to ``_canonical_base_name``, which snake-cases ``table_TEDS`` into
    ``table_teds`` and aliases ``overall`` to ``normalized_score``. This mirrors the adapter's own
    LEGACY_METRIC_NAMES mapping, so a migrated report and a fresh run produce the same identities.
    """
    dimensions.setdefault('language', match.group('language').lower())
    return match.group('metric'), aggregation


def _apply_mrcr_scope(match: Match[str], dimensions: Dict[str, Scalar], aggregation: str) -> Tuple[str, str]:
    """OpenAI-MRCR: either the overall roll-up or one context-length bucket."""
    if match.group('minimum') is None:
        dimensions.setdefault('scope', 'overall')
    else:
        dimensions.setdefault('min_tokens', int(match.group('minimum')))
        dimensions.setdefault('max_tokens', int(match.group('maximum')))
    return 'mrcr_score', 'mean'


def _apply_wide_search(match: Match[str], dimensions: Dict[str, Scalar], aggregation: str) -> Tuple[str, str]:
    """WideSearch: ``{kind}@{k}_{scope}/{metric}``, optionally prefixed by a row/item target."""
    raw_name = match.group('metric')
    dimensions.setdefault('k', int(match.group('k')))
    dimensions.setdefault('scope', _snake_case(match.group('scope')))
    if raw_name.startswith(('row_', 'item_')):
        target, raw_name = raw_name.split('_', 1)
        dimensions.setdefault('target', target)
    return raw_name, {'avg': 'mean', 'pass': 'pass_at_k', 'max': 'max'}[match.group('kind')]


def _apply_hallusion_target(match: Match[str], dimensions: Dict[str, Scalar], aggregation: str) -> Tuple[str, str]:
    """HallusionBench: accuracy per aggregation bucket and per scoring target.

    The level prefix is optional because the benchmark spells both forms: ``Overall_aAcc`` in a
    stored report and the bare ``aAcc`` in ``metric_list``. Without the bare form the three
    targets would all degrade to the same ``accuracy`` identity and lose which one they measure.
    """
    level = match.group('level')
    if level:
        dimensions.setdefault('level', _snake_case(level))
    dimensions.setdefault('target', {'a': 'answer', 'f': 'figure', 'q': 'question'}[match.group('target')])
    return 'accuracy', 'mean'


#: Benchmark -> its historical metric-name rule. One entry per benchmark, one pattern per entry.
_BENCHMARK_RULES: Dict[str, _BenchmarkRule] = {
    'general_qa': _BenchmarkRule(
        pattern=re.compile(rf'(?:{_BLEU_N.pattern}|{_ROUGE_VARIANT.pattern})'),
        apply=_apply_overlap_aggregation,
    ),
    'general_vqa': _BenchmarkRule(
        pattern=re.compile(rf'(?:{_BLEU_N.pattern}|{_ROUGE_VARIANT.pattern})'),
        apply=_apply_overlap_aggregation,
    ),
    'longmemeval': _scoped_suffix_rule('acc', 'accuracy'),
    'locomo': _scoped_suffix_rule('f1', 'f1'),
    'omni_doc_bench': _BenchmarkRule(
        pattern=re.compile(r'(?P<metric>.+)_(?P<language>EN|CH)'),
        apply=_apply_language_suffix,
    ),
    'openai_mrcr': _BenchmarkRule(
        pattern=re.compile(r'(?:overall|(?P<minimum>\d+)-(?P<maximum>\d+))_mrcr_score'),
        apply=_apply_mrcr_scope,
    ),
    'wide_search': _BenchmarkRule(
        pattern=re.compile(r'(?P<kind>avg|pass|max)@(?P<k>\d+)_(?P<scope>[^/]+)/(?P<metric>[^/]+)'),
        apply=_apply_wide_search,
    ),
    'hallusion_bench': _BenchmarkRule(
        pattern=re.compile(r'(?:(?P<level>.+)_)?(?P<target>[afq])Acc'),
        apply=_apply_hallusion_target,
        before_scope_split=False,
    ),
}


def _canonical_base_name(name: str, dimensions: Dict[str, Scalar]) -> str:
    exact_match_targets = {
        'Act.EM': 'action',
        'Plan.EM': 'plan',
    }
    if name in exact_match_targets:
        dimensions.setdefault('target', exact_match_targets[name])
        return 'exact_match'

    explicit = _EXACT_ALIASES.get(name)
    if explicit:
        return explicit

    overlap_name = _canonical_overlap_name(name, dimensions)
    if overlap_name is not None:
        return overlap_name

    threshold = _THRESHOLD_ACC.fullmatch(name)
    if threshold:
        dimensions.setdefault('threshold', float(threshold.group('threshold')))
        return 'accuracy'

    if name.startswith('mean_'):
        name = name[5:]

    explicit = _EXACT_ALIASES.get(name)
    if explicit:
        return explicit

    snake_name = _snake_case(name)
    aliases = {
        'average_accuracy': 'accuracy',
        'f_1': 'f1',
        'rouge_l': 'rouge',
        'center_acc': 'accuracy',
        'a_acc': 'accuracy',
        'f_acc': 'accuracy',
        'q_acc': 'accuracy',
    }
    return aliases.get(snake_name, snake_name)


def migrate_legacy_identity(
    metric_name: str,
    aggregation: Optional[str],
    dimensions: Optional[Dict[str, Scalar]] = None,
    benchmark_name: Optional[str] = None,
) -> MetricIdentity:
    """Convert known built-in/v1 spelling into a v2 identity.

    This function is intentionally explicit at the structural boundaries: dynamic ``k``,
    thresholds, scope, and the Hallusion levels become dimensions instead of punctuation in a
    name. Callers decide whether a legacy spelling is trusted before invoking it.
    """
    identity_dimensions = dict(dimensions or {})
    raw_name = metric_name
    raw_aggregation = aggregation or 'identity'

    rule = _BENCHMARK_RULES.get(benchmark_name or '')
    if rule is not None and rule.before_scope_split:
        match = rule.pattern.fullmatch(raw_name)
        if match:
            raw_name, raw_aggregation = rule.apply(match, identity_dimensions, raw_aggregation)

    scope_match = _SCOPE_METRIC.fullmatch(raw_name)
    if scope_match:
        identity_dimensions.setdefault('scope', _snake_case(scope_match.group('scope')))
        raw_name = scope_match.group('name')

    if rule is not None and not rule.before_scope_split:
        match = rule.pattern.fullmatch(raw_name)
        if match:
            raw_name, raw_aggregation = rule.apply(match, identity_dimensions, raw_aggregation)

    dynamic = _DYNAMIC_K.fullmatch(raw_name)
    if dynamic:
        raw_name = dynamic.group('name')
        raw_aggregation = 'pass_at_k' if dynamic.group('kind') == 'pass' else 'vote_at_k'
        identity_dimensions.setdefault('k', int(dynamic.group('k')))

    dynamic_hat = _DYNAMIC_HAT_K.fullmatch(raw_name)
    if dynamic_hat:
        raw_name = dynamic_hat.group('name')
        raw_aggregation = 'pass_hat_k'
        identity_dimensions.setdefault('k', int(dynamic_hat.group('k')))

    aggregation_k = _K_AGGREGATION.fullmatch(raw_aggregation)
    if aggregation_k:
        kind = aggregation_k.group('kind')
        raw_aggregation = {
            'avg': 'mean',
            'mean': 'mean',
            'pass': 'pass_at_k',
            'vote': 'vote_at_k',
            'max': 'max',
        }[kind]
        identity_dimensions.setdefault('k', int(aggregation_k.group('k')))

    if raw_name.startswith('mean_') and raw_aggregation in ('', 'identity'):
        raw_aggregation = 'mean'

    canonical_name = _canonical_base_name(raw_name, identity_dimensions)
    canonical_aggregation = _AGGREGATION_ALIASES.get(raw_aggregation, _snake_case(raw_aggregation))
    return MetricIdentity(
        name=canonical_name,
        aggregation=canonical_aggregation,
        dimensions=identity_dimensions,
    )


def is_known_dynamic_legacy_name(metric_name: str, benchmark_name: Optional[str] = None) -> bool:
    """Whether a non-catalogued v1 name belongs to a supported structured family."""
    if _DYNAMIC_K.fullmatch(metric_name) or _DYNAMIC_HAT_K.fullmatch(metric_name):
        return True
    if _BLEU_N.fullmatch(metric_name) or _ROUGE_VARIANT.fullmatch(metric_name) or _THRESHOLD_ACC.fullmatch(metric_name):
        return True
    scope_metric = _SCOPE_METRIC.fullmatch(metric_name)
    if scope_metric and scope_metric.group('name') in {'success_rate', 'precision', 'recall', 'f1'}:
        return True
    rule = _BENCHMARK_RULES.get(benchmark_name or '')
    return rule is not None and rule.pattern.fullmatch(metric_name) is not None
