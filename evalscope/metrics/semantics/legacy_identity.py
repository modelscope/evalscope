"""Read-old metric naming: historical spellings rewritten into v2 identities.

Everything here is permissive by design and is reserved for built-in adapter output and stored v1
reports. New producers go through ``evalscope.metrics.semantics.naming``, which never reassigns an
ambiguous name.

``_BENCHMARK_RULES`` declares one pattern per benchmark shape, and both :func:`migrate_legacy_identity`
and :func:`is_known_legacy_spelling` derive from it. Declaring it once is what keeps the rewrite and
the membership gate from disagreeing about whether a spelling is supported.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Match, Optional, Pattern, Tuple

from evalscope.api.metric.semantics import MetricIdentity, Scalar
from evalscope.metrics.semantics.aliases import METRIC_ALIASES, AliasScope, aliases_in_scope
from evalscope.metrics.semantics.naming import (
    BLEU_N,
    ROUGE_VARIANT,
    canonical_aggregation,
    canonical_overlap_name,
    snake_case,
)

__all__ = [
    'LegacyNameRule',
    'Stage',
    'canonical_metric_list_name',
    'is_known_legacy_spelling',
    'migrate_legacy_identity',
]

#: Punctuation the v1 names used to encode a structural axis. Read here only: a new producer
#: declares these axes through ``aggregation`` and ``dimensions`` instead of spelling them.
_DYNAMIC_K = re.compile(r'^(?P<name>.+?)_(?P<kind>pass|vote)@(?P<k>\d+)$')
_DYNAMIC_HAT_K = re.compile(r'^(?P<name>.+?)_pass\^(?P<k>\d+)$')
_THRESHOLD_ACC = re.compile(r'^(?:mean_)?ACC@(?P<threshold>\d+(?:\.\d+)?)$')
_SCOPE_METRIC = re.compile(r'^(?P<scope>[^/]+)/(?P<name>[^/]+)$')
_K_AGGREGATION = re.compile(r'^(?P<kind>avg|mean|pass|max|vote)@(?P<k>\d+)$')


class Stage(str, Enum):
    """When a benchmark rule runs, relative to the generic ``scope/metric`` split."""

    RAW_NAME = 'raw_name'
    """Before the split, so the rule sees the stored name including any scope prefix."""

    AFTER_SCOPE_SPLIT = 'after_scope_split'
    """After the split. ``Overall_aAcc`` under a scope must lose the scope first, otherwise the
    level would absorb it."""


@dataclass(frozen=True)
class LegacyNameRule:
    """How one benchmark's historical metric-name shape maps onto a v2 identity."""

    benchmarks: Tuple[str, ...]
    """Benchmarks that spell their metrics this way."""

    pattern: Pattern[str]
    """Full-match pattern over the metric name, read by both the rewrite and the membership gate."""

    apply: Callable[[Match[str], Dict[str, Scalar], str], Tuple[str, str]]
    """``(match, dimensions, aggregation) -> (raw_name, aggregation)``, mutating ``dimensions``."""

    examples: Tuple[str, ...] = ()
    """Spellings this rule claims, sampled by the tests that pin its behaviour."""

    stage: Stage = Stage.RAW_NAME


def _scoped_suffix_rule(benchmark: str, suffix: str, canonical_name: str, examples: Tuple[str, ...]) -> LegacyNameRule:
    """Build the ``{scope}_{suffix}`` rule shared by longmemeval and locomo.

    Both benchmarks report one metric per question type plus two roll-ups, spelled as a prefix on
    the metric name. ``overall`` is a plain mean over all questions, ``task_averaged`` is a macro
    mean over the question types, and anything else names a single question type.

    Args:
        benchmark: Benchmark the rule belongs to.
        suffix: Metric suffix the benchmark uses (``acc`` / ``f1``).
        canonical_name: Canonical metric name the suffix stands for.
        examples: Spellings for the tests to sample.

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
        dimensions.setdefault('question_type', snake_case(scope))
        return canonical_name, 'mean'

    return LegacyNameRule(
        benchmarks=(benchmark,),
        pattern=re.compile(rf'(?P<scope>.+)_{suffix}'),
        apply=apply,
        examples=examples,
    )


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
    dimensions.setdefault('scope', snake_case(match.group('scope')))
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
        dimensions.setdefault('level', snake_case(level))
    dimensions.setdefault('target', {'a': 'answer', 'f': 'figure', 'q': 'question'}[match.group('target')])
    return 'accuracy', 'mean'


#: One rule per historical metric-name shape.
LEGACY_NAME_RULES: Tuple[LegacyNameRule, ...] = (
    LegacyNameRule(
        benchmarks=('general_qa', 'general_vqa'),
        pattern=re.compile(rf'(?:{BLEU_N.pattern}|{ROUGE_VARIANT.pattern})'),
        apply=_apply_overlap_aggregation,
        examples=('Bleu_4', 'Rouge-L-R'),
    ),
    _scoped_suffix_rule(
        'longmemeval', 'acc', 'accuracy', ('overall_acc', 'task_averaged_acc', 'single_session_user_acc')
    ),
    _scoped_suffix_rule('locomo', 'f1', 'f1', ('overall_f1', 'task_averaged_f1', 'multi_hop_f1')),
    LegacyNameRule(
        benchmarks=('omni_doc_bench',),
        pattern=re.compile(r'(?P<metric>.+)_(?P<language>EN|CH)'),
        apply=_apply_language_suffix,
        examples=('table_TEDS_EN', 'overall_CH'),
    ),
    LegacyNameRule(
        benchmarks=('openai_mrcr',),
        pattern=re.compile(r'(?:overall|(?P<minimum>\d+)-(?P<maximum>\d+))_mrcr_score'),
        apply=_apply_mrcr_scope,
        examples=('overall_mrcr_score', '4096-8192_mrcr_score'),
    ),
    LegacyNameRule(
        benchmarks=('wide_search',),
        pattern=re.compile(r'(?P<kind>avg|pass|max)@(?P<k>\d+)_(?P<scope>[^/]+)/(?P<metric>[^/]+)'),
        apply=_apply_wide_search,
        examples=('avg@4_row/f1', 'pass@2_item/precision', 'max@8_Scope Name/success_rate'),
    ),
    LegacyNameRule(
        benchmarks=('hallusion_bench',),
        pattern=re.compile(r'(?:(?P<level>.+)_)?(?P<target>[afq])Acc'),
        apply=_apply_hallusion_target,
        examples=('Overall_aAcc', 'Easy_qAcc', 'aAcc', 'fAcc', 'qAcc'),
        stage=Stage.AFTER_SCOPE_SPLIT,
    ),
)

_BENCHMARK_RULES: Dict[str, LegacyNameRule] = {
    benchmark: rule for rule in LEGACY_NAME_RULES for benchmark in rule.benchmarks
}


def _apply_alias(name: str, dimensions: Dict[str, Scalar]) -> Optional[str]:
    """Resolve one exact alias, adding any axis its spelling encodes."""
    alias = METRIC_ALIASES.get(name)
    if alias is None:
        return None
    for key, value in alias.dimensions.items():
        dimensions.setdefault(key, value)
    return alias.canonical_name


def _canonical_base_name(name: str, dimensions: Dict[str, Scalar]) -> str:
    """Canonicalize a stored metric name, extracting the axes its spelling encodes."""
    explicit = _apply_alias(name, dimensions)
    if explicit:
        return explicit

    overlap_name = canonical_overlap_name(name, dimensions)
    if overlap_name is not None:
        return overlap_name

    threshold = _THRESHOLD_ACC.fullmatch(name)
    if threshold:
        dimensions.setdefault('threshold', float(threshold.group('threshold')))
        return 'accuracy'

    if name.startswith('mean_'):
        name = name[5:]

    explicit = _apply_alias(name, dimensions)
    if explicit:
        return explicit

    snake_name = snake_case(name)
    return _apply_alias(snake_name, dimensions) or snake_name


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
    if rule is not None and rule.stage is Stage.RAW_NAME:
        match = rule.pattern.fullmatch(raw_name)
        if match:
            raw_name, raw_aggregation = rule.apply(match, identity_dimensions, raw_aggregation)

    scope_match = _SCOPE_METRIC.fullmatch(raw_name)
    if scope_match:
        identity_dimensions.setdefault('scope', snake_case(scope_match.group('scope')))
        raw_name = scope_match.group('name')

    if rule is not None and rule.stage is Stage.AFTER_SCOPE_SPLIT:
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

    return MetricIdentity(
        name=_canonical_base_name(raw_name, identity_dimensions),
        aggregation=canonical_aggregation(raw_aggregation),
        dimensions=identity_dimensions,
    )


_METRIC_LIST_ALIASES = aliases_in_scope(AliasScope.METRIC_LIST)


def canonical_metric_list_name(raw_name: str, benchmark_name: str) -> Optional[str]:
    """Canonicalize one ``BenchmarkMeta.metric_list`` entry, or return ``None`` to leave it alone.

    Only the ``METRIC_LIST`` alias scope is rewritten, whose contract is that the canonical name
    still resolves through ``get_metric()``. Which spellings qualify is alias knowledge, so the
    adapter boundary asks this instead of carrying its own list.

    Args:
        raw_name: Metric name as the benchmark declared it.
        benchmark_name: Benchmark being constructed, for benchmark-scoped rules.

    Returns:
        The canonical name, or ``None`` when the spelling is not a rewritable alias.
    """
    if raw_name not in _METRIC_LIST_ALIASES:
        return None
    return migrate_legacy_identity(raw_name, 'identity', benchmark_name=benchmark_name).name


def is_known_legacy_spelling(metric_name: str, benchmark_name: Optional[str] = None) -> bool:
    """Whether a non-catalogued v1 name belongs to a supported structured family."""
    if _DYNAMIC_K.fullmatch(metric_name) or _DYNAMIC_HAT_K.fullmatch(metric_name):
        return True
    if BLEU_N.fullmatch(metric_name) or ROUGE_VARIANT.fullmatch(metric_name) or _THRESHOLD_ACC.fullmatch(metric_name):
        return True
    scope_metric = _SCOPE_METRIC.fullmatch(metric_name)
    if scope_metric and scope_metric.group('name') in {'success_rate', 'precision', 'recall', 'f1'}:
        return True
    rule = _BENCHMARK_RULES.get(benchmark_name or '')
    return rule is not None and rule.pattern.fullmatch(metric_name) is not None
