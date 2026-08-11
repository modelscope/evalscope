"""Metric identity normalization and v1 migration.

New producers must pass canonical names. The permissive rules in this module are reserved for
built-in adapter output and historical reports; they are deliberately not used by the v2
resolver.
"""

import re
from typing import Dict, Optional, Tuple

from evalscope.api.metric.semantics import MetricIdentity, Scalar

_EXACT_ALIASES = {
    'acc': 'accuracy',
    'AverageAccuracy': 'accuracy',
    'WeightedAverageAccuracy': 'accuracy',
    'f1_score': 'f1',
    'F1': 'f1',
    'em': 'exact_match',
    'winrate': 'win_rate',
    'BLEU': 'bleu',
    'Rouge': 'rouge',
    'Rouge-L': 'rouge',
    'ROUGE_L': 'rouge',
    'METEOR': 'meteor',
    'CIDEr': 'cider',
    'IoU': 'iou',
    'mean_IoU': 'iou',
    'score': 'normalized_score',
    'overall': 'normalized_score',
    'total_score': 'judge_score',
    'gpt_score': 'judge_score',
    'avg_score': 'judge_score',
    'HalluRate': 'hallucination_rate',
    'total_wall_time_s': 'total_wall_time',
    'total_model_time_s': 'total_model_time',
    'total_tool_time_s': 'total_tool_time',
    'total_other_time_s': 'total_other_time',
    'HPSv2.1Score': 'hps_v2_1_score',
    'PickScore': 'pick_score',
    'VQAScore': 'vqa_score',
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
_BLEU_N = re.compile(r'^(?:mean_)?[Bb]leu_(?P<ngram>\d+)$')
_THRESHOLD_ACC = re.compile(r'^(?:mean_)?ACC@(?P<threshold>\d+(?:\.\d+)?)$')
_SCOPE_METRIC = re.compile(r'^(?P<scope>[^/]+)/(?P<name>[^/]+)$')
_K_AGGREGATION = re.compile(r'^(?P<kind>avg|mean|pass|max|vote)@(?P<k>\d+)$')
_SNAKE_BOUNDARY = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_NON_NAME = re.compile(r'[^a-z0-9]+')


def _snake_case(value: str) -> str:
    value = _SNAKE_BOUNDARY.sub('_', value).lower()
    return _NON_NAME.sub('_', value).strip('_')


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

    bleu = _BLEU_N.fullmatch(name)
    if bleu:
        dimensions.setdefault('ngram', int(bleu.group('ngram')))
        return 'bleu'

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

    if benchmark_name == 'longmemeval':
        match = re.fullmatch(r'(?P<scope>.+)_acc', raw_name)
        if match:
            scope = match.group('scope')
            raw_name = 'accuracy'
            if scope == 'overall':
                identity_dimensions.setdefault('scope', 'overall')
                raw_aggregation = 'mean'
            elif scope == 'task_averaged':
                identity_dimensions.setdefault('scope', 'question_types')
                raw_aggregation = 'macro_mean'
            else:
                identity_dimensions.setdefault('question_type', _snake_case(scope))
                raw_aggregation = 'mean'

    if benchmark_name == 'locomo':
        match = re.fullmatch(r'(?P<scope>.+)_f1', raw_name)
        if match:
            scope = match.group('scope')
            raw_name = 'f1'
            if scope == 'overall':
                identity_dimensions.setdefault('scope', 'overall')
                raw_aggregation = 'mean'
            elif scope == 'task_averaged':
                identity_dimensions.setdefault('scope', 'question_types')
                raw_aggregation = 'macro_mean'
            else:
                identity_dimensions.setdefault('question_type', _snake_case(scope))
                raw_aggregation = 'mean'

    if benchmark_name == 'openai_mrcr':
        if raw_name == 'overall_mrcr_score':
            raw_name = 'mrcr_score'
            raw_aggregation = 'mean'
            identity_dimensions.setdefault('scope', 'overall')
        else:
            token_range = re.fullmatch(r'(?P<minimum>\d+)-(?P<maximum>\d+)_mrcr_score', raw_name)
            if token_range:
                raw_name = 'mrcr_score'
                raw_aggregation = 'mean'
                identity_dimensions.setdefault('min_tokens', int(token_range.group('minimum')))
                identity_dimensions.setdefault('max_tokens', int(token_range.group('maximum')))

    if benchmark_name == 'wide_search':
        wide_search = re.fullmatch(r'(?P<kind>avg|pass|max)@(?P<k>\d+)_(?P<scope>[^/]+)/(?P<metric>[^/]+)', raw_name)
        if wide_search:
            raw_name = wide_search.group('metric')
            raw_aggregation = {
                'avg': 'mean',
                'pass': 'pass_at_k',
                'max': 'max',
            }[wide_search.group('kind')]
            identity_dimensions.setdefault('k', int(wide_search.group('k')))
            identity_dimensions.setdefault('scope', _snake_case(wide_search.group('scope')))
            if raw_name.startswith(('row_', 'item_')):
                target, raw_name = raw_name.split('_', 1)
                identity_dimensions.setdefault('target', target)

    scope_match = _SCOPE_METRIC.fullmatch(raw_name)
    if scope_match:
        identity_dimensions.setdefault('scope', _snake_case(scope_match.group('scope')))
        raw_name = scope_match.group('name')

    hallusion = re.fullmatch(r'(?P<level>.+)_(?P<target>[afq])Acc', raw_name)
    if benchmark_name == 'hallusion_bench' and hallusion:
        target = {'a': 'answer', 'f': 'figure', 'q': 'question'}[hallusion.group('target')]
        identity_dimensions.setdefault('level', _snake_case(hallusion.group('level')))
        identity_dimensions.setdefault('target', target)
        raw_name = 'accuracy'
        raw_aggregation = 'mean'

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


def legacy_aliases() -> Tuple[str, ...]:
    """Exact aliases whose non-snake spelling is accepted only on migration paths."""
    return tuple(_EXACT_ALIASES)


def is_known_dynamic_legacy_name(metric_name: str, benchmark_name: Optional[str] = None) -> bool:
    """Whether a non-catalogued v1 name belongs to a supported structured family."""
    if _DYNAMIC_K.fullmatch(metric_name) or _DYNAMIC_HAT_K.fullmatch(metric_name):
        return True
    if _BLEU_N.fullmatch(metric_name) or _THRESHOLD_ACC.fullmatch(metric_name):
        return True
    scope_metric = _SCOPE_METRIC.fullmatch(metric_name)
    if scope_metric and scope_metric.group('name') in {'success_rate', 'precision', 'recall', 'f1'}:
        return True
    benchmark_patterns = {
        'hallusion_bench': r'.+_[afq]Acc',
        'longmemeval': r'.+_acc',
        'locomo': r'.+_f1',
        'openai_mrcr': r'(?:overall|\d+-\d+)_mrcr_score',
        'wide_search': r'(?:avg|pass|max)@\d+_[^/]+/[^/]+',
    }
    pattern = benchmark_patterns.get(benchmark_name or '')
    return bool(pattern and re.fullmatch(pattern, metric_name))
