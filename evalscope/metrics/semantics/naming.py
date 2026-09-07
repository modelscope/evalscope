"""Metric identity naming: canonical spelling without inferring what a metric measures.

A new producer must express structural axes through ``aggregation`` and ``dimensions``, so nothing
here reassigns an ambiguous name -- that is exclusive to
``evalscope.metrics.semantics.legacy_identity``, which reads historical spellings. The two are kept
apart deliberately: they have opposite policies, and calling the permissive one from new code
silently invents semantics the producer never declared.
"""

import re
from typing import Dict, Optional

from evalscope.api.metric.semantics import MetricIdentity, Scalar
from evalscope.metrics.semantics.aliases import AliasScope, aliases_in_scope

_PRODUCER_ALIASES = aliases_in_scope(AliasScope.PRODUCER)

_AGGREGATION_ALIASES = {
    'avg': 'mean',
    'average': 'mean',
    'macro': 'macro_mean',
    'micro': 'micro_mean',
    'weighted': 'weighted_mean',
    '': 'identity',
}

#: Overlap-metric spellings, shared with the read-old path because both must read them alike.
BLEU_N = re.compile(r'^(?:mean_)?[Bb]leu[-_](?P<ngram>\d+)$')
ROUGE_VARIANT = re.compile(r'^(?:mean_)?Rouge-(?P<variant>[12L])-(?P<statistic>[RPF])$')

_SNAKE_BOUNDARY = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_NON_NAME = re.compile(r'[^a-z0-9]+')

__all__ = [
    'BLEU_N',
    'ROUGE_VARIANT',
    'canonical_aggregation',
    'canonical_overlap_name',
    'canonicalize_producer_identity',
    'snake_case',
]


def snake_case(value: str) -> str:
    """Lower-case ``value`` and reduce every non-name character to a single underscore."""
    value = _SNAKE_BOUNDARY.sub('_', value).lower()
    return _NON_NAME.sub('_', value).strip('_')


def canonical_aggregation(raw_aggregation: str) -> str:
    """Canonicalize an aggregation spelling without touching its ``k``."""
    return _AGGREGATION_ALIASES.get(raw_aggregation, snake_case(raw_aggregation))


def canonical_overlap_name(name: str, dimensions: Dict[str, Scalar]) -> Optional[str]:
    """Return the canonical identity for an unambiguous BLEU or ROUGE spelling.

    Args:
        name: Raw metric name.
        dimensions: Identity dimensions, extended in place with the axes the spelling encodes.

    Returns:
        ``'bleu'`` or ``'rouge'``, or ``None`` when the name is neither.
    """
    bleu = BLEU_N.fullmatch(name)
    if bleu:
        dimensions.setdefault('ngram', int(bleu.group('ngram')))
        return 'bleu'

    rouge = ROUGE_VARIANT.fullmatch(name)
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
    canonical_aggregation_name = canonical_aggregation(aggregation or 'identity')
    identity_dimensions = dict(dimensions or {})
    canonical_name = canonical_overlap_name(metric_name, identity_dimensions)
    if canonical_name is None:
        snake_name = snake_case(metric_name)
        canonical_name = _PRODUCER_ALIASES.get(snake_name, snake_name)

    try:
        MetricIdentity(name=canonical_name, aggregation='identity')
    except ValueError:
        canonical_name = 'legacy_metric'
        identity_dimensions['original_name'] = original_name

    return MetricIdentity(
        name=canonical_name,
        aggregation=canonical_aggregation_name,
        dimensions=identity_dimensions,
    )
