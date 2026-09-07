"""Single manifest of every metric name alias, and which rewrites each one is allowed to drive.

An alias is one spelling of a metric that some producer or some archived report used. The scopes
exist because the three rewrites have different obligations, and conflating them has broken things
before:

- ``READ_OLD`` only reinterprets a stored report, so it may reassign an ambiguous name.
- ``PRODUCER`` rewrites what a fresh aggregate declares, so it must be an exact synonym.
- ``METRIC_LIST`` rewrites a ``BenchmarkMeta.metric_list`` entry, which the default scoring loop
  passes to ``get_metric()``. The canonical name must therefore resolve in the metric registry too
  -- the constraint the ``bertscore`` / ``bert_score`` mismatch violated.

``baseline`` is the read-old semantics of the spelling, present only when opening an old report
should show something other than what the canonical name resolves to today.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Mapping, Optional

from evalscope.api.metric.semantics import Scalar

__all__ = ['AliasScope', 'METRIC_ALIASES', 'MetricAlias', 'aliases_in_scope', 'read_old_baselines']


class AliasScope(str, Enum):
    """Which rewrite an alias is declared for."""

    READ_OLD = 'read_old'
    """Migrating a stored v1 report or built-in adapter output."""

    PRODUCER = 'producer'
    """Canonicalizing a fresh aggregate's own metric name."""

    METRIC_LIST = 'metric_list'
    """Normalizing a ``BenchmarkMeta.metric_list`` entry."""


_READ_OLD = frozenset({AliasScope.READ_OLD})
_EVERY_SCOPE = frozenset(AliasScope)


@dataclass(frozen=True)
class MetricAlias:
    """One spelling, the canonical name it stands for, and what may rewrite it.

    A frozen dataclass rather than a Pydantic model: unlike ``MetricEntry``, a row here is never
    serialized and never resolved through contract validation, and positional construction is what
    lets the table below read as a vocabulary instead of a wall of keywords.
    """

    canonical_name: str
    baseline: Optional[str] = None
    dimensions: Mapping[str, Scalar] = field(default_factory=dict)
    """Axes the spelling itself encodes, such as ``Act.EM`` naming the action target."""

    scopes: FrozenSet[AliasScope] = _READ_OLD


METRIC_ALIASES: Dict[str, MetricAlias] = {
    # --- exact synonyms, safe for every rewrite ------------------------------------------
    'acc': MetricAlias('accuracy', 'quality.accuracy.ratio', scopes=_EVERY_SCOPE),
    'f1_score': MetricAlias('f1', 'quality.f1.ratio', scopes=_EVERY_SCOPE),
    'em': MetricAlias('exact_match', 'quality.exact_match.ratio', scopes=_EVERY_SCOPE),
    # `bert_score` is registered as an alias of the scorer, so the registry lookup survives.
    'bertscore': MetricAlias('bert_score', scopes=frozenset({AliasScope.READ_OLD, AliasScope.PRODUCER})),
    # Snake-casing already turns `F1` into `f1`, so no producer rewrite is needed for it.
    'F1': MetricAlias('f1', 'quality.f1.ratio', scopes=frozenset({AliasScope.READ_OLD, AliasScope.METRIC_LIST})),
    # --- read-old only: spellings no current producer emits -------------------------------
    'AverageAccuracy': MetricAlias('accuracy', 'quality.accuracy.ratio'),
    'WeightedAverageAccuracy': MetricAlias('accuracy', 'quality.accuracy.ratio'),
    'average_accuracy': MetricAlias('accuracy'),
    'center_acc': MetricAlias('accuracy'),
    'f_1': MetricAlias('f1'),
    'winrate': MetricAlias('win_rate', 'quality.win_rate.ratio'),
    'BLEU': MetricAlias('bleu'),
    'Rouge': MetricAlias('rouge'),
    'Rouge-L': MetricAlias('rouge'),
    'rouge_l': MetricAlias('rouge'),
    'ROUGE_L': MetricAlias('rouge', 'quality.rouge.ratio'),
    'METEOR': MetricAlias('meteor', 'quality.meteor.ratio'),
    'CIDEr': MetricAlias('cider', 'quality.cider.unbounded'),
    'IoU': MetricAlias('iou'),
    'mean_IoU': MetricAlias('iou', 'quality.iou.ratio'),
    'HalluRate': MetricAlias('hallucination_rate'),
    'total_wall_time_s': MetricAlias('total_wall_time'),
    'total_model_time_s': MetricAlias('total_model_time'),
    'total_tool_time_s': MetricAlias('total_tool_time'),
    'total_other_time_s': MetricAlias('total_other_time'),
    'HPSv2.1Score': MetricAlias('hps_v2_1_score', 'quality.model_score.unbounded'),
    'PickScore': MetricAlias('pick_score', 'quality.model_score.unbounded'),
    # Historical VQAScore is a model score; canonical `vqa_score` denotes bounded VQA accuracy.
    'VQAScore': MetricAlias('vqa_model_score', 'quality.model_score.unbounded'),
    # --- read-old only: names that reassign an ambiguous score ---------------------------
    # These reinterpret rather than re-spell, so no fresh producer may be rewritten by them.
    'score': MetricAlias('normalized_score', 'quality.score.ratio'),
    'overall': MetricAlias('normalized_score', 'quality.score.ratio'),
    'total_score': MetricAlias('judge_score', 'quality.judge_score.unbounded'),
    'gpt_score': MetricAlias('judge_score', 'quality.judge_score.unbounded'),
    'avg_score': MetricAlias('judge_score'),
    # --- read-old only: the spelling carries a dimension ---------------------------------
    'Act.EM': MetricAlias('exact_match', dimensions={'target': 'action'}),
    'Plan.EM': MetricAlias('exact_match', dimensions={'target': 'plan'}),
    # Bare `?Acc` keeps only the concept here; which target it measures is benchmark knowledge and
    # is assigned by the hallusion_bench rule, which alone knows the spelling is scoped to it.
    'a_acc': MetricAlias('accuracy'),
    'f_acc': MetricAlias('accuracy'),
    'q_acc': MetricAlias('accuracy'),
}


def aliases_in_scope(scope: AliasScope) -> Dict[str, str]:
    """Return ``alias -> canonical_name`` for every alias declared for ``scope``.

    Args:
        scope: The rewrite being performed.

    Returns:
        The mapping that rewrite is allowed to apply.
    """
    return {name: alias.canonical_name for name, alias in METRIC_ALIASES.items() if scope in alias.scopes}


def read_old_baselines() -> Dict[str, str]:
    """Return ``alias -> baseline`` for the aliases that carry read-old semantics."""
    return {name: alias.baseline for name, alias in METRIC_ALIASES.items() if alias.baseline is not None}
