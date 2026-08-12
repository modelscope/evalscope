"""Single manifest for exact legacy aliases shared by identity and semantic migration."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LegacyMetricAlias:
    """Canonical identity target and optional read-old semantics of one exact alias."""

    canonical_name: str
    baseline: Optional[str] = None


LEGACY_METRIC_ALIASES: Dict[str, LegacyMetricAlias] = {
    'acc': LegacyMetricAlias('accuracy', 'quality.accuracy.ratio'),
    'AverageAccuracy': LegacyMetricAlias('accuracy', 'quality.accuracy.ratio'),
    'WeightedAverageAccuracy': LegacyMetricAlias('accuracy', 'quality.accuracy.ratio'),
    'f1_score': LegacyMetricAlias('f1', 'quality.f1.ratio'),
    'F1': LegacyMetricAlias('f1', 'quality.f1.ratio'),
    'em': LegacyMetricAlias('exact_match', 'quality.exact_match.ratio'),
    'winrate': LegacyMetricAlias('win_rate', 'quality.win_rate.ratio'),
    'BLEU': LegacyMetricAlias('bleu'),
    'Rouge': LegacyMetricAlias('rouge'),
    'Rouge-L': LegacyMetricAlias('rouge'),
    'ROUGE_L': LegacyMetricAlias('rouge', 'quality.rouge.ratio'),
    'METEOR': LegacyMetricAlias('meteor', 'quality.meteor.ratio'),
    'CIDEr': LegacyMetricAlias('cider', 'quality.cider.unbounded'),
    'IoU': LegacyMetricAlias('iou'),
    'mean_IoU': LegacyMetricAlias('iou', 'quality.iou.ratio'),
    'score': LegacyMetricAlias('normalized_score', 'quality.score.ratio'),
    'overall': LegacyMetricAlias('normalized_score', 'quality.score.ratio'),
    'total_score': LegacyMetricAlias('judge_score', 'quality.judge_score.unbounded'),
    'gpt_score': LegacyMetricAlias('judge_score', 'quality.judge_score.unbounded'),
    'avg_score': LegacyMetricAlias('judge_score'),
    'HalluRate': LegacyMetricAlias('hallucination_rate'),
    'bertscore': LegacyMetricAlias('bert_score'),
    'total_wall_time_s': LegacyMetricAlias('total_wall_time'),
    'total_model_time_s': LegacyMetricAlias('total_model_time'),
    'total_tool_time_s': LegacyMetricAlias('total_tool_time'),
    'total_other_time_s': LegacyMetricAlias('total_other_time'),
    'HPSv2.1Score': LegacyMetricAlias('hps_v2_1_score', 'quality.model_score.unbounded'),
    'PickScore': LegacyMetricAlias('pick_score', 'quality.model_score.unbounded'),
    # Historical VQAScore is a model score; canonical `vqa_score` now denotes a bounded score.
    'VQAScore': LegacyMetricAlias('vqa_score', 'quality.model_score.unbounded'),
}

__all__ = ['LEGACY_METRIC_ALIASES', 'LegacyMetricAlias']
