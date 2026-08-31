"""Aggregate per-sample judge summaries into run-level summaries."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from evalscope.api.metric import JudgeSummary, SampleScore
from evalscope.constants import ScoreStatus


def summarize_judge_runs(score_groups: Iterable[List[SampleScore]]) -> Optional[JudgeSummary]:
    """Combine per-sample judge summaries without treating unavailable samples as zero scores."""
    summaries = [
        score.score.judge_summary
        for scores in score_groups
        for score in scores
        if score.score.judge_summary is not None
    ]
    if not summaries:
        return None
    total = len(summaries)
    scored = sum(summary.status.is_usable for summary in summaries)
    failures: Dict[str, int] = defaultdict(int)
    models = []
    for summary in summaries:
        for name, count in summary.failures.items():
            failures[name] += count
        models.extend(model for model in summary.judge_models if model not in models)
    status = (
        ScoreStatus.EXCLUDED
        if not scored
        else ScoreStatus.DEGRADED
        if scored != total or any(summary.status is not ScoreStatus.SUCCESS for summary in summaries)
        else ScoreStatus.SUCCESS
    )
    return JudgeSummary(
        status=status,
        scored=scored,
        total=total,
        coverage=scored / total,
        judge_models=models,
        valid_observations=sum(summary.valid_observations for summary in summaries),
        total_observations=sum(summary.total_observations for summary in summaries),
        failures=dict(failures),
        disagreement=summarize_judge_disagreement(summaries),
    )


def summarize_judge_disagreement(summaries: List[JudgeSummary]) -> Dict[str, object]:
    """Aggregate comparable disagreement diagnostics without pretending sample variances are global variance."""
    numeric: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    categorical: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    position_consistency: List[float] = []
    swap_flip_count = 0
    for summary in summaries:
        detail = summary.disagreement
        for name, values in detail.get('numeric', {}).get('all_observations', {}).items():
            for key in ('std', 'range'):
                if key in values:
                    numeric[name][key].append(float(values[key]))
        for name, values in detail.get('categorical', {}).items():
            for key in ('agreement_ratio', 'vote_entropy'):
                if key in values:
                    categorical[name][key].append(float(values[key]))
        if detail.get('position_consistency') is not None:
            position_consistency.append(float(detail['position_consistency']))
        swap_flip_count += int(detail.get('swap_flip_count', 0))

    return {
        'numeric': {
            name: {
                'mean_std': sum(values['std']) / len(values['std']) if values['std'] else 0.0,
                'max_range': max(values['range']) if values['range'] else 0.0,
                'samples': max(len(values['std']), len(values['range'])),
            }
            for name, values in numeric.items()
        },
        'categorical': {
            name: {
                'mean_agreement_ratio': sum(values['agreement_ratio']) / len(values['agreement_ratio'])
                if values['agreement_ratio']
                else 0.0,
                'mean_vote_entropy': sum(values['vote_entropy']) / len(values['vote_entropy'])
                if values['vote_entropy']
                else 0.0,
                'samples': max(len(values['agreement_ratio']), len(values['vote_entropy'])),
            }
            for name, values in categorical.items()
        },
        'position_consistency': {
            'mean': sum(position_consistency) / len(position_consistency) if position_consistency else None,
            'samples': len(position_consistency),
            'swap_flip_count': swap_flip_count,
        },
    }
