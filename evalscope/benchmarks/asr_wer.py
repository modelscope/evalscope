from collections.abc import Iterable

from evalscope.api.benchmark import AudioLanguageAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.metrics import macro_mean
from evalscope.metrics.aggregators.aggregators import METRIC_WEIGHTS_KEY, collect_metric_weights
from evalscope.report import Report
from evalscope.report.report import normalize_score


class CorpusWERAdapter(AudioLanguageAdapter):
    """Pool word errors over reference tokens, including insertions when the reference is empty."""

    def _wer_language(self, task_state: TaskState) -> str:
        return 'en'

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from evalscope.metrics.utils.text_normalizer.wer import normalize_text, word_errors

        language = self._wer_language(task_state)
        prediction = normalize_text(filtered_prediction or '', language)
        reference = normalize_text(reference, language)
        edits, words = word_errors([reference], [prediction], language)
        return Score(
            value={'wer': edits / (words or 1)},
            extracted_prediction=prediction,
            prediction=original_prediction,
            metadata={METRIC_WEIGHTS_KEY: {'wer': words}},
        )

    def aggregate_scores(self, sample_scores: list[SampleScore]) -> list[AggScore]:
        aggregates = super().aggregate_scores(sample_scores)
        if self.aggregation != 'weighted_mean':
            return aggregates
        silent = [s for s in sample_scores if 'wer' in s.score.value and collect_metric_weights(s).get('wer') == 0]
        if not silent:
            return aggregates
        insertions = sum(s.score.value['wer'] for s in silent)
        for aggregate in aggregates:
            if aggregate.metric_name == 'wer':
                if aggregate.aggregation == 'weighted_mean':
                    aggregate.score += insertions / aggregate.num
                return aggregates
        # WeightedMean omits a metric when its total weight is zero.
        aggregates.append(
            AggScore(
                metric_name='wer',
                aggregation='weighted_mean',
                score=insertions,
                num=0,
                ids=[s.sample_id for s in silent],
                metadata={'weighted': True, 'samples': len(silent), 'total_weight': 0},
            )
        )
        return aggregates

    def _on_generate_report(self, scores: dict[str, list[AggScore]], model_name: str) -> Report:
        report = super()._on_generate_report(scores, model_name)
        by_subset = {
            subset: agg
            for subset, aggregates in scores.items()
            for agg in aggregates
            if agg.metric_name == 'wer' and agg.aggregation == 'weighted_mean'
        }
        if not any(agg.num == 0 for agg in by_subset.values()):
            return report

        def pooled(parts: Iterable[AggScore]) -> float:
            # A zero-word subset stores its insertion count directly in its score.
            edits = words = 0
            for part in parts:
                edits += round(part.score * (part.num or 1))
                words += part.num
            return normalize_score(edits / (words or 1))

        for metric in report.metrics:
            if metric.identity.name == 'wer' and metric.identity.aggregation == 'weighted_mean':
                for category in metric.categories:
                    category.score = pooled(by_subset[s.name] for s in category.subsets)
                metric.score = pooled(by_subset.values())
                metric.macro_score = normalize_score(macro_mean([c for c in metric.categories if c.num > 0]))
        return report
