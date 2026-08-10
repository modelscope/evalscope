"""Central metric semantics catalog.

The catalog answers one question only: *what does this final report metric name mean?* It is
organized by final report metric name (not by benchmark), because the 219 built-in benchmarks
produce 395 ``(benchmark, metric)`` pairs but only ~131 distinct metric names, and 149
benchmarks emit a single metric. Direction / unit / scale / precision therefore need to be
declared once per name and are reused by every benchmark.

Two tables live here:

- :data:`METRIC_NAME_SEMANTICS` -- final report metric name -> :class:`MetricEntry` (a baseline
  reference plus optional field overrides). Also holds the historical report names, grouped in a
  dedicated section so they can be dropped once no report of that vintage is opened again.
- :data:`BENCHMARK_METRIC_OVERRIDES` -- ``(benchmark_name, final_metric_name)`` -> entry, used
  *only* when the same name means different things in different benchmarks (a collision).

The primary metric of a benchmark is **not** declared here: it is ``BenchmarkMeta.primary_metric``
(next to ``metric_list``), applied as a role adjustment by the resolver.

Every lookup is an exact-key dictionary lookup: no regular expressions, no name normalization,
no fuzzy or magnitude based inference. Importing this module validates every entry (each
``MetricEntry`` resolves against :data:`SEMANTIC_BASELINES` and passes the contract validation),
so an illegal declaration or a dangling baseline reference aborts the import immediately.

The catalog is deliberately incomplete: a final metric name embeds ``AggScore.aggregation_name``,
which several benchmarks derive from the data, so those names cannot be declared ahead of time.
An undeclared name degrades to ``diagnostic.unspecified`` and is logged, never rejected.
"""

from typing import Dict, Tuple

from evalscope.api.metric.semantics import BASELINE_TABLE_LOCATION, MetricEntry
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES

#: Where to declare a metric name, used in audit and validation messages.
METRIC_NAME_TABLE_LOCATION = 'evalscope/metrics/semantics/catalog.py::METRIC_NAME_SEMANTICS'

#: Where to declare a benchmark level collision override, used in validation messages.
BENCHMARK_OVERRIDE_TABLE_LOCATION = 'evalscope/metrics/semantics/catalog.py::BENCHMARK_METRIC_OVERRIDES'

METRIC_NAME_SEMANTICS: Dict[str, MetricEntry] = {
    # --- quality ratios: one line each, reused by every benchmark ------------------------
    # Bounded [0, 1] ratios rendered as percent, higher is better.
    'mean_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'accuracy': MetricEntry(baseline='quality.accuracy.ratio'),
    'multi_choice_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_multi_choice_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'relaxed_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'schema_accuracy': MetricEntry(baseline='quality.accuracy.ratio'),
    'process_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'task_averaged_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'correct_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'error_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_number_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_unit_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    # Accuracy over one kind of UI target, reported alongside the overall accuracy
    # (screenspot_pro).
    'mean_text_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_icon_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_Center_ACC': MetricEntry(baseline='quality.accuracy.ratio'),
    # hallusion_bench prefixes each accuracy with the aggregation bucket. The buckets come from
    # the data, so only the `Overall_` ones can be declared; the per-category rows degrade.
    'Overall_aAcc': MetricEntry(baseline='quality.accuracy.ratio'),
    'Overall_fAcc': MetricEntry(baseline='quality.accuracy.ratio'),
    'Overall_qAcc': MetricEntry(baseline='quality.accuracy.ratio'),
    # Grounding accuracy at a fixed IoU threshold (refcoco).
    'mean_ACC@0.1': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_ACC@0.3': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_ACC@0.5': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_ACC@0.7': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_ACC@0.9': MetricEntry(baseline='quality.accuracy.ratio'),
    # Puzzle accuracy per size / difficulty bucket (zebralogicbench).
    'puzzle_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'cell_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'easy_puzzle_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'medium_puzzle_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'hard_puzzle_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'small_puzzle_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'large_puzzle_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    'xl_puzzle_acc': MetricEntry(baseline='quality.accuracy.ratio'),
    # Instruction following: strict / loose at prompt and instruction level.
    'mean_prompt_level_strict': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_prompt_level_loose': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_inst_level_strict': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_inst_level_loose': MetricEntry(baseline='quality.accuracy.ratio'),
    # Graded-answer benchmarks (simple_qa, browsecomp, chinese_simple_qa) report the share of
    # correct answers as `is_correct`.
    'is_correct': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_is_correct': MetricEntry(baseline='quality.accuracy.ratio'),
    # Agent style task completion ratio (miniwob, wide_search).
    'success_rate': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_success_rate': MetricEntry(baseline='quality.accuracy.ratio'),
    # --- exact match ---------------------------------------------------------------------
    'em': MetricEntry(baseline='quality.exact_match.ratio'),
    'mean_em': MetricEntry(baseline='quality.exact_match.ratio'),
    'exact_match': MetricEntry(baseline='quality.exact_match.ratio'),
    # Tool-use benchmarks score the action and the plan by exact match (tool_bench).
    'mean_Act.EM': MetricEntry(baseline='quality.exact_match.ratio'),
    'mean_Plan.EM': MetricEntry(baseline='quality.exact_match.ratio'),
    # --- pass ratios ---------------------------------------------------------------------
    'pass_rate': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_pass_rate': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'pass_at_k': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_pass_at_k': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'pass_hat_k': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_pass_hat_k': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'Pass@1': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'pass@1': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'strict_pass': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_strict_pass': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'main_problem_pass_rate': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'subproblem_pass_rate': MetricEntry(baseline='quality.pass_at_k.ratio'),
    # --- F1 / precision / recall ---------------------------------------------------------
    'f1': MetricEntry(baseline='quality.f1.ratio'),
    'F1': MetricEntry(baseline='quality.f1.ratio'),
    'f1_score': MetricEntry(baseline='quality.f1.ratio'),
    'f1_macro': MetricEntry(baseline='quality.f1.ratio'),
    'f1_micro': MetricEntry(baseline='quality.f1.ratio'),
    'f1_weighted': MetricEntry(baseline='quality.f1.ratio'),
    'mean_f1': MetricEntry(baseline='quality.f1.ratio'),
    'mean_F1': MetricEntry(baseline='quality.f1.ratio'),
    'task_averaged_f1': MetricEntry(baseline='quality.f1.ratio'),
    'simple_f1_score': MetricEntry(baseline='quality.f1.ratio'),
    'tool_call_f1': MetricEntry(baseline='quality.f1.ratio'),
    'precision': MetricEntry(baseline='quality.precision.ratio'),
    'mean_boundary_precision': MetricEntry(baseline='quality.precision.ratio'),
    'recall': MetricEntry(baseline='quality.recall.ratio'),
    # --- text generation overlap and similarity ------------------------------------------
    'Bleu_1': MetricEntry(baseline='quality.bleu.ratio'),
    'Bleu_2': MetricEntry(baseline='quality.bleu.ratio'),
    'Bleu_3': MetricEntry(baseline='quality.bleu.ratio'),
    'Bleu_4': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_Bleu_1': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_Bleu_2': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_Bleu_3': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_Bleu_4': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_BLEU': MetricEntry(baseline='quality.bleu.ratio'),
    'mean_bleu': MetricEntry(baseline='quality.bleu.ratio'),
    'ROUGE_L': MetricEntry(baseline='quality.rouge.ratio'),
    'mean_ROUGE_L': MetricEntry(baseline='quality.rouge.ratio'),
    'mean_Rouge': MetricEntry(baseline='quality.rouge.ratio'),
    'mean_Rouge-L': MetricEntry(baseline='quality.rouge.ratio'),
    'METEOR': MetricEntry(baseline='quality.meteor.ratio'),
    'mean_METEOR': MetricEntry(baseline='quality.meteor.ratio'),
    'CIDEr': MetricEntry(baseline='quality.cider.unbounded'),
    'mean_CIDEr': MetricEntry(baseline='quality.cider.unbounded'),
    'bert_score': MetricEntry(baseline='quality.similarity.ratio'),
    'mean_bert_score': MetricEntry(baseline='quality.similarity.ratio'),
    'mean_comet': MetricEntry(baseline='quality.similarity.ratio'),
    'sem_score': MetricEntry(baseline='quality.similarity.ratio'),
    # ANLS is the normalized similarity between answer strings (docvqa, infovqa).
    'anls': MetricEntry(baseline='quality.similarity.ratio'),
    'Semantic Consistency': MetricEntry(baseline='quality.similarity.ratio'),
    'Perceptual Similarity': MetricEntry(baseline='quality.similarity.ratio'),
    # --- localization --------------------------------------------------------------------
    'mean_IoU': MetricEntry(baseline='quality.iou.ratio'),
    # --- speech recognition error rates: lower is better ---------------------------------
    'wer': MetricEntry(baseline='quality.wer.ratio'),
    'mean_wer': MetricEntry(baseline='quality.wer.ratio'),
    'audio_wer': MetricEntry(baseline='quality.wer.ratio'),
    'mean_audio_wer': MetricEntry(baseline='quality.wer.ratio'),
    'cer': MetricEntry(baseline='quality.cer.ratio'),
    'mean_cer': MetricEntry(baseline='quality.cer.ratio'),
    'mean_mer': MetricEntry(baseline='quality.mer.ratio'),
    # --- graded failure rates: lower is better -------------------------------------------
    'mean_error_rate': MetricEntry(baseline='quality.error_rate.ratio'),
    'mean_HalluRate': MetricEntry(baseline='quality.error_rate.ratio'),
    'mean_distractor_leakage': MetricEntry(baseline='quality.error_rate.ratio'),
    # --- bounded quality scores ----------------------------------------------------------
    'score': MetricEntry(baseline='quality.score.ratio'),
    'mean_score': MetricEntry(baseline='quality.score.ratio'),
    'vqa_score': MetricEntry(baseline='quality.score.ratio'),
    'overall': MetricEntry(baseline='quality.score.ratio'),
    'overall_mrcr_score': MetricEntry(baseline='quality.score.ratio'),
    'mean_partial_credit': MetricEntry(baseline='quality.score.ratio'),
    'mean_submission_ready': MetricEntry(baseline='quality.score.ratio'),
    'mean_required_coverage': MetricEntry(baseline='quality.coverage.ratio'),
    'mean_coverage_score': MetricEntry(baseline='quality.coverage.ratio'),
    # --- official 0-100 scales -----------------------------------------------------------
    'mean_eq_bench_score': MetricEntry(baseline='quality.score.points_100'),
    # --- win rates -----------------------------------------------------------------------
    'winrate': MetricEntry(baseline='quality.win_rate.ratio'),
    'mean_winrate': MetricEntry(baseline='quality.win_rate.ratio'),
    'win_rate': MetricEntry(baseline='quality.win_rate.ratio'),
    # --- judge scores (unbounded) --------------------------------------------------------
    'gpt_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    'total_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    'mean_total_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    'mean_normalized_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    'mean_avg_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    'mean_net_match_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    # health_bench grades each answer along named rubric axes.
    'communication_quality': MetricEntry(baseline='quality.judge_score.unbounded'),
    'completeness': MetricEntry(baseline='quality.judge_score.unbounded'),
    'context_awareness': MetricEntry(baseline='quality.judge_score.unbounded'),
    'instruction_following': MetricEntry(baseline='quality.judge_score.unbounded'),
    # --- scoring model outputs -----------------------------------------------------------
    'HPSv2.1Score': MetricEntry(baseline='quality.model_score.unbounded'),
    'PickScore': MetricEntry(baseline='quality.model_score.unbounded'),
    'VQAScore': MetricEntry(baseline='quality.model_score.unbounded'),
    # Vendor verification rates: a correctly deployed vendor reports 1.0 for both, so these
    # grade the deployment rather than merely describing it.
    'param_immutable_reject_rate': MetricEntry(baseline='quality.accuracy.ratio'),
    'param_default_accept_rate': MetricEntry(baseline='quality.accuracy.ratio'),
    # --- diagnostics: distribution shares and raw counts carry no direction --------------
    'error_rate': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'is_incorrect': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'is_not_attempted': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'mean_is_incorrect': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'mean_is_not_attempted': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'inference_error_rate': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'yes_ratio': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'maybe_ratio': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'no_answer_num': MetricEntry(baseline='diagnostic.count.items'),
    'count_successful_tool_call': MetricEntry(baseline='diagnostic.count.items'),
    'count_finish_reason_tool_call': MetricEntry(baseline='diagnostic.count.items'),
    'count_finish_reason_tool_calls': MetricEntry(baseline='diagnostic.count.items'),
    # Average reasoning length: a behavioural observation, not a quality signal.
    'avg_reason_lens': MetricEntry(baseline='diagnostic.count.items'),
    # --- legacy names: only produced by report files written before the semantics -------
    # contract. Safe to drop once no report of that vintage is expected to be opened again.
    'AverageAccuracy': MetricEntry(baseline='quality.accuracy.ratio'),
    'WeightedAverageAccuracy': MetricEntry(baseline='quality.accuracy.ratio'),
    'WeightedScorePercent': MetricEntry(baseline='quality.score.points_100'),
    'AverageOutputTps': MetricEntry(baseline='perf.throughput.tokens_per_second'),
    # Names emitted by earlier revisions of adapters that have since renamed their metrics.
    # Kept so an archived report still renders with the right direction and unit instead of
    # degrading to a bare number.
    'mean_precision': MetricEntry(baseline='quality.precision.ratio'),
    'mean_recall': MetricEntry(baseline='quality.recall.ratio'),
    'mean_f1_score': MetricEntry(baseline='quality.f1.ratio'),
    'official_mean_precision': MetricEntry(baseline='quality.precision.ratio'),
    'official_mean_recall': MetricEntry(baseline='quality.recall.ratio'),
    'official_mean_f1_score': MetricEntry(baseline='quality.f1.ratio'),
    'official_mean_all_answers_correct': MetricEntry(baseline='quality.accuracy.ratio'),
    'mean_simple_pass_rate': MetricEntry(baseline='quality.pass_at_k.ratio'),
    'mean_simple_partial_credit': MetricEntry(baseline='quality.score.ratio'),
    'mean_simple_error_rate': MetricEntry(baseline='quality.error_rate.ratio'),
    'mean_compliance_score': MetricEntry(baseline='quality.score.ratio'),
    'mean_max_score': MetricEntry(baseline='quality.judge_score.unbounded'),
    # Document parsing: TEDS and CDM are similarity scores, edit distances are error rates.
    'overall_CH': MetricEntry(baseline='quality.score.ratio'),
    'overall_EN': MetricEntry(baseline='quality.score.ratio'),
    'table_TEDS': MetricEntry(baseline='quality.similarity.ratio'),
    'table_TEDS_structure_only': MetricEntry(baseline='quality.similarity.ratio'),
    'display_formula_CDM': MetricEntry(baseline='quality.similarity.ratio'),
    'text_block_Edit_dist': MetricEntry(baseline='quality.error_rate.ratio'),
    'display_formula_Edit_dist': MetricEntry(baseline='quality.error_rate.ratio'),
    'table_Edit_dist': MetricEntry(baseline='quality.error_rate.ratio'),
    'reading_order_Edit_dist': MetricEntry(baseline='quality.error_rate.ratio'),
    # Response and rater failure shares, and run cost breakdowns: observations, not grades.
    'rate_empty_model_response': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'rate_empty_auto_rater_response': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'rate_invalid_auto_rater_response': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'official_mean_fully_incorrect_items': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'official_mean_correct_with_excessive_answers': MetricEntry(baseline='diagnostic.parse_status.ratio'),
    'mean_total_tokens': MetricEntry(baseline='diagnostic.count.items'),
    'mean_total_model_input_tokens': MetricEntry(baseline='diagnostic.count.items'),
    'mean_total_model_output_tokens': MetricEntry(baseline='diagnostic.count.items'),
    'mean_total_wall_time_s': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
    'mean_total_model_time_s': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
    'mean_total_tool_time_s': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
    'mean_total_other_time_s': MetricEntry(baseline='diagnostic.unspecified', raw_unit='s', display_precision=2),
}
"""Final report metric name -> catalog entry, reused by every benchmark.

Seeded from the metric names this repository actually emits, so the common names render with the
right direction and unit. A name that is not here degrades to a diagnostic rather than blocking.
"""

#: ``k`` values the runtime-sized ``pass@k`` / ``pass^k`` / ``vote@k`` families are declared for.
#: Exact-key lookup cannot cover an unbounded family, so the common powers-of-two and decimal
#: sample counts are declared explicitly; a ``k`` outside this tuple degrades to diagnostic with
#: an audit message pointing here.
DYNAMIC_K_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100)


def _dynamic_pass_at_k_names() -> Dict[str, MetricEntry]:
    """Build the ``mean_acc_pass@{k}`` style entries of the runtime-sized families.

    Returns:
        Final report metric name -> entry, one per template and declared ``k``.
    """
    templates = ('mean_acc_pass@{k}', 'mean_acc_pass^{k}', 'mean_acc_vote@{k}')
    return {
        template.format(k=k): MetricEntry(baseline='quality.pass_at_k.ratio')
        for template in templates
        for k in DYNAMIC_K_VALUES
    }


METRIC_NAME_SEMANTICS.update(_dynamic_pass_at_k_names())

BENCHMARK_METRIC_OVERRIDES: Dict[Tuple[str, str], MetricEntry] = {
    # `total_score` is a judge score in mia_bench but the raw sum of passed rubric weights in
    # job_bench, i.e. an intermediate judge value: reassign the collision to a diagnostic.
    ('job_bench', 'total_score'): MetricEntry(baseline='diagnostic.unspecified'),
    ('job_bench', 'mean_total_score'): MetricEntry(baseline='diagnostic.unspecified'),
}
"""``(benchmark_name, final_metric_name)`` -> entry, only for same-name / different-meaning
collisions. Each entry carries the collision reason in a comment."""


def _validate_catalog() -> None:
    """Materialize every catalog entry at import time so illegal declarations fail fast.

    Each :class:`MetricEntry` is resolved: its ``baseline`` reference must exist in
    :data:`SEMANTIC_BASELINES` and the merged declaration must pass the contract validation.

    Raises:
        ValueError: If an entry references a baseline absent from the baseline table.
        pydantic.ValidationError: If a resolved entry violates the metric semantics contract.
    """
    for name, entry in METRIC_NAME_SEMANTICS.items():
        if entry.baseline is not None and entry.baseline not in SEMANTIC_BASELINES:
            raise ValueError(
                f"metric name '{name}' in {METRIC_NAME_TABLE_LOCATION} references unknown baseline "
                f"'{entry.baseline}'; declare it at {BASELINE_TABLE_LOCATION}"
            )
        entry.resolve(name)

    for (benchmark_name, metric_name), entry in BENCHMARK_METRIC_OVERRIDES.items():
        if entry.baseline is not None and entry.baseline not in SEMANTIC_BASELINES:
            raise ValueError(
                f"override ('{benchmark_name}', '{metric_name}') in {BENCHMARK_OVERRIDE_TABLE_LOCATION} "
                f"references unknown baseline '{entry.baseline}'; declare it at {BASELINE_TABLE_LOCATION}"
            )
        entry.resolve(metric_name)


_validate_catalog()

__all__ = [
    'BENCHMARK_METRIC_OVERRIDES',
    'METRIC_NAME_SEMANTICS',
    'METRIC_NAME_TABLE_LOCATION',
]
