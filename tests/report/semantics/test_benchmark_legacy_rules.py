"""Behaviour lock for the per-benchmark legacy metric-name rules.

``LEGACY_NAME_RULES`` in ``evalscope/metrics/semantics/legacy_identity.py`` declares one pattern per
benchmark shape and both :func:`migrate_legacy_identity` and :func:`is_known_legacy_spelling` derive
from it. Before that table existed the same knowledge was written twice -- an if-chain in the first
function and a separate dict in the second -- so the two could disagree about whether a legacy name
was supported.

These cases pin the resulting identities so the shared table can be edited without silently
changing how a historical report is read. They also pin the ordering constraint that made the table
non-trivial: the ``hallusion_bench`` rule runs *after* the generic ``scope/metric`` split, so a
scoped name keeps its scope instead of folding it into the level.
"""
from typing import Optional

import pytest

from evalscope.metrics.semantics.legacy_identity import is_known_legacy_spelling, migrate_legacy_identity

#: ``(metric_name, benchmark_name, aggregation, expected_key)`` for every rule shape.
_RULE_CASES = [
    # longmemeval / locomo share one rule shape: overall, task_averaged, or a question type.
    ('overall_acc', 'longmemeval', 'identity', 'accuracy:mean[scope="overall"]'),
    ('task_averaged_acc', 'longmemeval', 'identity', 'accuracy:macro_mean[scope="question_types"]'),
    (
        'single_session_user_acc',
        'longmemeval',
        'identity',
        'accuracy:mean[question_type="single_session_user"]',
    ),
    ('overall_f1', 'locomo', 'identity', 'f1:mean[scope="overall"]'),
    ('task_averaged_f1', 'locomo', 'identity', 'f1:macro_mean[scope="question_types"]'),
    ('multi_hop_f1', 'locomo', 'identity', 'f1:mean[question_type="multi_hop"]'),
    # omni_doc_bench: language is an axis of the same metric, not part of its name.
    ('table_TEDS_EN', 'omni_doc_bench', 'identity', 'table_teds:identity[language="en"]'),
    ('overall_CH', 'omni_doc_bench', 'identity', 'normalized_score:identity[language="ch"]'),
    # openai_mrcr: the overall roll-up and one context-length bucket.
    ('overall_mrcr_score', 'openai_mrcr', 'identity', 'mrcr_score:mean[scope="overall"]'),
    (
        '4096-8192_mrcr_score',
        'openai_mrcr',
        'identity',
        'mrcr_score:mean[max_tokens=8192,min_tokens=4096]',
    ),
    # wide_search: kind@k plus a scope, and an optional row/item target.
    ('avg@4_row/f1', 'wide_search', 'identity', 'f1:mean[k=4,scope="row"]'),
    ('pass@2_item/precision', 'wide_search', 'identity', 'precision:pass_at_k[k=2,scope="item"]'),
    (
        'max@8_Scope Name/success_rate',
        'wide_search',
        'identity',
        'success_rate:max[k=8,scope="scope_name"]',
    ),
    # hallusion_bench: aggregation bucket plus scoring target.
    ('Overall_aAcc', 'hallusion_bench', 'identity', 'accuracy:mean[level="overall",target="answer"]'),
    ('Easy_qAcc', 'hallusion_bench', 'identity', 'accuracy:mean[level="easy",target="question"]'),
    # The level prefix is optional: `metric_list` declares the bare spelling. Without this the
    # three targets would share one `accuracy:mean` identity and lose what they measure.
    ('aAcc', 'hallusion_bench', 'identity', 'accuracy:mean[target="answer"]'),
    ('fAcc', 'hallusion_bench', 'identity', 'accuracy:mean[target="figure"]'),
    ('qAcc', 'hallusion_bench', 'identity', 'accuracy:mean[target="question"]'),
    # A target letter outside `[afq]` stays untouched. A name that is only the suffix falls through
    # to the alias table, which reads `acc` as `accuracy` however it was capitalized.
    ('xAcc', 'hallusion_bench', 'identity', 'x_acc:identity'),
    ('Acc', 'hallusion_bench', 'identity', 'accuracy:identity'),
    # general_qa / general_vqa stored post-aggregation overlap names without a `mean_` prefix.
    ('Bleu_4', 'general_qa', 'identity', 'bleu:mean[ngram=4]'),
    ('Rouge-L-R', 'general_vqa', 'identity', 'rouge:mean[statistic="recall",variant="l"]'),
    ('Bleu_4', 'general_vqa', 'identity', 'bleu:mean[ngram=4]'),
    ('Rouge-L-R', 'general_qa', 'identity', 'rouge:mean[statistic="recall",variant="l"]'),
    # An explicit aggregation is not overwritten by the general_qa recovery.
    ('Bleu_4', 'general_qa', 'macro_mean', 'bleu:macro_mean[ngram=4]'),
    # A rule only applies to its own benchmark.
    ('overall_acc', 'unrelated_bench', 'identity', 'overall_acc:identity'),
    ('table_TEDS_EN', 'unrelated_bench', 'identity', 'table_teds_en:identity'),
    ('Overall_aAcc', None, 'identity', 'overall_a_acc:identity'),
]


@pytest.mark.parametrize(
    ('metric_name', 'benchmark_name', 'aggregation', 'expected_key'),
    _RULE_CASES,
)
def test_benchmark_rule_produces_expected_identity(
    metric_name: str, benchmark_name: Optional[str], aggregation: str, expected_key: str
) -> None:
    identity = migrate_legacy_identity(metric_name, aggregation, benchmark_name=benchmark_name)
    assert identity.key == expected_key


@pytest.mark.parametrize(
    ('metric_name', 'benchmark_name', 'expected_key'),
    [
        # The generic `scope/metric` split runs before the hallusion rule, so the scope survives
        # instead of being absorbed into the level.
        ('scope/Overall_aAcc', 'hallusion_bench', 'accuracy:mean[level="overall",scope="scope",target="answer"]'),
        ('Figures/Hard_qAcc', 'hallusion_bench', 'accuracy:mean[level="hard",scope="figures",target="question"]'),
        # Conversely, a rule that runs before the split sees the raw name including the scope.
        ('bucket/overall_acc', 'longmemeval', 'accuracy:mean[question_type="bucket_overall"]'),
    ],
)
def test_scope_split_ordering_is_preserved(metric_name: str, benchmark_name: str, expected_key: str) -> None:
    assert migrate_legacy_identity(metric_name, 'identity', benchmark_name=benchmark_name).key == expected_key


@pytest.mark.parametrize(
    ('metric_name', 'benchmark_name', 'expected'),
    [
        ('overall_acc', 'longmemeval', True),
        ('overall_acc', 'locomo', False),
        ('overall_f1', 'locomo', True),
        ('table_TEDS_EN', 'omni_doc_bench', True),
        ('table_TEDS_DE', 'omni_doc_bench', False),
        ('overall_mrcr_score', 'openai_mrcr', True),
        ('4096-8192_mrcr_score', 'openai_mrcr', True),
        ('mrcr_score', 'openai_mrcr', False),
        ('avg@4_row/f1', 'wide_search', True),
        ('Overall_aAcc', 'hallusion_bench', True),
        ('Overall_zAcc', 'hallusion_bench', False),
        ('overall_acc', None, False),
    ],
)
def test_is_known_uses_the_same_patterns(metric_name: str, benchmark_name: Optional[str], expected: bool) -> None:
    """The membership gate and the rewrite must agree, since they read one declaration."""
    assert is_known_legacy_spelling(metric_name, benchmark_name) is expected


def test_every_declared_rule_ships_examples_covered_by_the_cases_above() -> None:
    """A rule added to the table must come with cases pinning how it reads a legacy name.

    The expectation is derived from each rule's own ``examples``, so a new rule that ships none, or
    whose spellings are not pinned above, fails here instead of going untested.
    """
    from evalscope.metrics.semantics.legacy_identity import LEGACY_NAME_RULES

    pinned = {(metric_name, benchmark_name) for metric_name, benchmark_name, _, _ in _RULE_CASES}
    missing = sorted(
        f'{benchmark}:{example}' for rule in LEGACY_NAME_RULES for benchmark in rule.benchmarks
        for example in rule.examples if (example, benchmark) not in pinned
    )

    assert all(rule.examples for rule in LEGACY_NAME_RULES), 'every rule must declare examples'
    assert missing == [], f'these declared rule examples have no pinned identity: {missing}'
