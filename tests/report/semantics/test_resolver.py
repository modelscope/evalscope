"""Tests for the metric semantics resolver.

Covers the fixed priority chain and the role attribution:

* ``TestPriorityChain`` -- Property 9: the highest available source wins, deterministically.
* ``TestNoNameInference`` -- Property 10: a name variant never hits the declared entry.
* ``TestDegradation`` -- Property 11: an unresolvable metric degrades safely, never blocks.
* ``TestCrossBenchmarkConsistency`` -- Property 12: one name means the same everywhere.
* ``TestPrimaryRoleAttribution`` -- Property 7: the primary declaration decides the roles.
"""

import strategies
from hypothesis import given
from hypothesis import strategies as st
from typing import Dict, Tuple

from evalscope.api.metric.semantics import MetricDirection, MetricEntry, MetricRole
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.resolver import (
    AUDIT_MESSAGE_PREFIX,
    DIAGNOSTIC_FALLBACK_SEMANTIC_ID,
    ResolvedSemantics,
    SemanticsResolver,
    SemanticsSource,
    diagnostic_fallback,
    get_semantics_resolver,
)

#: Benchmark bundled with EvalScope, used where the shipped tables are exercised.
BUILTIN_BENCHMARK = 'builtin_bench'

#: Benchmark outside this repository.
THIRD_PARTY_BENCHMARK = 'third_party_bench'


def _resolver(
    names: Dict[str, MetricEntry] = None,
    overrides: Dict[Tuple[str, str], MetricEntry] = None,
) -> SemanticsResolver:
    """Build a resolver over injected tables."""
    return SemanticsResolver(
        name_table=names if names is not None else {},
        override_table=overrides if overrides is not None else {},
        perf_fields={},
    )


class TestPriorityChain:
    """Feature: metric-semantics-governance, Property 9: resolution returns the semantics of the
    highest priority available source; a report anchor wins over every catalog table."""

    def test_report_anchor_wins_over_tables(self) -> None:
        resolver = _resolver(
            names={'metric': MetricEntry(baseline='quality.f1.ratio')},
            overrides={(BUILTIN_BENCHMARK, 'metric'): MetricEntry(baseline='quality.recall.ratio')},
        )

        resolved = resolver.resolve(BUILTIN_BENCHMARK, 'metric', embedded_semantic_id='quality.accuracy.ratio')

        assert resolved.source is SemanticsSource.REPORT_ANCHOR
        assert resolved.semantics.semantic_id == 'quality.accuracy.ratio'

    def test_override_wins_over_name_table(self) -> None:
        resolver = _resolver(
            names={'metric': MetricEntry(baseline='quality.f1.ratio')},
            overrides={(BUILTIN_BENCHMARK, 'metric'): MetricEntry(baseline='diagnostic.unspecified')},
        )

        resolved = resolver.resolve(BUILTIN_BENCHMARK, 'metric')

        assert resolved.source is SemanticsSource.BENCHMARK_OVERRIDE
        assert resolved.semantics.semantic_id == 'diagnostic.unspecified'

    def test_name_table_is_used_when_no_override_matches(self) -> None:
        resolver = _resolver(
            names={'metric': MetricEntry(baseline='quality.f1.ratio')},
            overrides={('other_bench', 'metric'): MetricEntry(baseline='diagnostic.unspecified')},
        )

        resolved = resolver.resolve(BUILTIN_BENCHMARK, 'metric')

        assert resolved.source is SemanticsSource.METRIC_NAME
        assert resolved.semantics.semantic_id == 'quality.f1.ratio'

    def test_unknown_anchor_falls_back_to_name_resolution(self) -> None:
        resolver = _resolver(names={'metric': MetricEntry(baseline='quality.f1.ratio')})

        resolved = resolver.resolve(BUILTIN_BENCHMARK, 'metric', embedded_semantic_id='quality.renamed.away')

        assert resolved.source is SemanticsSource.METRIC_NAME
        assert resolved.semantics.semantic_id == 'quality.f1.ratio'

    @given(baseline_id=st.sampled_from(sorted(SEMANTIC_BASELINES)))
    def test_any_anchor_materializes_from_the_baseline_table(self, baseline_id: str) -> None:
        resolved = _resolver().resolve(BUILTIN_BENCHMARK, 'whatever', embedded_semantic_id=baseline_id)

        assert resolved.source is SemanticsSource.REPORT_ANCHOR
        assert resolved.semantics.semantic_id == baseline_id
        assert not resolved.degraded

    def test_resolution_is_deterministic(self) -> None:
        resolver = _resolver(names={'metric': MetricEntry(baseline='quality.accuracy.ratio')})

        first = resolver.resolve(BUILTIN_BENCHMARK, 'metric')
        second = resolver.resolve(BUILTIN_BENCHMARK, 'metric')

        assert first.semantics == second.semantics
        assert first.source is second.source


class TestNoNameInference:
    """Feature: metric-semantics-governance, Property 10: a variant of a declared name that is
    not itself declared degrades to diagnostic and never hits the original entry."""

    @given(variant=strategies.name_variants('mean_acc'))
    def test_name_variants_do_not_hit_the_declared_entry(self, variant: str) -> None:
        resolver = _resolver(names={'mean_acc': MetricEntry(baseline='quality.accuracy.ratio')})

        resolved = resolver.resolve(THIRD_PARTY_BENCHMARK, variant)

        assert resolved.source is SemanticsSource.DIAGNOSTIC_FALLBACK
        assert resolved.semantics.semantic_id == DIAGNOSTIC_FALLBACK_SEMANTIC_ID

    def test_resolver_module_contains_no_regex_matching(self) -> None:
        from pathlib import Path

        source = Path(SemanticsResolver.__module__.replace('.', '/') + '.py')
        text = (Path.cwd() / source).read_text(encoding='utf-8') if source.exists() else None
        if text is None:  # pragma: no cover - resolved via module file below
            import evalscope.metrics.semantics.resolver as resolver_module
            text = Path(resolver_module.__file__).read_text(encoding='utf-8')

        assert 'import re' not in text
        assert 're.match' not in text
        assert 're.search' not in text


class TestDegradation:
    """Feature: metric-semantics-governance, Property 11: an unresolvable metric degrades safely
    with a locatable audit message, and never stops the caller.

    Degrading unconditionally is the point: a final metric name embeds the aggregation name, which
    several benchmarks derive from the data, so no exact-key catalog can be complete."""

    def test_third_party_degrades_with_a_warning(self) -> None:
        resolved = _resolver().resolve(THIRD_PARTY_BENCHMARK, 'unknown_metric')

        assert resolved.degraded
        assert resolved.audit_messages

    def test_builtin_degrades_the_same_way(self) -> None:
        resolved = _resolver().resolve(BUILTIN_BENCHMARK, 'unknown_metric')

        assert resolved.degraded
        assert resolved.semantics.role is MetricRole.DIAGNOSTIC
        assert resolved.audit_messages

    def test_data_dependent_aggregation_prefix_degrades_instead_of_failing(self) -> None:
        # `hallusion_bench` composes `{subcategory}_aAcc` from the data, so the name cannot be
        # declared ahead of time. It must still resolve to something renderable.
        resolved = get_semantics_resolver().resolve('hallusion_bench', 'VD_aAcc')

        assert resolved.degraded
        assert resolved.semantics.semantic_id == DIAGNOSTIC_FALLBACK_SEMANTIC_ID

    def test_audit_message_names_the_metric_and_the_entry_location(self) -> None:
        resolved = _resolver().resolve(BUILTIN_BENCHMARK, 'unknown_metric')

        message = '\n'.join(resolved.audit_messages)
        assert AUDIT_MESSAGE_PREFIX in message
        assert BUILTIN_BENCHMARK in message
        assert 'unknown_metric' in message
        assert "METRIC_NAME_SEMANTICS['unknown_metric']" in message

    def test_fallback_semantics_claim_nothing(self) -> None:
        semantics = diagnostic_fallback('whatever')

        assert semantics.role is MetricRole.DIAGNOSTIC
        assert semantics.direction is MetricDirection.NONE
        assert semantics.display_multiplier is None
        assert semantics.display_unit is None
        assert semantics.value_range is None

    @given(metric_name=strategies.undeclared_metric_names())
    def test_degradation_never_raises(self, metric_name: str) -> None:
        resolved = _resolver().resolve(THIRD_PARTY_BENCHMARK, metric_name)

        assert isinstance(resolved, ResolvedSemantics)
        assert resolved.semantics.metric_name == metric_name


class TestCrossBenchmarkConsistency:
    """Feature: metric-semantics-governance, Property 12: without a collision override, one
    metric name resolves to the same semantic_id / direction / display fields everywhere."""

    @given(
        first=strategies.synthetic_benchmark_names(),
        second=strategies.synthetic_benchmark_names(),
    )
    def test_same_name_same_semantics_across_benchmarks(self, first: str, second: str) -> None:
        resolver = _resolver(names={'shared': MetricEntry(baseline='quality.accuracy.ratio')})

        left = resolver.resolve(first, 'shared').semantics
        right = resolver.resolve(second, 'shared').semantics

        assert left.semantic_id == right.semantic_id
        assert left.direction == right.direction
        assert left.display_kind == right.display_kind
        assert left.display_multiplier == right.display_multiplier
        assert left.display_unit == right.display_unit
        assert left.display_precision == right.display_precision


class TestPrimaryRoleAttribution:
    """Feature: metric-semantics-governance, Property 7: the benchmark's primary_metric promotes
    exactly one metric to primary and demotes the other non-diagnostic ones to auxiliary."""

    def _ner_resolver(self) -> SemanticsResolver:
        return _resolver(
            names={
                'f1_score': MetricEntry(baseline='quality.f1.ratio'),
                'precision': MetricEntry(baseline='quality.precision.ratio'),
                'recall': MetricEntry(baseline='quality.recall.ratio'),
                'accuracy': MetricEntry(baseline='quality.accuracy.ratio'),
                'no_answer_num': MetricEntry(baseline='diagnostic.count.items'),
            }
        )

    def test_declared_primary_is_promoted_and_others_demoted(self) -> None:
        resolver = self._ner_resolver()
        names = ['f1_score', 'precision', 'recall', 'accuracy', 'no_answer_num']

        roles = {
            name: resolver.resolve(BUILTIN_BENCHMARK, name, primary_metric_name='f1_score').semantics.role
            for name in names
        }

        assert roles['f1_score'] is MetricRole.PRIMARY
        assert roles['precision'] is MetricRole.AUXILIARY
        assert roles['recall'] is MetricRole.AUXILIARY
        assert roles['accuracy'] is MetricRole.AUXILIARY
        assert roles['no_answer_num'] is MetricRole.DIAGNOSTIC

    def test_exactly_one_primary_after_attribution(self) -> None:
        resolver = self._ner_resolver()
        names = ['f1_score', 'precision', 'recall', 'accuracy', 'no_answer_num']

        primaries = [
            name for name in names if resolver.resolve(BUILTIN_BENCHMARK, name, primary_metric_name='accuracy'
                                                       ).semantics.role is MetricRole.PRIMARY
        ]

        assert primaries == ['accuracy']

    def test_without_declaration_the_default_role_is_kept(self) -> None:
        resolver = self._ner_resolver()

        assert resolver.resolve(BUILTIN_BENCHMARK, 'precision').semantics.role is MetricRole.PRIMARY

    def test_diagnostic_metric_is_never_promoted(self) -> None:
        resolver = self._ner_resolver()

        resolved = resolver.resolve(BUILTIN_BENCHMARK, 'no_answer_num', primary_metric_name='no_answer_num')

        assert resolved.semantics.role is MetricRole.DIAGNOSTIC
        assert resolved.semantics.direction is MetricDirection.NONE


class TestShippedResolver:
    """The process-wide resolver reads the shipped tables."""

    def test_shipped_resolver_resolves_gsm8k_accuracy(self) -> None:
        resolved = get_semantics_resolver().resolve('gsm8k', 'mean_acc')

        assert resolved.semantics.semantic_id == 'quality.accuracy.ratio'
        assert resolved.semantics.role is MetricRole.PRIMARY
        assert not resolved.degraded
