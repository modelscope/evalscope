"""Tests for the central metric semantics catalog.

``catalog.py`` calls ``_validate_catalog()`` at import time, which resolves **every** entry through
the full ``MetricSemantics`` contract. A dangling baseline or an invalid entry therefore
makes the module unimportable and this file error at collection. Tests that merely re-assert
"every entry resolves" / "no baseline dangles" cannot fail independently and were removed; what is
left exercises the failure path explicitly (via monkeypatch) and pins concrete semantic choices no
validator enforces.
"""

import pytest
from pydantic import ValidationError

from evalscope.api.metric.semantics import MetricDirection, MetricKind
from evalscope.metrics.semantics import catalog as catalog_module
from evalscope.metrics.semantics.catalog import LEGACY_METRIC_MIGRATIONS, METRIC_DEFINITIONS
from evalscope.metrics.semantics.entry import MetricEntry
from evalscope.metrics.semantics.legacy_identity import migrate_legacy_identity


class TestImportTimeValidation:
    """An illegal entry or a dangling baseline must abort the catalog validation."""

    def test_dangling_baseline_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(METRIC_DEFINITIONS, 'bogus_metric', MetricEntry(baseline='quality.does.not.exist'))

        with pytest.raises(ValueError, match='unknown baseline'):
            catalog_module._validate_catalog()

    def test_baseline_is_required(self) -> None:
        with pytest.raises(ValidationError, match='baseline'):
            MetricEntry()

    def test_illegal_entry_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # kind=quality with direction=none violates the contract and must not resolve.
        monkeypatch.setitem(
            METRIC_DEFINITIONS,
            'bogus_metric',
            MetricEntry(
                baseline='quality.accuracy.ratio',
                direction=MetricDirection.NONE,
            ),
        )

        with pytest.raises(ValueError):
            catalog_module._validate_catalog()


class TestGsm8kAccuracy:
    """GSM8K's canonical accuracy definition has the expected base quality contract."""

    def test_accuracy_resolves_to_quality_semantics(self) -> None:
        semantics = METRIC_DEFINITIONS['accuracy'].resolve('accuracy')

        assert semantics.semantic_id == 'quality.accuracy.ratio'
        assert semantics.kind is MetricKind.QUALITY
        assert semantics.direction is MetricDirection.HIGHER_IS_BETTER


def test_job_bench_normalized_score_is_a_bounded_ratio() -> None:
    semantics = METRIC_DEFINITIONS['normalized_score'].resolve('normalized_score')

    assert semantics.semantic_id == 'quality.score.ratio'
    assert semantics.value_range is not None
    assert semantics.value_range.min == 0
    assert semantics.value_range.max == 1


def test_v2_registry_contains_only_canonical_non_dynamic_names() -> None:
    forbidden_names = {'score', 'overall', 'total_score'}
    for name in METRIC_DEFINITIONS:
        assert name == name.lower()
        assert not name.startswith('mean_')
        assert not name.endswith(('_s', '_ms'))
        assert all(character not in name for character in ('@', '/', ' '))
        assert name not in forbidden_names


def test_no_legacy_entry_duplicates_a_canonical_declaration() -> None:
    """A read-old entry must earn its place by differing from what the v2 tables already say.

    A canonical name migrates to itself and resolves through ``METRIC_DEFINITIONS``, so restating it
    here would only add a second place to forget when its display fields change.
    """
    redundant = []
    for name, entry in LEGACY_METRIC_MIGRATIONS.items():
        identity = migrate_legacy_identity(name, 'identity')
        migrates_to_itself = identity.name == name and identity.aggregation == 'identity' and not identity.dimensions
        if migrates_to_itself and METRIC_DEFINITIONS.get(name) == entry:
            redundant.append(name)

    assert sorted(redundant) == [], (
        f'these read-old entries repeat their METRIC_DEFINITIONS declaration verbatim: {sorted(redundant)}; '
        f'drop them and let the resolver read the canonical table'
    )
