"""Constraints on the alias manifest that no validator can enforce.

``METRIC_ALIASES`` replaced five separate declarations (the read-old manifest, the producer-safe
set, two inline maps inside the migration function, and a literal in ``BenchmarkMeta``). These gates
keep them from drifting apart again, and pin the obligation each scope carries.
"""
import pytest

from evalscope.api.registry import METRIC_REGISTRY
from evalscope.metrics.semantics.aliases import METRIC_ALIASES, AliasScope, aliases_in_scope, read_old_baselines
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.catalog import METRIC_DEFINITIONS
from evalscope.metrics.semantics.legacy_identity import migrate_legacy_identity
from evalscope.metrics.semantics.naming import canonicalize_producer_identity


def test_every_canonical_target_is_declared() -> None:
    """An alias pointing at an undeclared name would silently degrade to a diagnostic."""
    undeclared = sorted({
        alias.canonical_name
        for alias in METRIC_ALIASES.values() if alias.canonical_name not in METRIC_DEFINITIONS
    })

    assert undeclared == []


def test_every_read_old_baseline_exists() -> None:
    dangling = sorted(name for name, baseline in read_old_baselines().items() if baseline not in SEMANTIC_BASELINES)

    assert dangling == []


class TestScopeObligations:

    def test_metric_list_aliases_keep_the_registry_lookup_working(self) -> None:
        """``metric_list`` entries are passed to ``get_metric()``, so both spellings must resolve."""
        broken = {
            alias: canonical
            for alias, canonical in aliases_in_scope(AliasScope.METRIC_LIST).items()
            if alias in METRIC_REGISTRY and canonical not in METRIC_REGISTRY
        }

        assert broken == {}

    def test_producer_aliases_are_exact_synonyms(self) -> None:
        """A producer rewrite may re-spell a name; reassigning what it measures is read-old only."""
        reassigning = {'score', 'overall', 'total_score', 'gpt_score', 'avg_score', 'VQAScore'}
        overlap = reassigning & set(aliases_in_scope(AliasScope.PRODUCER))

        assert overlap == set()

    def test_every_alias_is_readable_when_migrating(self) -> None:
        assert set(aliases_in_scope(AliasScope.READ_OLD)) == set(METRIC_ALIASES)


@pytest.mark.parametrize('alias', sorted(aliases_in_scope(AliasScope.PRODUCER)))
def test_producer_path_applies_its_own_scope(alias: str) -> None:
    canonical = METRIC_ALIASES[alias].canonical_name

    assert canonicalize_producer_identity(alias, 'mean').name == canonical


@pytest.mark.parametrize('alias', sorted(METRIC_ALIASES))
def test_read_old_path_applies_every_alias(alias: str) -> None:
    declared = METRIC_ALIASES[alias]
    identity = migrate_legacy_identity(alias, 'identity')

    assert identity.name == declared.canonical_name
    assert declared.dimensions.items() <= identity.dimensions.items()
