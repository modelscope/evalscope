"""Alignment tests between the plugin registries and the metric semantics catalog.

``@register_metric`` and ``METRIC_DEFINITIONS`` are two separate namespaces: the first registers a
*computation*, the second declares how a *reported metric* is interpreted and displayed. Nothing at
runtime links them, so a metric can be registered and still reach the report with no direction, no
unit and no value range -- which is exactly the guessing the semantics contract exists to remove.

These are policy gates, not behaviour tests. They fail in CI so the gap is visible when a metric is
added, and they deliberately do not change runtime behaviour: an undeclared metric still degrades to
``diagnostic.unspecified`` rather than aborting a run.
"""
import pytest
import re
from pathlib import Path
from typing import Dict, List, Set

import evalscope  # noqa: F401  # imported for its registration side effects
from evalscope.api.benchmark.adapters.text2image_adapter import T2I_REPORT_METRIC_NAMES
from evalscope.api.metric.semantics import KNOWN_AGGREGATIONS, MetricIdentity
from evalscope.api.registry import BENCHMARK_REGISTRY, METRIC_REGISTRY
from evalscope.metrics.semantics.catalog import METRIC_DEFINITIONS, METRIC_NAME_TABLE_LOCATION
from evalscope.metrics.semantics.identity import canonicalize_producer_identity, migrate_legacy_identity
from evalscope.metrics.semantics.resolver import get_semantics_resolver


def _canonical_metric_name(registered_name: str) -> str:
    """Canonical report metric name produced by a registered scorer."""
    if registered_name in T2I_REPORT_METRIC_NAMES:
        return T2I_REPORT_METRIC_NAMES[registered_name]
    return canonicalize_producer_identity(registered_name, 'mean').name


class TestRegisteredMetricsHaveSemantics:
    """Every registered metric must resolve to a declared canonical name."""

    def test_no_registered_metric_falls_back_to_diagnostic(self) -> None:
        undeclared: List[str] = sorted(
            f'{name} -> {_canonical_metric_name(name)}' for name in METRIC_REGISTRY
            if _canonical_metric_name(name) not in METRIC_DEFINITIONS
        )
        assert undeclared == [], (
            'these registered metrics have no declared semantics, so they render without a '
            f'direction, unit or value range: {undeclared}; declare them at '
            f'{METRIC_NAME_TABLE_LOCATION}'
        )

    def test_gate_actually_detects_a_missing_declaration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative check: the assertion above must not be vacuously true."""
        monkeypatch.delitem(METRIC_DEFINITIONS, _canonical_metric_name('accuracy'))
        with pytest.raises(AssertionError, match='no declared semantics'):
            self.test_no_registered_metric_falls_back_to_diagnostic()

    def test_every_builtin_primary_selector_resolves_to_declared_semantics(self) -> None:
        resolver = get_semantics_resolver()
        undeclared = []
        for benchmark_name, meta in sorted(BENCHMARK_REGISTRY.items()):
            selector = meta.primary_metric
            if selector is None:
                continue
            identity = MetricIdentity(
                name=selector.name,
                aggregation=selector.aggregation or 'identity',
                dimensions=selector.dimensions,
            )
            if resolver.resolve(benchmark_name, identity).degraded:
                undeclared.append(f'{benchmark_name} -> {identity.key}')

        assert undeclared == [], (
            'these built-in primary selectors resolve to diagnostic semantics and cannot generate '
            f'a primary report metric: {undeclared}; declare them at {METRIC_NAME_TABLE_LOCATION}'
        )


class TestAggregationAxisVocabulary:
    """The identity's aggregation axis must stay a closed, declared vocabulary.

    ``BenchmarkMeta.aggregation`` is deliberately not asserted against
    :data:`KNOWN_AGGREGATIONS`: it selects an aggregator, and an adapter that computes its own
    aggregates uses the field as a free-form label (``elo``, ``f1``) that never reaches an identity.
    What must hold is that every name actually written into ``AggScore.aggregation`` is declared.
    """

    def test_registered_aggregator_names_are_known(self) -> None:
        """Names a registered aggregator writes into every aggregate it produces.

        Only classes that declare ``name`` in their own body are checked: those are the ones whose
        class attribute reaches ``AggScore.aggregation``. A compound aggregator sets ``self.name``
        in ``__init__`` and then delegates to ``Mean``, so it emits ``mean``, not its own name --
        ``test_compound_aggregators_never_reach_an_identity`` pins that separately.
        """
        from evalscope.api.registry import AGGREGATION_REGISTRY

        emitted = {
            vars(cls)['name']
            for cls in AGGREGATION_REGISTRY.values() if isinstance(vars(cls).get('name'), str)
        }
        assert emitted, 'expected at least one aggregator to declare a class-level name'
        assert emitted <= KNOWN_AGGREGATIONS, (
            f'aggregators emit undeclared aggregation names: {sorted(emitted - KNOWN_AGGREGATIONS)}; '
            f'add them to KNOWN_AGGREGATIONS in evalscope/api/metric/semantics.py'
        )

    def test_aggregation_literals_in_shipped_code_are_known(self) -> None:
        """Every ``AggScore(..., aggregation='x')`` literal in the package must be declared."""
        package_root = Path(evalscope.__file__).parent
        emitted: Dict[str, List[str]] = {}
        for path in sorted(package_root.rglob('*.py')):
            text = path.read_text(encoding='utf-8')
            for block in re.finditer(r'AggScore\((?:[^()]|\([^()]*\))*?\)', text, re.S):
                literal = re.search(r"aggregation\s*=\s*'([^']*)'", block.group(0))
                if literal:
                    emitted.setdefault(literal.group(1), []).append(path.name)
        assert emitted, 'expected to find at least one explicit aggregation literal'
        undeclared = {name: sorted(set(files)) for name, files in emitted.items() if name not in KNOWN_AGGREGATIONS}
        assert undeclared == {}, (
            f'adapters emit undeclared aggregation names: {undeclared}; add them to '
            f'KNOWN_AGGREGATIONS in evalscope/api/metric/semantics.py'
        )

    def test_every_benchmark_aggregation_is_dispatchable_or_self_computed(self) -> None:
        """A declared aggregation must either resolve to an aggregator or be computed by the adapter.

        ``DefaultDataAdapter.aggregate_scores`` calls ``get_aggregation(self.aggregation)``, so a
        benchmark that declares an unregistered name without overriding that method raises at
        aggregation time -- after the whole evaluation has already run.
        """
        from evalscope.api.registry import AGGREGATION_REGISTRY

        broken: Dict[str, str] = {}
        for benchmark_name in sorted(BENCHMARK_REGISTRY):
            meta = BENCHMARK_REGISTRY[benchmark_name]
            declared = getattr(meta, 'aggregation', None)
            if not declared or declared in AGGREGATION_REGISTRY:
                continue
            adapter = getattr(meta, 'data_adapter', None)
            owner = next(
                (klass.__name__ for klass in getattr(adapter, '__mro__', ()) if 'aggregate_scores' in vars(klass)),
                None,
            )
            if owner in (None, 'DefaultDataAdapter'):
                broken[benchmark_name] = declared
        assert broken == {}, (
            f'these benchmarks declare an unregistered aggregation and inherit the default '
            f'aggregation path, so they raise at aggregation time: {broken}; either register the '
            f'aggregator or override aggregate_scores'
        )

    @pytest.mark.parametrize(
        ('legacy_aggregation', 'expected'),
        [('avg@4', 'mean'), ('pass@2', 'pass_at_k'), ('vote@8', 'vote_at_k'), ('max@3', 'max'), ('', 'identity')],
    )
    def test_normalized_aggregations_are_known(self, legacy_aggregation: str, expected: str) -> None:
        """Names produced by ``migrate_legacy_identity`` must also be declared."""
        canonical = migrate_legacy_identity('accuracy', legacy_aggregation).aggregation
        assert canonical == expected
        assert canonical in KNOWN_AGGREGATIONS

    def test_compound_aggregators_never_reach_an_identity(self) -> None:
        """`mean_and_*` delegate to `Mean`, so they must not be in the identity vocabulary."""
        from evalscope.api.registry import AGGREGATION_REGISTRY

        compound = {name for name in AGGREGATION_REGISTRY if name.startswith('mean_and_')}
        assert compound, 'expected at least one compound aggregator to be registered'
        assert compound & KNOWN_AGGREGATIONS == set()


class TestMetricListNormalization:
    """``BenchmarkMeta._normalize_metric_list`` rewrites names that are also registry lookup keys.

    A ``metric_list`` entry is passed straight to ``get_metric()`` by the default scoring loop, so
    rewriting one is only safe while the canonical form resolves too. This is the constraint that
    the ``bertscore`` / ``bert_score`` mismatch violated.
    """

    def test_normalized_aliases_keep_the_registry_lookup_working(self) -> None:
        aliases = _declared_metric_list_aliases()
        broken = {
            alias: canonical
            for alias, canonical in ((a, migrate_legacy_identity(a, 'identity').name) for a in aliases)
            # Safe when neither spelling names a scorer: the adapter computes its own metrics.
            if alias in METRIC_REGISTRY and canonical not in METRIC_REGISTRY
        }
        assert broken == {}, (
            f'these metric_list aliases are rewritten to a name that is not registered, so '
            f'get_metric() would fail after normalization: {broken}; register the canonical name '
            f'as an alias of the scorer, as bert_score does'
        )

    def test_no_alias_is_dead(self) -> None:
        """An alias no adapter declares only pretends the compatibility shim is still needed."""
        declared: Set[str] = set()
        entries = re.compile(r'metric_list\s*=\s*\[([^\]]*)\]', re.S)
        # Resolved off the installed package, not the working directory: a relative path makes this
        # gate fail from any other cwd, and pass vacuously wherever the directory is simply absent.
        benchmarks_root = Path(evalscope.__file__).parent / 'benchmarks'
        for path in benchmarks_root.rglob('*.py'):
            for block in entries.finditer(path.read_text(encoding='utf-8')):
                declared.update(re.findall(r"'([^']+)'", block.group(1)))

        dead = sorted(_declared_metric_list_aliases() - declared)
        assert dead == [], (
            f'these aliases are no longer declared by any adapter: {dead}; drop them so a '
            f'third-party adapter emitting the name is not silently normalized instead of warned'
        )


def _declared_metric_list_aliases() -> Set[str]:
    """The alias set ``_normalize_metric_list`` rewrites, read off the implementation."""
    import inspect

    from evalscope.api.benchmark.meta import BenchmarkMeta

    source = inspect.getsource(BenchmarkMeta._normalize_metric_list)
    literal = re.search(r'aliases\s*=\s*\{([^}]*)\}', source)
    assert literal, 'could not locate the alias set literal'
    return set(re.findall(r"'([^']+)'", literal.group(1)))
