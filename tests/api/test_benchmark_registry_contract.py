"""Contract tests for the benchmark registry.

These pin the observable behaviour of ``BENCHMARK_REGISTRY`` / ``get_benchmark()``
that callers rely on, so that changing *when* adapter modules are imported cannot
change *what* the registry resolves.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List, Optional

import pytest

from evalscope import benchmarks as benchmark_plugins
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import BENCHMARK_REGISTRY, LazyRegistry, get_benchmark
from evalscope.benchmarks import _INDEX_PATH, adapter_modules, build_index

# A benchmark whose name matches its module leaf.
SIMPLE_NAME = 'gsm8k'
# Two benchmarks registered by one single module, whose names are not derivable
# from the module path (``evalscope.benchmarks.aime.aime_adapter``).
MULTI_NAME_SIBLINGS = ('aime24', 'aime25')
# A benchmark whose name is a suffixed variant of its module leaf.
SUFFIXED_NAME = 'gpqa_diamond'


def test_get_benchmark_resolves_a_simple_name() -> None:
    adapter = get_benchmark(SIMPLE_NAME)

    assert adapter.benchmark_meta.name == SIMPLE_NAME


@pytest.mark.parametrize('name', MULTI_NAME_SIBLINGS)
def test_get_benchmark_resolves_names_sharing_one_module(name: str) -> None:
    """One adapter module registers several names; each must resolve on its own."""
    adapter = get_benchmark(name)

    assert adapter.benchmark_meta.name == name


def test_get_benchmark_resolves_a_name_that_differs_from_its_module_leaf() -> None:
    adapter = get_benchmark(SUFFIXED_NAME)

    assert adapter.benchmark_meta.name == SUFFIXED_NAME


def test_get_benchmark_returns_independent_metadata_copies() -> None:
    """Callers mutate the returned metadata, so it must not alias the registry entry."""
    first = get_benchmark(SIMPLE_NAME)
    second = get_benchmark(SIMPLE_NAME)

    assert first.benchmark_meta is not second.benchmark_meta
    assert first.benchmark_meta is not BENCHMARK_REGISTRY[SIMPLE_NAME]


def test_unknown_benchmark_raises_value_error_with_suggestions() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_benchmark('gsm8kk')

    message = str(excinfo.value)
    assert "Benchmark 'gsm8kk' not found." in message
    # The suggestion machinery must have seen the full name set.
    assert 'Did you mean' in message or 'Available' in message


def test_unknown_benchmark_with_no_close_match_still_lists_alternatives() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_benchmark('definitely_not_a_benchmark')

    assert 'Did you mean' in str(excinfo.value) or 'Available' in str(excinfo.value)


def test_membership_and_lookup_agree() -> None:
    assert SIMPLE_NAME in BENCHMARK_REGISTRY
    assert 'definitely_not_a_benchmark' not in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY.lookup(SIMPLE_NAME) is BENCHMARK_REGISTRY[SIMPLE_NAME]


def test_lookup_of_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match='is not registered'):
        BENCHMARK_REGISTRY.lookup('definitely_not_a_benchmark')


def test_full_enumeration_is_self_consistent() -> None:
    """``keys()`` / ``values()`` / ``items()`` / iteration / ``len()`` must agree.

    Batch consumers (``evalscope benchmark --list``, the docs pipeline and several
    tests) depend on enumeration returning the complete set, not a partially
    populated one.
    """
    keys: List[str] = list(BENCHMARK_REGISTRY.keys())
    items: Dict[str, BenchmarkMeta] = dict(BENCHMARK_REGISTRY.items())

    assert len(keys) == len(BENCHMARK_REGISTRY)
    assert len(list(BENCHMARK_REGISTRY.values())) == len(keys)
    assert set(iter(BENCHMARK_REGISTRY)) == set(keys)
    assert set(items) == set(keys)
    assert BENCHMARK_REGISTRY.list_keys() == sorted(keys)


def test_enumeration_exposes_the_whole_shipped_catalog() -> None:
    """Guards against a discovery change silently shrinking the registry."""
    names = set(BENCHMARK_REGISTRY.keys())

    assert len(names) >= 250, f'registry unexpectedly small: {len(names)} benchmarks'
    assert {SIMPLE_NAME, SUFFIXED_NAME, *MULTI_NAME_SIBLINGS} <= names


def test_every_registered_entry_carries_a_usable_adapter() -> None:
    for name, meta in BENCHMARK_REGISTRY.items():
        assert meta.name == name, f'{name} -> {meta.name}'
        assert meta.data_adapter is not None, f'{name} has no data_adapter'


# --- generated index -------------------------------------------------------------


def test_committed_index_matches_the_registry() -> None:
    """``_index.json`` is generated; a stale copy must be regenerated, not edited.

    Run ``evalscope benchmark-info --update-index`` (or ``make docs-update``) to refresh.
    """
    with open(_INDEX_PATH, encoding='utf-8') as f:
        committed: Dict[str, str] = json.load(f)

    assert committed == build_index(), (
        'evalscope/benchmarks/_index.json is out of date; '
        'regenerate it with `evalscope benchmark-info --update-index`'
    )


def test_index_covers_every_registered_name() -> None:
    with open(_INDEX_PATH, encoding='utf-8') as f:
        committed = json.load(f)

    assert set(committed) == set(BENCHMARK_REGISTRY.keys())


def test_index_only_points_at_discoverable_adapter_modules() -> None:
    """Every indexed module must also be found by the glob discovery contract."""
    with open(_INDEX_PATH, encoding='utf-8') as f:
        committed = json.load(f)

    discoverable = set(adapter_modules())
    unknown = {module for module in committed.values() if module not in discoverable}

    assert not unknown, f'index references modules the glob does not discover: {sorted(unknown)}'


# --- import cost -----------------------------------------------------------------


def _probe(body: str) -> Dict[str, Any]:
    """Run ``body`` in a clean interpreter and return the JSON dict it prints."""
    script = textwrap.dedent(body)
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module')
def fresh_interpreter() -> Dict[str, Any]:
    """Measure the whole import/resolve lifecycle in one subprocess.

    Every assertion here needs a pristine interpreter, and starting one costs
    seconds, so the observations are collected once and shared by the tests below
    rather than paying for a process per assertion.
    """
    return _probe(
        """
        import json, sys

        import evalscope
        from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark

        after_import = len(sys.modules)
        observations = {
            'modules_after_import': after_import,
            'registered_after_import': dict.__len__(BENCHMARK_REGISTRY),
            'shadowed_subpackages': [
                name for name in ('agent', 'benchmarks', 'evaluator')
                if getattr(evalscope, name, None) is not sys.modules.get('evalscope.' + name)
            ],
            'leaked_names': [
                name for name in (
                    'os', 'glob', 'time', 'importlib', 'json', 'sys',
                    'pattern', 'files', 'import_times', 'file_path', 'relative_path',
                    'module_path', 'full_path', 'start_time', 'end_time',
                )
                if hasattr(evalscope, name)
            ],
        }

        get_benchmark('gsm8k')
        observations['modules_grew_by_one_lookup'] = len(sys.modules) - after_import
        observations['registered_after_one_lookup'] = dict.__len__(BENCHMARK_REGISTRY)

        observations['registered_after_enumeration'] = len(list(BENCHMARK_REGISTRY.keys()))
        print(json.dumps(observations))
        """
    )


def test_importing_evalscope_does_not_load_every_adapter(fresh_interpreter: Dict[str, Any]) -> None:
    """Guards the regression this lazy registry exists to prevent.

    Thresholds are deliberately loose: they catch a return to eager loading of all
    ~240 adapters without pinning an exact module count.
    """
    assert fresh_interpreter['registered_after_import'] == 0, (
        'importing evalscope should register no benchmark up front'
    )
    assert fresh_interpreter['modules_after_import'] < 2500, (
        f"import evalscope pulled {fresh_interpreter['modules_after_import']} modules; eager loading is back"
    )


def test_resolving_one_benchmark_loads_only_a_few_modules(fresh_interpreter: Dict[str, Any]) -> None:
    assert fresh_interpreter['registered_after_one_lookup'] == 1, (
        'resolving one name must not register the whole catalog'
    )
    assert fresh_interpreter['modules_grew_by_one_lookup'] < 500, (
        f"resolving one benchmark pulled {fresh_interpreter['modules_grew_by_one_lookup']} extra modules"
    )


def test_full_enumeration_still_loads_the_whole_catalog(fresh_interpreter: Dict[str, Any]) -> None:
    assert fresh_interpreter['registered_after_enumeration'] >= 250


def test_subpackage_attributes_are_not_shadowed_by_star_imports(fresh_interpreter: Dict[str, Any]) -> None:
    """``evalscope.<subpackage>`` must be the subpackage, not a module inside it.

    ``from evalscope.evaluator import *`` used to rebind ``evalscope.evaluator`` to
    the inner ``evalscope.evaluator.evaluator`` module, which silently broke
    ``evalscope.evaluator.batch_reviewer`` and its siblings.
    """
    assert fresh_interpreter['shadowed_subpackages'] == []


def test_star_imports_do_not_leak_implementation_names_into_the_package(fresh_interpreter: Dict[str, Any]) -> None:
    """``import evalscope`` must not publish stdlib modules or loop temporaries."""
    assert fresh_interpreter['leaked_names'] == []


def test_an_unindexed_name_falls_back_to_loading_everything() -> None:
    """A stale index must cost time, never correctness.

    Emptying the in-memory index simulates a name the index does not know: the
    lookup has to fall back to the full load and still succeed. This needs its own
    interpreter because it mutates the index before the first lookup.
    """
    measured = _probe(
        """
        import json

        import evalscope
        import evalscope.benchmarks as benchmarks
        from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark

        benchmarks._INDEX.clear()
        adapter = get_benchmark('gsm8k')
        print(json.dumps({
            'resolved': adapter.benchmark_meta.name == 'gsm8k',
            'registered': dict.__len__(BENCHMARK_REGISTRY),
        }))
        """
    )

    assert measured['resolved'] is True
    assert measured['registered'] >= 250, 'fallback should have loaded the whole catalog'


def test_invalid_index_shape_falls_back_to_an_empty_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path = tmp_path / '_index.json'
    index_path.write_text('[]', encoding='utf-8')
    monkeypatch.setattr(benchmark_plugins, '_INDEX_PATH', str(index_path))

    assert benchmark_plugins._read_index() == {}


def test_a_mismatched_index_entry_falls_back_to_loading_everything() -> None:
    """An existing but wrong index target must not hide a valid benchmark."""
    measured = _probe(
        """
        import json

        import evalscope
        import evalscope.benchmarks as benchmarks
        from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark

        benchmarks._INDEX['gsm8k'] = benchmarks._INDEX['aime24']
        adapter = get_benchmark('gsm8k')
        print(json.dumps({
            'resolved': adapter.benchmark_meta.name == 'gsm8k',
            'registered': dict.__len__(BENCHMARK_REGISTRY),
        }))
        """
    )

    assert measured['resolved'] is True
    assert measured['registered'] >= 250, 'mismatched index should have triggered a full load'


def test_lazy_registry_serializes_concurrent_first_resolution() -> None:
    """A second thread must wait, not observe the first resolver's partial state."""
    registry = LazyRegistry[str]('test')
    first_started = Event()
    release_first = Event()
    second_finished = Event()
    results: List[Optional[str]] = []

    def resolve_one(name: str) -> bool:
        assert name == 'target'
        first_started.set()
        assert release_first.wait(timeout=5)
        registry['target'] = 'resolved'
        return True

    def read_target(mark_done: bool = False) -> None:
        results.append(registry.get('target'))
        if mark_done:
            second_finished.set()

    registry.set_resolvers(resolve_one, lambda: None, lambda name: None)
    first = Thread(target=read_target)
    second = Thread(target=read_target, kwargs={'mark_done': True})
    first.start()
    assert first_started.wait(timeout=5)
    second.start()
    assert not second_finished.wait(timeout=0.1), 'second reader returned before first resolution completed'
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert results == ['resolved', 'resolved']


def test_indexed_builtin_name_rejects_a_custom_registration() -> None:
    """A custom adapter must not shadow an indexed shipped benchmark before lookup."""
    measured = _probe(
        """
        import json

        import evalscope
        from evalscope.api.benchmark import BenchmarkMeta
        from evalscope.api.registry import register_benchmark

        try:
            @register_benchmark(BenchmarkMeta(name='gsm8k', dataset_id='custom'))
            class CustomAdapter:
                pass
        except ValueError:
            rejected = True
        else:
            rejected = False
        print(json.dumps({'rejected': rejected}))
        """
    )

    assert measured['rejected'] is True


def test_pop_materializes_a_lazy_entry_before_removing_it() -> None:
    """``pop`` is part of Registry's documented dict-compatible surface."""
    measured = _probe(
        """
        import json

        import evalscope
        from evalscope.api.registry import BENCHMARK_REGISTRY

        meta = BENCHMARK_REGISTRY.pop('gsm8k')
        print(json.dumps({
            'name': meta.name,
            'registered': dict.__len__(BENCHMARK_REGISTRY),
            'present_after_pop': dict.__contains__(BENCHMARK_REGISTRY, 'gsm8k'),
        }))
        """
    )

    assert measured == {'name': 'gsm8k', 'registered': 0, 'present_after_pop': False}


def test_setdefault_materializes_an_indexed_entry_before_returning_it() -> None:
    """``setdefault`` must not let a default value shadow a lazy built-in entry."""
    measured = _probe(
        """
        import json

        import evalscope
        from evalscope.api.registry import BENCHMARK_REGISTRY

        default = object()
        meta = BENCHMARK_REGISTRY.setdefault('gsm8k', default)
        print(json.dumps({
            'name': meta.name,
            'used_default': meta is default,
            'registered': dict.__len__(BENCHMARK_REGISTRY),
        }))
        """
    )

    assert measured == {'name': 'gsm8k', 'used_default': False, 'registered': 1}
