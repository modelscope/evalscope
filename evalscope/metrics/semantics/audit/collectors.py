"""Read-only collectors of the metric audit.

Each collector answers "which final report metric names can this repository emit?" from one
source, without importing or running an evaluation:

* :func:`collect_declared_metrics` -- ``BenchmarkMeta.metric_list`` of every registered benchmark
* :func:`collect_default_aggregation_names` -- names a registered aggregator produces
* :func:`collect_custom_aggregation_names` -- literal ``AggScore(metric_name=...)`` values found
  by an AST scan of custom ``aggregate_scores()`` implementations
* :func:`collect_observed_metrics` -- names seen in reports under an explicit ``--observed-path``
* :func:`collect_perf_field_keys` -- public perf field names

:func:`group_metric_records` sorts the result into the three mutually exclusive buckets
(default aggregation / custom aggregation / dynamic) and :func:`collect_metric_inventory`
assembles the whole inventory. Nothing here writes to disk.
"""

import ast
import inspect
import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from evalscope.metrics.semantics.naming import compose_final_metric_name
from evalscope.metrics.semantics.resolver import builtin_benchmark_names
from evalscope.utils import get_logger

if TYPE_CHECKING:
    from evalscope.api.benchmark.meta import BenchmarkMeta

logger = get_logger()

#: Prefix of every message this module logs, greppable in a CI log.
AUDIT_LOG_PREFIX = '[metric-audit]'

#: Directory of the ``evalscope`` package, the root of every source scan.
PACKAGE_DIR = Path(__file__).parents[3]

#: Package-relative directories that hold ``*_adapter.py`` sources.
ADAPTER_SOURCE_ROOTS: Tuple[str, ...] = ('benchmarks', 'api')

#: File name pattern of a benchmark adapter source.
ADAPTER_FILE_GLOB = '*_adapter.py'

#: Class name ``AggScore`` is constructed under, matched on the callee name.
AGG_SCORE_CLASS_NAME = 'AggScore'

#: Method name of the aggregation step of a ``DataAdapter``.
AGGREGATE_SCORES_METHOD = 'aggregate_scores'

#: Attribute deciding whether the aggregation name is prefixed to the metric name.
ADD_AGGREGATION_NAME_ATTR = 'add_aggregation_name'

#: Classes whose ``aggregate_scores()`` *is* the default aggregation path. A benchmark whose
#: first definition of the method in the MRO is one of these does not customize aggregation.
DEFAULT_AGGREGATION_CLASSES = frozenset({'DataAdapter', 'DefaultDataAdapter'})

#: Value ``DataAdapter.__init__`` gives ``add_aggregation_name``.
DEFAULT_ADD_AGGREGATION_NAME = True

#: Placeholder reported when a benchmark declares no metric at all, so its metric names come
#: from a custom scorer writing ``Score.value`` keys the audit cannot see statically.
EMPTY_METRIC_LIST_PATTERN = '<empty metric_list>'


class MetricGroup(str, Enum):
    """Bucket a final report metric name belongs to (requirement 10.7)."""

    DEFAULT_AGGREGATION = 'default_aggregation'
    """Produced by a registered aggregator over ``BenchmarkMeta.metric_list``."""

    CUSTOM_AGGREGATION = 'custom_aggregation'
    """Spelled out as a literal in a custom ``aggregate_scores()`` implementation."""

    DYNAMIC = 'dynamic'
    """Not statically spellable: a pattern, an unregistered aggregator, or observed only."""


#: Group kept when several collectors report the same name, strongest first.
GROUP_PRECEDENCE: Tuple[MetricGroup, ...] = (
    MetricGroup.CUSTOM_AGGREGATION,
    MetricGroup.DEFAULT_AGGREGATION,
    MetricGroup.DYNAMIC,
)

#: Order the three buckets are printed in: default aggregation, custom aggregation, dynamic.
GROUP_DISPLAY_ORDER: Tuple[MetricGroup, ...] = (
    MetricGroup.DEFAULT_AGGREGATION,
    MetricGroup.CUSTOM_AGGREGATION,
    MetricGroup.DYNAMIC,
)


class AggregatorBehaviour(BaseModel):
    """How a registered aggregator spells the names it emits.

    The value an aggregator writes into ``AggScore.aggregation_name`` is not always its own
    registered name: the ``*_at_k`` aggregators inject extra per-sample metrics and then delegate
    to ``Mean``, so their scores carry ``aggregation_name='mean'`` plus a suffixed metric name.
    Reflection cannot see that, hence this small explicit table -- kept honest by a test that
    runs every registered aggregator and compares the emitted names.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    aggregation_name: str
    """Value written to ``AggScore.aggregation_name``."""

    dynamic_metric_suffixes: Tuple[str, ...] = Field(default=())
    """Suffixes appended to the metric name for runtime-sized families, e.g. ``_pass@{k}``."""

    registered: bool = Field(default=True)
    """Whether the aggregation name resolves through the aggregation registry."""


#: Registered aggregation name -> the names it produces (requirement 10.3).
DEFAULT_AGGREGATOR_BEHAVIOURS: Dict[str, AggregatorBehaviour] = {
    'mean': AggregatorBehaviour(aggregation_name='mean'),
    'mean_and_pass_at_k': AggregatorBehaviour(aggregation_name='mean', dynamic_metric_suffixes=('_pass@{k}', )),
    'mean_and_vote_at_k': AggregatorBehaviour(aggregation_name='mean', dynamic_metric_suffixes=('_vote@{k}', )),
    'mean_and_pass_hat_k': AggregatorBehaviour(aggregation_name='mean', dynamic_metric_suffixes=('_pass^{k}', )),
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AggScoreConstruction(BaseModel):
    """One ``AggScore(...)`` construction found inside a custom ``aggregate_scores()``."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    metric_name: Optional[str] = Field(default=None)
    """Literal ``metric_name``, ``None`` when the argument is not a string literal."""

    metric_name_expr: Optional[str] = Field(default=None)
    """Source of a non-literal ``metric_name``, e.g. ``f'acc_{task_name}'``."""

    aggregation_name: Optional[str] = Field(default=None)
    """Literal ``aggregation_name``. Empty string when the argument is omitted."""

    aggregation_name_expr: Optional[str] = Field(default=None)
    """Source of a non-literal ``aggregation_name``, e.g. ``self.aggregation``."""

    lineno: int = Field(default=0)
    """Line of the construction in its source file."""

    def final_metric_name(self, add_aggregation_name: bool) -> str:
        """Spell the final report metric name, using ``<expr>`` for non-literal parts.

        Args:
            add_aggregation_name: Effective ``DataAdapter.add_aggregation_name`` of the adapter.

        Returns:
            The final report metric name, or a pattern containing ``<...>`` placeholders when a
            part of it is only known at runtime.
        """
        from evalscope.api.metric import AggScore

        agg_score = AggScore(
            metric_name=self.metric_name if self.metric_name is not None else f'<{self.metric_name_expr}>',
            aggregation_name=(
                self.aggregation_name if self.aggregation_name is not None else f'<{self.aggregation_name_expr}>'
            ),
        )
        return compose_final_metric_name(agg_score, add_aggregation_name)

    def is_static(self, add_aggregation_name: bool) -> bool:
        """Whether the final report metric name is fully known from the source alone.

        Args:
            add_aggregation_name: Effective ``DataAdapter.add_aggregation_name`` of the adapter.

        Returns:
            ``True`` when every part the name is composed of is a string literal. A non-literal
            ``aggregation_name`` does not matter when the adapter drops the prefix.
        """
        if self.metric_name is None:
            return False
        return self.aggregation_name is not None or not add_aggregation_name


class AdapterScan(BaseModel):
    """What the static scan of one adapter class found."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    source_path: str
    """Absolute path of the scanned source file."""

    class_name: str
    """Name of the scanned class."""

    overrides_aggregate_scores: bool = Field(default=False)
    """Whether the class body defines ``aggregate_scores()``."""

    calls_super_aggregation: bool = Field(default=False)
    """Whether the override calls ``super().aggregate_scores()``, keeping the default names."""

    add_aggregation_name: Optional[bool] = Field(default=None)
    """Literal value assigned to ``self.add_aggregation_name``, ``None`` when not assigned."""

    agg_score_constructions: List[AggScoreConstruction] = Field(default_factory=list)
    """``AggScore(...)`` constructions inside the override, in source order."""

    delegated_calls: List[str] = Field(default_factory=list)
    """Callees the override returns, e.g. ``aggregate_official_scores``. Names built there are
    outside the scanned function and therefore counted as dynamic."""

    @property
    def key(self) -> Tuple[str, str]:
        """Index key of this scan: source path and class name."""
        return (self.source_path, self.class_name)


class BenchmarkDeclaration(BaseModel):
    """What one registered benchmark declares about its metrics, without running anything."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    benchmark_name: str
    """Registered benchmark name, the catalog key."""

    declared_metric_names: List[str] = Field(default_factory=list)
    """Metric names of ``BenchmarkMeta.metric_list`` (requirement 10.2)."""

    aggregation: str = Field(default='mean')
    """``BenchmarkMeta.aggregation``: the registered aggregator, or a custom function name."""

    primary_metric: Optional[str] = Field(default=None)
    """``BenchmarkMeta.primary_metric``: the raw metric name declared as primary, if any."""

    add_aggregation_name: bool = Field(default=DEFAULT_ADD_AGGREGATION_NAME)
    """Effective ``DataAdapter.add_aggregation_name``, recovered from the adapter sources."""

    adapter_class_name: Optional[str] = Field(default=None)
    """Class name of ``BenchmarkMeta.data_adapter``."""

    aggregation_override: Optional[AdapterScan] = Field(default=None)
    """Scan of the class in the MRO that customizes ``aggregate_scores()``, if any."""

    @property
    def uses_default_aggregation(self) -> bool:
        """Whether the default aggregation path contributes names for this benchmark."""
        return self.aggregation_override is None or self.aggregation_override.calls_super_aggregation


class MetricRecord(BaseModel):
    """One final report metric name, or name pattern, of one benchmark."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    benchmark_name: str
    """Benchmark that emits the name."""

    metric_name: str
    """Final report metric name, or a pattern when ``is_pattern`` is set."""

    group: MetricGroup
    """Bucket of the name. Exactly one per name after :func:`group_metric_records`."""

    is_pattern: bool = Field(default=False)
    """Whether ``metric_name`` contains runtime placeholders instead of being a literal name."""

    sources: List[str] = Field(default_factory=list)
    """Provenance strings, one per collector that reported this name."""

    @property
    def key(self) -> Tuple[str, str]:
        """Identity of the record: benchmark name and metric name."""
        return (self.benchmark_name, self.metric_name)


class PerfFieldRecord(BaseModel):
    """One public perf field key reflected from the perf name constants."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    field_key: str
    """Field key as used by the perf archive API, e.g. ``'Avg Latency (s)'``."""

    holder: str
    """Class the constant is declared on: ``Metrics`` or ``PercentileMetrics``."""

    constant_name: str
    """Name of the class attribute, e.g. ``AVERAGE_LATENCY``."""


class MetricInventory(BaseModel):
    """The full read-only inventory: what can be emitted, grouped into three buckets."""

    model_config = ConfigDict(extra='forbid')

    declarations: Dict[str, BenchmarkDeclaration] = Field(default_factory=dict)
    """Benchmark name -> its declaration, for every audited benchmark."""

    default_aggregation: List[MetricRecord] = Field(default_factory=list)
    """Names of the default aggregation path."""

    custom_aggregation: List[MetricRecord] = Field(default_factory=list)
    """Names literally spelled out in custom ``aggregate_scores()`` implementations."""

    dynamic: List[MetricRecord] = Field(default_factory=list)
    """Patterns and names that are only known at runtime."""

    perf_field_keys: List[PerfFieldRecord] = Field(default_factory=list)
    """Public perf field keys (requirement 10.6)."""

    coverage_base: List[str] = Field(default_factory=list)
    """Benchmark names the catalog must cover, from ``evalscope/benchmarks/_meta/``."""

    observed_paths: List[str] = Field(default_factory=list)
    """Explicit paths observed metrics were read from. Empty for a default audit."""

    def grouped(self) -> Dict[MetricGroup, List[MetricRecord]]:
        """Return the three mutually exclusive buckets keyed by group."""
        return {
            MetricGroup.DEFAULT_AGGREGATION: list(self.default_aggregation),
            MetricGroup.CUSTOM_AGGREGATION: list(self.custom_aggregation),
            MetricGroup.DYNAMIC: list(self.dynamic),
        }

    def records(self) -> List[MetricRecord]:
        """Return every record of every bucket, sorted by benchmark and metric name."""
        return _sorted_records(self.default_aggregation + self.custom_aggregation + self.dynamic)


# ---------------------------------------------------------------------------
# Static adapter source scan
# ---------------------------------------------------------------------------


def _callee_name(node: ast.Call) -> Optional[str]:
    """Return the trailing name of a call target, or ``None`` for a computed callee."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_super_aggregation_call(node: ast.Call) -> bool:
    """Whether a call is ``super().aggregate_scores(...)``."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != AGGREGATE_SCORES_METHOD:
        return False
    inner = func.value
    return isinstance(inner, ast.Call) and _callee_name(inner) == 'super'


def _literal_keyword(node: ast.Call, name: str) -> Tuple[Optional[str], Optional[str]]:
    """Read one keyword argument of a call as a literal string.

    Args:
        node: Call to inspect.
        name: Keyword to read.

    Returns:
        ``(literal, None)`` for a string literal, ``('', None)`` when the keyword is absent
        (the ``AggScore`` field defaults to an empty string), and ``(None, source)`` when the
        argument is an expression that cannot be evaluated statically.
    """
    keyword = next((item for item in node.keywords if item.arg == name), None)
    if keyword is None:
        # ``**kwargs`` may still carry the field, so an unpacking keyword is not "absent".
        if any(item.arg is None for item in node.keywords):
            return None, '**kwargs'
        return '', None
    if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
        return keyword.value.value, None
    return None, ast.unparse(keyword.value)


def _scan_aggregate_scores(function: ast.AST) -> Tuple[List[AggScoreConstruction], bool, List[str]]:
    """Extract the metric names a custom ``aggregate_scores()`` spells out.

    Args:
        function: AST of the ``aggregate_scores`` definition.

    Returns:
        The ``AggScore`` constructions, whether ``super().aggregate_scores()`` is called, and the
        callees the function returns instead of building scores itself.
    """
    constructions: List[AggScoreConstruction] = []
    calls_super = False

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if _is_super_aggregation_call(node):
            calls_super = True
            continue
        if _callee_name(node) != AGG_SCORE_CLASS_NAME:
            continue
        metric_name, metric_expr = _literal_keyword(node, 'metric_name')
        aggregation_name, aggregation_expr = _literal_keyword(node, 'aggregation_name')
        constructions.append(
            AggScoreConstruction(
                metric_name=metric_name,
                metric_name_expr=metric_expr,
                aggregation_name=aggregation_name,
                aggregation_name_expr=aggregation_expr,
                lineno=node.lineno,
            )
        )

    delegated_calls = sorted({
        name
        for node in ast.walk(function) if isinstance(node, ast.Return) and node.value is not None
        for inner in ast.walk(node.value)
        if isinstance(inner, ast.Call) and (name := _callee_name(inner)) and name != AGG_SCORE_CLASS_NAME
    })
    return constructions, calls_super, delegated_calls


def _scan_add_aggregation_name(class_def: ast.ClassDef) -> Optional[bool]:
    """Read the literal ``self.add_aggregation_name`` assignment of a class body.

    Args:
        class_def: AST of the class to inspect.

    Returns:
        The last literal boolean assigned in the class body, or ``None`` when the class does not
        assign the attribute (then the value is inherited).
    """
    value: Optional[bool] = None
    for node in ast.walk(class_def):
        if not isinstance(node, ast.Assign):
            continue
        assigns_attr = any(
            isinstance(target, ast.Attribute) and target.attr == ADD_AGGREGATION_NAME_ATTR
            and isinstance(target.value, ast.Name) and target.value.id == 'self' for target in node.targets
        )
        if assigns_attr and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
            value = node.value.value
    return value


def _scan_source_file(path: Path) -> List[AdapterScan]:
    """Scan one adapter source file, returning a scan per class that carries a signal.

    Args:
        path: Adapter source file. Only read, never imported or executed.

    Returns:
        Scans of the classes that override ``aggregate_scores()`` or assign
        ``add_aggregation_name``. Unparsable files are skipped with a warning.
    """
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except (OSError, SyntaxError, ValueError) as error:
        logger.warning(f'{AUDIT_LOG_PREFIX} skipping unparsable adapter source {path}: {error}')
        return []

    source_path = str(path.resolve())
    scans: List[AdapterScan] = []
    for class_def in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        add_aggregation_name = _scan_add_aggregation_name(class_def)
        overrides = [
            node for node in class_def.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == AGGREGATE_SCORES_METHOD
        ]
        if not overrides and add_aggregation_name is None:
            continue

        constructions: List[AggScoreConstruction] = []
        calls_super = False
        delegated_calls: List[str] = []
        for override in overrides:
            found, found_super, found_calls = _scan_aggregate_scores(override)
            constructions.extend(found)
            calls_super = calls_super or found_super
            delegated_calls.extend(found_calls)

        scans.append(
            AdapterScan(
                source_path=source_path,
                class_name=class_def.name,
                overrides_aggregate_scores=bool(overrides),
                calls_super_aggregation=calls_super,
                add_aggregation_name=add_aggregation_name,
                agg_score_constructions=constructions,
                delegated_calls=sorted(set(delegated_calls)),
            )
        )
    return scans


def adapter_source_files(roots: Optional[Sequence[Path]] = None) -> List[Path]:
    """List the adapter source files of the audit scan.

    Args:
        roots: Directories to search. Defaults to the bundled adapter directories.

    Returns:
        Sorted, de-duplicated ``*_adapter.py`` paths.
    """
    search_roots = [PACKAGE_DIR / name for name in ADAPTER_SOURCE_ROOTS] if roots is None else list(roots)
    files = {path.resolve() for root in search_roots if root.is_dir() for path in root.rglob(ADAPTER_FILE_GLOB)}
    return sorted(files)


def scan_adapter_sources(roots: Optional[Sequence[Path]] = None) -> Dict[Tuple[str, str], AdapterScan]:
    """Statically scan the adapter sources for aggregation facts.

    Adapter classes are parsed instead of imported and instantiated, which keeps the audit
    read-only and independent of optional third-party packages (requirement 10.1).

    Args:
        roots: Directories to search. Defaults to the bundled adapter directories, whose scan is
            cached because those sources do not change while the process runs.

    Returns:
        ``(source_path, class_name)`` -> scan of that class.
    """
    if roots is None:
        return dict(_cached_adapter_scans())
    return {scan.key: scan for path in adapter_source_files(roots) for scan in _scan_source_file(path)}


@lru_cache(maxsize=1)
def _cached_adapter_scans() -> Dict[Tuple[str, str], AdapterScan]:
    """Cached scan of the bundled adapter sources."""
    return {scan.key: scan for path in adapter_source_files() for scan in _scan_source_file(path)}


def _class_source_path(cls: type) -> Optional[str]:
    """Return the resolved source path of a class, or ``None`` when it has none."""
    try:
        source = inspect.getsourcefile(cls)
    except TypeError:  # built-in classes such as ``object``
        return None
    return str(Path(source).resolve()) if source else None


def _mro_scans(adapter_cls: Optional[type], scans: Mapping[Tuple[str, str], AdapterScan]) -> List[AdapterScan]:
    """Return the scans of an adapter class MRO, most derived first.

    Args:
        adapter_cls: Adapter class of the benchmark, may be ``None``.
        scans: Index returned by :func:`scan_adapter_sources`.

    Returns:
        One scan per class of the MRO the scan index knows about, in MRO order. Classes whose
        source was not scanned simply do not appear.
    """
    if adapter_cls is None:
        return []
    resolved: List[AdapterScan] = []
    for cls in getattr(adapter_cls, '__mro__', ()):
        source_path = _class_source_path(cls)
        if source_path is None:
            continue
        scan = scans.get((source_path, cls.__name__))
        if scan is not None:
            resolved.append(scan)
    return resolved


def _resolve_add_aggregation_name(mro_scans: Sequence[AdapterScan]) -> bool:
    """Resolve the effective ``add_aggregation_name`` along the MRO."""
    for scan in mro_scans:
        if scan.add_aggregation_name is not None:
            return scan.add_aggregation_name
    return DEFAULT_ADD_AGGREGATION_NAME


def _resolve_aggregation_override(
    adapter_cls: Optional[type],
    mro_scans: Sequence[AdapterScan],
) -> Optional[AdapterScan]:
    """Return the scan of the class that customizes ``aggregate_scores()``, if any.

    The first class of the MRO defining the method wins, exactly like Python attribute lookup.
    When that class is one of :data:`DEFAULT_AGGREGATION_CLASSES` the benchmark uses the default
    aggregation path.

    Args:
        adapter_cls: Adapter class of the benchmark.
        mro_scans: Scans of the MRO, most derived first.

    Returns:
        The scan of the overriding class, or ``None`` when aggregation is not customized.
    """
    definer = next(
        (cls for cls in getattr(adapter_cls, '__mro__', ()) if AGGREGATE_SCORES_METHOD in vars(cls)),
        None,
    )
    if definer is None or definer.__name__ in DEFAULT_AGGREGATION_CLASSES:
        return None
    return next(
        (scan for scan in mro_scans if scan.class_name == definer.__name__ and scan.overrides_aggregate_scores),
        None,
    )


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def _metric_list_names(metric_list: Iterable[Union[str, Dict]]) -> List[str]:
    """Read the metric names of a ``BenchmarkMeta.metric_list``.

    Mirrors ``DefaultDataAdapter.match_score()``: a string entry is the metric name, a dict entry
    is keyed by the metric name and carries its keyword arguments.

    Args:
        metric_list: Raw ``metric_list`` value.

    Returns:
        Metric names in declaration order, without duplicates.
    """
    names: List[str] = []
    for entry in metric_list or ():
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict) and entry:
            name = next(iter(entry))
        else:
            continue
        if name not in names:
            names.append(name)
    return names


def _benchmark_registry() -> Mapping[str, 'BenchmarkMeta']:
    """Return the benchmark registry, importing the adapters so it is populated."""
    import evalscope.benchmarks  # noqa: F401  populates BENCHMARK_REGISTRY through decorators
    from evalscope.api.registry import BENCHMARK_REGISTRY

    return BENCHMARK_REGISTRY


def collect_declared_metrics(
    registry: Optional[Mapping[str, 'BenchmarkMeta']] = None,
    scans: Optional[Mapping[Tuple[str, str], AdapterScan]] = None,
    benchmarks: Optional[Iterable[str]] = None,
) -> Dict[str, BenchmarkDeclaration]:
    """Collect what every registered benchmark declares about its metrics (requirement 10.2).

    Reads ``BenchmarkMeta.metric_list`` and ``BenchmarkMeta.aggregation`` from the registry and
    completes them with the two aggregation facts recovered from the adapter sources:
    ``add_aggregation_name`` and the class customizing ``aggregate_scores()``.

    Args:
        registry: Benchmark name -> ``BenchmarkMeta``. Defaults to ``BENCHMARK_REGISTRY``.
        scans: Adapter source scans. Defaults to the bundled scan.
        benchmarks: Restrict the audit to these benchmark names. ``None`` audits all of them.

    Returns:
        Benchmark name -> declaration, ordered by benchmark name.
    """
    resolved_registry = _benchmark_registry() if registry is None else registry
    resolved_scans = scan_adapter_sources() if scans is None else scans
    selected = None if benchmarks is None else set(benchmarks)

    declarations: Dict[str, BenchmarkDeclaration] = {}
    for benchmark_name in sorted(resolved_registry):
        if selected is not None and benchmark_name not in selected:
            continue
        meta = resolved_registry[benchmark_name]
        adapter_cls = getattr(meta, 'data_adapter', None)
        mro_scans = _mro_scans(adapter_cls, resolved_scans)
        declarations[benchmark_name] = BenchmarkDeclaration(
            benchmark_name=benchmark_name,
            declared_metric_names=_metric_list_names(getattr(meta, 'metric_list', ()) or ()),
            aggregation=getattr(meta, 'aggregation', 'mean') or 'mean',
            primary_metric=getattr(meta, 'primary_metric', None),
            add_aggregation_name=_resolve_add_aggregation_name(mro_scans),
            adapter_class_name=getattr(adapter_cls, '__name__', None),
            aggregation_override=_resolve_aggregation_override(adapter_cls, mro_scans),
        )
    return declarations


def _aggregator_behaviour(aggregation: str) -> AggregatorBehaviour:
    """Return how an aggregation name spells its metric names.

    Args:
        aggregation: ``BenchmarkMeta.aggregation`` value.

    Returns:
        The declared behaviour of a registered aggregator, or a ``registered=False`` behaviour
        assuming the aggregator names its scores after itself. Names of an unregistered
        aggregator are reported as dynamic, since nothing static pins their spelling down.
    """
    behaviour = DEFAULT_AGGREGATOR_BEHAVIOURS.get(aggregation)
    if behaviour is not None:
        return behaviour
    return AggregatorBehaviour(aggregation_name=aggregation, registered=False)


def collect_default_aggregation_names(
    declarations: Mapping[str, BenchmarkDeclaration],
    add_dynamic_families: bool = True,
) -> List[MetricRecord]:
    """Derive the final report metric names of the default aggregation path (requirement 10.3).

    Composition goes through ``compose_final_metric_name()``, the single implementation of the
    final report metric name spelling rule shared with ``ReportGenerator`` (requirement 2.4).

    Args:
        declarations: Declarations from :func:`collect_declared_metrics`.
        add_dynamic_families: Whether to also emit the runtime-sized ``pass@k`` style patterns of
            the ``*_at_k`` aggregators.

    Returns:
        Records of the default path: ``default_aggregation`` for statically spelled names,
        ``dynamic`` for runtime-sized families and unregistered aggregators.
    """
    from evalscope.api.metric import AggScore

    records: List[MetricRecord] = []
    for declaration in declarations.values():
        if not declaration.uses_default_aggregation:
            continue
        behaviour = _aggregator_behaviour(declaration.aggregation)
        via_super = ' via super()' if declaration.aggregation_override is not None else ''

        if not declaration.declared_metric_names:
            records.append(
                MetricRecord(
                    benchmark_name=declaration.benchmark_name,
                    metric_name=EMPTY_METRIC_LIST_PATTERN,
                    group=MetricGroup.DYNAMIC,
                    is_pattern=True,
                    sources=[
                        'BenchmarkMeta.metric_list is empty, the metric names are written by a '
                        f"custom scorer and aggregated by '{declaration.aggregation}'{via_super}"
                    ],
                )
            )
            continue

        for metric_name in declaration.declared_metric_names:
            base = compose_final_metric_name(
                AggScore(metric_name=metric_name, aggregation_name=behaviour.aggregation_name),
                declaration.add_aggregation_name,
            )
            if behaviour.registered:
                source = f"metric_list metric '{metric_name}' aggregated by '{declaration.aggregation}'{via_super}"
                group = MetricGroup.DEFAULT_AGGREGATION
                is_pattern = False
            else:
                source = (
                    f"metric_list metric '{metric_name}' aggregated by unregistered "
                    f"aggregation '{declaration.aggregation}'{via_super}"
                )
                group = MetricGroup.DYNAMIC
                is_pattern = True
            records.append(
                MetricRecord(
                    benchmark_name=declaration.benchmark_name,
                    metric_name=base,
                    group=group,
                    is_pattern=is_pattern,
                    sources=[source],
                )
            )

            if not add_dynamic_families:
                continue
            for suffix in behaviour.dynamic_metric_suffixes:
                records.append(
                    MetricRecord(
                        benchmark_name=declaration.benchmark_name,
                        metric_name=compose_final_metric_name(
                            AggScore(
                                metric_name=f'{metric_name}{suffix}',
                                aggregation_name=behaviour.aggregation_name,
                            ),
                            declaration.add_aggregation_name,
                        ),
                        group=MetricGroup.DYNAMIC,
                        is_pattern=True,
                        sources=[
                            f"metric_list metric '{metric_name}' expanded by "
                            f"'{declaration.aggregation}' for every k <= repeats"
                        ],
                    )
                )
    return _sorted_records(records)


def collect_custom_aggregation_names(declarations: Mapping[str, BenchmarkDeclaration]) -> List[MetricRecord]:
    """Collect the names custom ``aggregate_scores()`` implementations spell out (req 10.4).

    Literal ``AggScore(metric_name=..., aggregation_name=...)`` arguments become
    ``custom_aggregation`` records. Anything the AST cannot evaluate -- an f-string, a variable, a
    delegation to a helper defined elsewhere -- becomes a ``dynamic`` pattern instead, which the
    catalog covers through ``dynamic_metric_names`` (requirement 9.4).

    Args:
        declarations: Declarations from :func:`collect_declared_metrics`.

    Returns:
        Records of the custom aggregation path.
    """
    records: List[MetricRecord] = []
    for declaration in declarations.values():
        override = declaration.aggregation_override
        if override is None:
            continue
        location = f'{_relative_source(override.source_path)}::{override.class_name}.{AGGREGATE_SCORES_METHOD}'

        for construction in override.agg_score_constructions:
            is_static = construction.is_static(declaration.add_aggregation_name)
            records.append(
                MetricRecord(
                    benchmark_name=declaration.benchmark_name,
                    metric_name=construction.final_metric_name(declaration.add_aggregation_name),
                    group=MetricGroup.CUSTOM_AGGREGATION if is_static else MetricGroup.DYNAMIC,
                    is_pattern=not is_static,
                    sources=[f'{location}:{construction.lineno}'],
                )
            )

        if not override.agg_score_constructions:
            # The override builds its scores somewhere else, e.g. in a helper of a sibling
            # module. Report it as dynamic so the gap is visible instead of silently missing.
            delegated = ', '.join(f'{name}()' for name in override.delegated_calls) or 'no AggScore construction found'
            records.append(
                MetricRecord(
                    benchmark_name=declaration.benchmark_name,
                    metric_name=f'<{delegated}>',
                    group=MetricGroup.DYNAMIC,
                    is_pattern=True,
                    sources=[location],
                )
            )
    return _sorted_records(records)


def collect_observed_metrics(
    observed_paths: Iterable[Union[str, Path]],
    benchmarks: Optional[Iterable[str]] = None,
) -> List[MetricRecord]:
    """Collect final report metric names seen in report files under explicit paths (req 10.5).

    Only the paths passed in are read. A default audit passes none, so it never depends on the
    contents of the workspace ``outputs/`` directory; observing historical reports or test
    fixtures is an explicit opt-in.

    Args:
        observed_paths: Report files or directories to walk. Files are opened read-only.
        benchmarks: Restrict the result to these benchmark names.

    Returns:
        Records with group ``dynamic``. Names that a static collector also reports are folded
        back into their static group by :func:`group_metric_records`.
    """
    selected = None if benchmarks is None else set(benchmarks)
    records: List[MetricRecord] = []

    for path in _observed_report_files(observed_paths):
        for benchmark_name, metric_name in _read_report_metric_names(path):
            if selected is not None and benchmark_name not in selected:
                continue
            records.append(
                MetricRecord(
                    benchmark_name=benchmark_name,
                    metric_name=metric_name,
                    group=MetricGroup.DYNAMIC,
                    sources=[f'observed in {_relative_source(str(path))}'],
                )
            )
    return _sorted_records(records)


def _observed_report_files(observed_paths: Iterable[Union[str, Path]]) -> List[Path]:
    """Expand the explicit observed paths into JSON files, keeping them sorted and unique."""
    files: List[Path] = []
    for raw_path in observed_paths or ():
        path = Path(raw_path).expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob('*.json')))
        elif path.is_file():
            files.append(path)
        else:
            logger.warning(f'{AUDIT_LOG_PREFIX} observed path does not exist: {path}')
    unique: List[Path] = []
    for path in files:
        if path not in unique:
            unique.append(path)
    return unique


def _read_report_metric_names(path: Path) -> List[Tuple[str, str]]:
    """Read ``(dataset_name, metric.name)`` pairs of a report file.

    Args:
        path: JSON file that may hold a report, or a list of reports.

    Returns:
        The pairs found, empty when the file is not a report. The file is only read.
    """
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        logger.warning(f'{AUDIT_LOG_PREFIX} skipping unreadable observed file {path}: {error}')
        return []

    candidates = payload if isinstance(payload, list) else [payload]
    pairs: List[Tuple[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        benchmark_name = candidate.get('dataset_name')
        metrics = candidate.get('metrics')
        if not isinstance(benchmark_name, str) or not isinstance(metrics, list):
            continue
        for metric in metrics:
            metric_name = metric.get('name') if isinstance(metric, dict) else None
            if isinstance(metric_name, str) and (benchmark_name, metric_name) not in pairs:
                pairs.append((benchmark_name, metric_name))
    return pairs


def collect_perf_field_keys() -> List[PerfFieldRecord]:
    """Reflect the public perf field keys from the perf name constants (requirement 10.6).

    Returns:
        One record per public string constant of ``Metrics`` and ``PercentileMetrics``, sorted by
        field key. Empty when the perf extra is unavailable.
    """
    try:
        from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics
    except ImportError:  # pragma: no cover - the perf constants ship with the package
        logger.warning(f'{AUDIT_LOG_PREFIX} perf constants are unavailable, skipping perf field keys')
        return []

    records: List[PerfFieldRecord] = []
    for holder in (Metrics, PercentileMetrics):
        for constant_name, value in vars(holder).items():
            if constant_name.startswith('_') or not isinstance(value, str):
                continue
            records.append(PerfFieldRecord(field_key=value, holder=holder.__name__, constant_name=constant_name))
    return sorted(records, key=lambda record: (record.field_key, record.holder, record.constant_name))


# ---------------------------------------------------------------------------
# Grouping and inventory
# ---------------------------------------------------------------------------


def _sorted_records(records: Iterable[MetricRecord]) -> List[MetricRecord]:
    """Sort records by benchmark name then metric name for deterministic output."""
    return sorted(records, key=lambda record: (record.benchmark_name, record.metric_name))


def _relative_source(path: str) -> str:
    """Render a path relative to the repository root when it lives inside the package."""
    try:
        return str(Path(path).resolve().relative_to(PACKAGE_DIR.parent))
    except ValueError:
        return path


def group_metric_records(*record_lists: Iterable[MetricRecord]) -> Dict[MetricGroup, List[MetricRecord]]:
    """Merge collector output into the three mutually exclusive buckets (requirement 10.7).

    Records of the same ``(benchmark, metric)`` pair are merged: the strongest group in
    :data:`GROUP_PRECEDENCE` wins and the provenance strings of every collector are kept, so a
    name reported by two collectors appears exactly once, in one bucket.

    Args:
        *record_lists: Record lists of the individual collectors, in any order.

    Returns:
        Group -> records, each list sorted by benchmark and metric name.
    """
    merged: Dict[Tuple[str, str], MetricRecord] = {}
    for records in record_lists:
        for record in records:
            existing = merged.get(record.key)
            if existing is None:
                merged[record.key] = record.model_copy(deep=True)
                continue
            strongest = min(
                (existing.group, record.group),
                key=GROUP_PRECEDENCE.index,
            )
            sources = list(existing.sources)
            sources.extend(source for source in record.sources if source not in sources)
            merged[record.key] = existing.model_copy(
                update={
                    'group': strongest,
                    'is_pattern': existing.is_pattern and record.is_pattern,
                    'sources': sources,
                }
            )

    grouped: Dict[MetricGroup, List[MetricRecord]] = {group: [] for group in GROUP_PRECEDENCE}
    for record in merged.values():
        grouped[record.group].append(record)
    return {group: _sorted_records(records) for group, records in grouped.items()}


def collect_metric_inventory(
    benchmarks: Optional[Iterable[str]] = None,
    observed_paths: Iterable[Union[str, Path]] = (),
    registry: Optional[Mapping[str, 'BenchmarkMeta']] = None,
    scans: Optional[Mapping[Tuple[str, str], AdapterScan]] = None,
) -> MetricInventory:
    """Run every collector and return the grouped, read-only inventory.

    Args:
        benchmarks: Restrict the audit to these benchmark names. ``None`` audits all of them.
        observed_paths: Explicit report paths to observe. Empty by default, so a default audit
            never reads the workspace ``outputs/`` directory (requirement 10.5).
        registry: Benchmark registry override, for tests.
        scans: Adapter source scan override, for tests.

    Returns:
        The inventory: declarations, the three mutually exclusive buckets, the public perf field
        keys and the catalog coverage base.
    """
    declarations = collect_declared_metrics(registry=registry, scans=scans, benchmarks=benchmarks)
    grouped = group_metric_records(
        collect_custom_aggregation_names(declarations),
        collect_default_aggregation_names(declarations),
        collect_observed_metrics(observed_paths, benchmarks=benchmarks),
    )
    return MetricInventory(
        declarations=declarations,
        default_aggregation=grouped[MetricGroup.DEFAULT_AGGREGATION],
        custom_aggregation=grouped[MetricGroup.CUSTOM_AGGREGATION],
        dynamic=grouped[MetricGroup.DYNAMIC],
        perf_field_keys=collect_perf_field_keys(),
        coverage_base=sorted(builtin_benchmark_names()),
        observed_paths=[str(path) for path in _observed_report_files(observed_paths)],
    )
