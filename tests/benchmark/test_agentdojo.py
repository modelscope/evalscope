"""Tests for the AgentDojo benchmark.

Upstream-free tests (pure logic, import isolation, simulated upstream failures and construction
errors) run without the optional ``agentdojo`` package. Classes marked ``requires_upstream`` need it and are skipped
otherwise; they drive the adapter with the offline ``agentdojo_scripted`` model API defined below, so nothing
touches the network or needs a model API key.
"""

import asyncio
import contextlib
import copy
import datetime
import functools
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field

import evalscope
from evalscope.agent.external.config import ExternalAgentConfig
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.evaluator.cache import ModelResult
from evalscope.api.messages import ChatMessage, ChatMessageAssistant
from evalscope.api.metric import SampleScore, Score
from evalscope.api.metric.semantics import MetricDirection
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput, get_model
from evalscope.api.model.model_output import ChatCompletionChoice, StopReason
from evalscope.api.registry import (
    BENCHMARK_REGISTRY,
    MODEL_APIS,
    get_benchmark,
    get_model_api,
    get_strategy,
    register_model_api,
)
from evalscope.api.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo
from evalscope.benchmarks.agentdojo import agentdojo_adapter
from evalscope.benchmarks.agentdojo import upstream as upstream_module
from evalscope.benchmarks.agentdojo.agentdojo_adapter import row_id
from evalscope.benchmarks.agentdojo.constants import (
    AGENTDOJO_INSTALL_HINT,
    BENCHMARK_VERSION,
    BENIGN_ATTACK,
    DEFAULT_ATTACK,
    DEFAULT_MODEL_NAME,
    RUN_RECORD_META_KEY,
    SAMPLE_META_KEY,
    SUITES,
    SUPPORTED_ATTACKS,
    UPSTREAM_VERSION,
)
from evalscope.benchmarks.agentdojo.injections import (
    InjectionPlanner,
    resolve_effective_model_name,
    resolve_victim_model_name,
)
from evalscope.benchmarks.agentdojo.scoring import (
    AgentDojoSampleMeta,
    EnvironmentCreationError,
    ScoringPayload,
    check_results,
    create_environment,
    restore_environment,
)
from evalscope.benchmarks.agentdojo.tool_binding import (
    ARGUMENT_PARSE_ERROR,
    EMPTY_FUNCTION_NAME_ERROR,
    INVALID_TOOL_ERROR,
    UPSTREAM_EMPTY_FUNCTION_NAME,
    AgentDojoStrategy,
    AgentDojoToolBinding,
    to_jsonable,
)
from evalscope.benchmarks.agentdojo.upstream import (
    UPSTREAM_DISTRIBUTION,
    UpstreamContext,
    UpstreamDataError,
    UpstreamUnavailableError,
    UpstreamVersionError,
    load_upstream,
)
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus
from evalscope.report import Report
from evalscope.run import run_task
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from evalscope.utils.io_utils import JsonlWriter, convert_normal_types, jsonl_to_list

requires_upstream = pytest.mark.skipif(
    importlib.util.find_spec('agentdojo') is None, reason="requires pip install 'evalscope[agentdojo]'"
)

SUITE = 'banking'
USER_TASK_ID = 'user_task_0'
"""Its ground truth does not read injected content, so one script serves benign and attack samples alike."""

INJECTION_TASK_ID = 'injection_task_0'
VICTIM_MODEL_NAME = 'Test Victim'
EVALSCOPE_UNKNOWN_TOOL_TEXT = 'is not registered. Available'
"""Fragment of EvalScope's own unknown-tool observation; AgentDojo observations must never contain it."""

# ---------------------------------------------------------------------------
# Scripted model API
# ---------------------------------------------------------------------------
#
# The reply is a pure function of the input: the suite is identified from the tool names, the user task from the
# first user message, and the turn index is the number of assistant messages already present. Scripts are installed
# per ``(suite, user_task_id)``; every sample of one user task shares the script. A turn index past the end of a
# script yields an empty text reply, which ends the loop as ``completed`` with ``empty_output=True``.

SCRIPTED_EVAL_TYPE = 'agentdojo_scripted'
SCRIPTED_MODEL_NAME = 'agentdojo-scripted'

ScriptKey = Tuple[str, str]


class ScriptedToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    function: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    parse_error: Optional[str] = None


def text_turn(text: str = '', stop_reason: StopReason = 'stop') -> ModelOutput:
    return ModelOutput.from_content(model=SCRIPTED_MODEL_NAME, content=text, stop_reason=stop_reason)


def tool_calls_turn(*calls: Tuple[str, Mapping[str, Any]], text: str = '') -> ModelOutput:
    tool_calls = [
        ToolCall(id=f'call_{index}', function=ToolFunction(name=name, arguments=dict(args)))
        for index, (name, args) in enumerate(calls)
    ]
    message = ChatMessageAssistant(content=text, model=SCRIPTED_MODEL_NAME, source='generate', tool_calls=tool_calls)
    return ModelOutput(
        model=SCRIPTED_MODEL_NAME, choices=[ChatCompletionChoice(message=message, stop_reason='tool_calls')]
    )


@functools.lru_cache(maxsize=1)
def _suite_lookup() -> Tuple[Dict[FrozenSet[str], str], Dict[Tuple[str, str], str]]:
    """Return ``({tool name set: suite}, {(suite, prompt): user_task_id})``."""
    by_tools: Dict[FrozenSet[str], str] = {}
    by_prompt: Dict[Tuple[str, str], str] = {}
    for suite_name, suite in load_upstream().suites.items():
        by_tools[frozenset(tool.name for tool in suite.tools)] = suite_name
        for user_task_id, user_task in suite.user_tasks.items():
            by_prompt[(suite_name, user_task.PROMPT)] = user_task_id
    return by_tools, by_prompt


class _ScriptedAgentDojoAPI(ModelAPI):
    """Stateless scripted model; scripts and the call log are class-level (models are memoized and threaded)."""

    _lock = threading.Lock()
    _scripts: Dict[ScriptKey, Tuple[ModelOutput, ...]] = {}
    _calls: List[Tuple[str, str, int]] = []

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, config=config)

    def generate(
        self,
        input: List[ChatMessage],  # noqa: A002
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        by_tools, by_prompt = _suite_lookup()
        suite = by_tools[frozenset(tool.name for tool in tools)]
        prompt = next(message.text for message in input if message.role == 'user')
        user_task_id = by_prompt[(suite, prompt)]
        turn_index = sum(1 for message in input if message.role == 'assistant')
        cls = type(self)
        with cls._lock:
            cls._calls.append((suite, user_task_id, turn_index))
            script = cls._scripts.get((suite, user_task_id))
        if script is None:
            raise LookupError(f'agentdojo_scripted: no script installed for {(suite, user_task_id)}.')
        output = script[turn_index].model_copy(deep=True) if turn_index < len(script) else text_turn()
        output.model = self.model_name
        for index, call in enumerate(output.message.tool_calls or []):
            call.id = f'scripted_{turn_index}_{index}'
        return output


def _scripted_api() -> type:
    if not MODEL_APIS.is_registered(SCRIPTED_EVAL_TYPE):
        register_model_api(SCRIPTED_EVAL_TYPE)(_ScriptedAgentDojoAPI)
    return get_model_api(SCRIPTED_EVAL_TYPE)


@contextlib.contextmanager
def installed_scripts(scripts: Mapping[ScriptKey, Sequence[ModelOutput]]) -> Iterator[None]:
    """Install scripts for the block and clear the call log on entry."""
    api = _scripted_api()
    with api._lock:
        previous = dict(api._scripts)
        api._scripts.clear()
        api._scripts.update({key: tuple(turns) for key, turns in scripts.items()})
        api._calls.clear()
    try:
        yield
    finally:
        with api._lock:
            api._scripts.clear()
            api._scripts.update(previous)


def model_calls() -> List[Tuple[str, str, int]]:
    """One ``(suite, user_task_id, turn_index)`` entry per ``generate`` call since the last reset."""
    api = _scripted_api()
    with api._lock:
        return list(api._calls)


def reset_model_calls() -> None:
    api = _scripted_api()
    with api._lock:
        api._calls.clear()


def ground_truth_turns(
    suite: str, user_task_id: str, injections: Optional[Mapping[str, str]] = None
) -> List[ModelOutput]:
    """One tool-call turn per ``ground_truth()`` call on the sample's environment, then ``GROUND_TRUTH_OUTPUT``."""
    task_suite = load_upstream().suites[suite]
    user_task = task_suite.get_user_task_by_id(user_task_id)
    environment = user_task.init_environment(task_suite.load_and_inject_default_environment(dict(injections or {})))
    turns: List[ModelOutput] = [
        tool_calls_turn((call.function, dict(call.args))) for call in user_task.ground_truth(environment)
    ]
    return [*turns, text_turn(text=user_task.GROUND_TRUTH_OUTPUT)]


def make_task_config(
    work_dir: Union[str, Path],
    *,
    subset: Union[str, Sequence[str]] = SUITE,
    attack: str = BENIGN_ATTACK,
    user_task_ids: Optional[Sequence[str]] = None,
    injection_task_ids: Optional[Sequence[str]] = None,
    victim_model_name: Optional[str] = None,
    limit: Optional[Union[int, float]] = None,
    eval_batch_size: int = 1,
    **overrides: Any,
) -> TaskConfig:
    """Offline AgentDojo ``TaskConfig`` driven by the scripted model; ``overrides`` go to ``TaskConfig`` as-is."""
    _scripted_api()
    extra_params: Dict[str, Any] = {'attack': attack}
    for name, value in (
        ('user_task_ids', user_task_ids),
        ('injection_task_ids', injection_task_ids),
        ('victim_model_name', victim_model_name),
    ):
        if value is not None:
            extra_params[name] = list(value) if name != 'victim_model_name' else value
    return TaskConfig(
        model=SCRIPTED_MODEL_NAME,
        eval_type=SCRIPTED_EVAL_TYPE,
        datasets=['agentdojo'],
        dataset_args={
            'agentdojo': {
                'subset_list': [subset] if isinstance(subset, str) else list(subset),
                'extra_params': extra_params,
            }
        },
        limit=limit,
        eval_batch_size=eval_batch_size,
        work_dir=str(work_dir),
        no_timestamp=True,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Shared helpers and fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def upstream() -> UpstreamContext:
    return load_upstream()


@pytest.fixture(scope='module')
def work_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Work and output directory shared by the adapters that are built once per module."""
    return str(tmp_path_factory.mktemp('agentdojo_work'))


# Upstream modules whose tools stamp ``datetime.datetime.now()`` into the environment and the observation.
_CLOCK_STAMPING_MODULES = (
    'agentdojo.default_suites.v1.tools.email_client',
    'agentdojo.default_suites.v1.tools.cloud_drive_client',
)


@pytest.fixture(scope='class')
def frozen_tool_clock() -> Iterator[None]:
    """Freeze the clock of the stamping upstream tools so runs started at different instants are comparable."""
    frozen_now = datetime.datetime(2024, 5, 15, 12, 0, 0)

    class FrozenDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz: Optional[datetime.tzinfo] = None) -> datetime.datetime:
            return frozen_now

    clock = types.SimpleNamespace(datetime=FrozenDatetime)
    with pytest.MonkeyPatch.context() as patch:
        for module_name in _CLOCK_STAMPING_MODULES:
            patch.setattr(importlib.import_module(module_name), 'datetime', clock)
        yield


@pytest.fixture(scope='class')
def loop_runner_cleanup() -> Iterator[None]:
    """Stop the asyncio loop thread used by direct ``run_inference`` calls instead of leaving it to interpreter exit."""
    yield
    AsyncioLoopRunner.shutdown_for_thread()


class _Call(NamedTuple):
    name: str
    args: Dict[str, Any]


@functools.lru_cache(maxsize=None)
def _ground_truth_pool(suite_name: str) -> Tuple[_Call, ...]:
    """Ground-truth calls of every user and injection task of the suite on the benign environment."""
    suite = load_upstream().suites[suite_name]
    base = suite.load_and_inject_default_environment({})
    pool: List[_Call] = []
    for _, task in sorted(suite.user_tasks.items()):
        pool.extend(
            _Call(c.function, dict(c.args)) for c in task.ground_truth(task.init_environment(copy.deepcopy(base)))
        )
    for _, task in sorted(suite.injection_tasks.items()):
        pool.extend(_Call(c.function, dict(c.args)) for c in task.ground_truth(copy.deepcopy(base)))
    return tuple(pool)


@functools.lru_cache(maxsize=None)
def _tool_names(suite_name: str) -> FrozenSet[str]:
    return frozenset(tool.name for tool in load_upstream().suites[suite_name].tools)


_BAD_VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=10),
    st.sampled_from(["['a', 'b']", '[1, 2]', '[1', "('a',)", '[]', "{'a': 1}"]),
    st.lists(st.integers(min_value=0, max_value=9), max_size=3),
)


@st.composite
def _mutated_calls(draw: st.DrawFn, suite_name: str) -> ScriptedToolCall:
    """A ground-truth call kept as-is or mutated into a failing, stringified, unknown, empty or unparsable call."""
    base = draw(st.sampled_from(_ground_truth_pool(suite_name)))
    name, args = base.name, copy.deepcopy(base.args)
    keys = sorted(args)
    kind = draw(st.sampled_from(['ground_truth', 'drop_arg', 'alter_arg', 'stringify', 'empty', 'unknown', 'parse']))
    if kind == 'drop_arg' and keys:
        del args[draw(st.sampled_from(keys))]
    elif kind == 'alter_arg' and keys:
        args[draw(st.sampled_from(keys))] = draw(_BAD_VALUES)
    elif kind == 'stringify' and keys:
        key = draw(st.sampled_from(keys))
        args[key] = str(args[key] if isinstance(args[key], list) else [args[key]])
    elif kind == 'empty':
        name = draw(st.sampled_from(['', UPSTREAM_EMPTY_FUNCTION_NAME]))
    elif kind == 'unknown':
        name = draw(
            st.one_of(st.just('submit'), st.text(min_size=1, max_size=12)).filter(
                lambda s: s not in _tool_names(suite_name) and s != UPSTREAM_EMPTY_FUNCTION_NAME
            )
        )
    elif kind == 'parse':
        return ScriptedToolCall(function=name, arguments=args, parse_error=draw(st.text(min_size=1, max_size=20)))
    return ScriptedToolCall(function=name, arguments=args)


def _meta(
    suite: str,
    user_task_id: str,
    injection_task_id: Optional[str] = None,
    injections: Optional[Mapping[str, str]] = None,
) -> AgentDojoSampleMeta:
    return AgentDojoSampleMeta(
        row_id=row_id(suite, user_task_id, injection_task_id),
        suite=suite,
        user_task_id=user_task_id,
        injection_task_id=injection_task_id,
        attack=BENIGN_ATTACK if injection_task_id is None else DEFAULT_ATTACK,
        victim_model_name=VICTIM_MODEL_NAME,
        effective_model_name=VICTIM_MODEL_NAME,
        injections=dict(injections or {}),
        benchmark_version=BENCHMARK_VERSION,
        upstream_version=UPSTREAM_VERSION,
    )


@functools.lru_cache(maxsize=1)
def _default_planner() -> InjectionPlanner:
    return InjectionPlanner(load_upstream(), DEFAULT_ATTACK, VICTIM_MODEL_NAME)


def _make_sample(adapter: Any, suite: str, user_task_id: str, injection_task_id: Optional[str]) -> Sample:
    """Convert one row through the adapter, without loading a whole suite; the sample id is 0."""
    record = {'suite': suite, 'user_task_id': user_task_id, 'injection_task_id': injection_task_id}
    sample = adapter.record_to_sample(record)
    sample.id = 0
    sample.group_id = 0
    return sample


def _round_trip(state: TaskState) -> TaskState:
    """Persist a TaskState like ``CacheManager.save_prediction_cache`` and restore it like ``filter_prediction_cache``.

    The restore dataset carries no metadata, so the scoring payload can only come from the cache row.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = os.path.join(tmp_dir, 'agentdojo.jsonl')
        writer = JsonlWriter(cache_file)
        try:
            writer.write(convert_normal_types(ModelResult.from_task_state(state, save_metadata=True).model_dump()))
        finally:
            writer.close()
        rows = jsonl_to_list(cache_file)
    assert len(rows) == 1
    blank = MemoryDataset([Sample(input='', id=index, metadata={}) for index in range(state.sample_id + 1)])
    restored = ModelResult.model_validate(rows[0]).to_task_state(dataset=blank)
    assert restored is not None
    return restored


def _raise(exc: BaseException) -> Callable[..., Any]:
    def raiser(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return raiser


# ===========================================================================
# Upstream-free tests
# ===========================================================================

EVALSCOPE_ROOT = Path(evalscope.__file__).resolve().parent
AGENTDOJO_PACKAGE_DIR = EVALSCOPE_ROOT / 'benchmarks' / 'agentdojo'


_ISOLATION_SCRIPT = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import pkgutil
    import sys


    class _BlockUpstream(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == 'agentdojo' or name.startswith('agentdojo.'):
                raise ModuleNotFoundError(f'No module named {name!r} (blocked)', name=name)
            return None


    sys.meta_path.insert(0, _BlockUpstream())
    import evalscope.benchmarks as benchmarks
    import evalscope.benchmarks.agentdojo as package
    from evalscope.api.registry import get_benchmark

    modules = [info.name for info in pkgutil.walk_packages(package.__path__, prefix=f'{package.__name__}.')]
    for name in modules:
        importlib.import_module(name)
    benchmarks.load_all()
    assert type(get_benchmark('agentdojo')).__name__ == 'AgentDojoAdapter'
    leaked = sorted(name for name in sys.modules if name == 'agentdojo' or name.startswith('agentdojo.'))
    assert not leaked, f'upstream modules were imported: {leaked}'
    print('IMPORTED', ','.join(sorted(modules)))
    """
)


class TestStatic:
    def test_registration(self) -> None:
        meta = BENCHMARK_REGISTRY['agentdojo']
        assert meta.data_adapter is agentdojo_adapter.AgentDojoAdapter
        assert meta.subset_list == list(SUITES)
        assert (meta.metric_list, meta.primary_metric.name) == (['utility', 'targeted_asr'], 'targeted_asr')

    @pytest.mark.parametrize(
        ('metric', 'display_name', 'direction'),
        [
            ('utility', 'Utility', MetricDirection.HIGHER_IS_BETTER),
            ('targeted_asr', 'Targeted ASR', MetricDirection.LOWER_IS_BETTER),
        ],
    )
    def test_metric_catalog(self, metric: str, display_name: str, direction: MetricDirection) -> None:
        from evalscope.metrics.semantics.catalog import METRIC_DEFINITIONS

        semantics = METRIC_DEFINITIONS[metric].resolve(metric)
        assert (semantics.metric_name, semantics.direction) == (display_name, direction)

    def test_generic_function_calling_strategy_is_not_replaced(self) -> None:
        from evalscope.agent.strategies.function_calling import FunctionCallingStrategy

        assert get_strategy('function_calling') is FunctionCallingStrategy
        assert issubclass(AgentDojoStrategy, FunctionCallingStrategy)

    def test_package_and_registry_work_without_upstream(self) -> None:
        proc = subprocess.run(
            [sys.executable, '-c', _ISOLATION_SCRIPT],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(EVALSCOPE_ROOT.parent),
        )
        assert proc.returncode == 0, f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
        imported_line = next(line for line in proc.stdout.splitlines() if line.startswith('IMPORTED '))
        imported = set(imported_line[len('IMPORTED ') :].split(','))
        expected = {
            f'evalscope.benchmarks.agentdojo.{path.stem}'
            for path in AGENTDOJO_PACKAGE_DIR.glob('*.py')
            if path.stem != '__init__'
        }
        assert expected <= imported, f'modules not imported: {sorted(expected - imported)}'


# ---------------------------------------------------------------------------
# Model-name resolution
# ---------------------------------------------------------------------------


class TestModelNames:
    @pytest.mark.parametrize(
        ('model_id', 'override', 'expected', 'used_default'),
        [
            ('provider/gpt-4o-mini', None, 'GPT-4o', False),
            ('provider/gpt-4', None, 'GPT-4', False),
            ('provider/GPT-4o', None, DEFAULT_MODEL_NAME, True),
            ('unknown', None, DEFAULT_MODEL_NAME, True),
            ('provider/gpt-4o', 'Custom Victim', 'Custom Victim', False),
        ],
    )
    def test_victim_model_name(self, model_id: str, override: Optional[str], expected: str, used_default: bool) -> None:
        names = {'gpt-4o': 'GPT-4o', 'gpt-4': 'GPT-4'}
        assert resolve_victim_model_name(model_id, override, names) == (expected, used_default)


# ---------------------------------------------------------------------------
# Upstream access errors, simulated
# ---------------------------------------------------------------------------

_REAL_VERSION = importlib.metadata.version
_REAL_IMPORT_MODULE = importlib.import_module


@pytest.fixture
def _clear_upstream_cache() -> Iterator[None]:
    load_upstream.cache_clear()
    yield
    load_upstream.cache_clear()


def _patch_version(monkeypatch: pytest.MonkeyPatch, installed: Optional[str]) -> None:
    """Make ``importlib.metadata.version('agentdojo')`` return ``installed``, or raise when it is None."""

    def fake_version(distribution_name: str) -> str:
        if distribution_name != UPSTREAM_DISTRIBUTION:
            return _REAL_VERSION(distribution_name)
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(distribution_name)
        return installed

    monkeypatch.setattr(importlib.metadata, 'version', fake_version)


def _patch_upstream_modules(
    monkeypatch: pytest.MonkeyPatch, get_suites: Callable[[str], Any], failing: Optional[Tuple[str, Exception]] = None
) -> List[str]:
    """Serve fake upstream modules from ``importlib.import_module``; return the requested names.

    Any other ``agentdojo`` import fails the test, so the real upstream package is never imported.
    """
    fakes = {
        upstream_module._SUITES_MODULE: types.SimpleNamespace(get_suites=get_suites),
        upstream_module._ATTACKS_MODULE: types.SimpleNamespace(ATTACKS={}),
        upstream_module._MODELS_MODULE: types.SimpleNamespace(MODEL_NAMES={}),
    }
    requested: List[str] = []

    def fake_import_module(name: str, package: Optional[str] = None) -> Any:
        if failing is not None and name == failing[0]:
            raise failing[1]
        if name in fakes:
            requested.append(name)
            return fakes[name]
        if name == 'agentdojo' or name.startswith('agentdojo.'):
            raise AssertionError(f'Unexpected upstream import: {name}')
        return _REAL_IMPORT_MODULE(name, package)

    monkeypatch.setattr(importlib, 'import_module', fake_import_module)
    return requested


@pytest.mark.usefixtures('_clear_upstream_cache')
class TestUpstreamAccessErrors:
    def test_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_version(monkeypatch, None)
        requested = _patch_upstream_modules(monkeypatch, lambda version: {})
        with pytest.raises(UpstreamUnavailableError) as exc_info:
            load_upstream()
        assert isinstance(exc_info.value, ImportError)
        assert AGENTDOJO_INSTALL_HINT in str(exc_info.value)
        assert 'PackageNotFoundError' in str(exc_info.value)
        assert requested == []

    @pytest.mark.parametrize(
        'module_name',
        [upstream_module._SUITES_MODULE, upstream_module._ATTACKS_MODULE, upstream_module._MODELS_MODULE],
    )
    def test_import_failure(self, monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
        exc = RuntimeError('broken upstream import side effect')
        _patch_version(monkeypatch, UPSTREAM_VERSION)
        _patch_upstream_modules(monkeypatch, lambda version: {}, failing=(module_name, exc))
        with pytest.raises(UpstreamUnavailableError) as exc_info:
            load_upstream()
        message = str(exc_info.value)
        assert AGENTDOJO_INSTALL_HINT in message and module_name in message and str(exc) in message
        assert exc_info.value.__cause__ is exc

    @pytest.mark.parametrize('installed', ['0.1.34', '0.1.35.post1'])
    def test_version_mismatch_is_checked_before_import(self, monkeypatch: pytest.MonkeyPatch, installed: str) -> None:
        _patch_version(monkeypatch, installed)
        requested = _patch_upstream_modules(monkeypatch, lambda version: {})
        with pytest.raises(UpstreamVersionError) as exc_info:
            load_upstream()
        assert UPSTREAM_VERSION in str(exc_info.value) and installed in str(exc_info.value)
        assert requested == []

    def test_missing_suites(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: List[str] = []

        def get_suites(version: str) -> Dict[str, Any]:
            calls.append(version)
            return {name: object() for name in SUITES if name not in ('workspace', 'banking')}

        _patch_version(monkeypatch, UPSTREAM_VERSION)
        _patch_upstream_modules(monkeypatch, get_suites)
        with pytest.raises(UpstreamDataError) as exc_info:
            load_upstream()
        assert BENCHMARK_VERSION in str(exc_info.value) and "['workspace', 'banking']" in str(exc_info.value)
        assert calls == [BENCHMARK_VERSION]

    def test_get_suites_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        exc = KeyError('suites unavailable')
        _patch_version(monkeypatch, UPSTREAM_VERSION)
        _patch_upstream_modules(monkeypatch, _raise(exc))
        with pytest.raises(UpstreamDataError) as exc_info:
            load_upstream()
        assert BENCHMARK_VERSION in str(exc_info.value) and 'KeyError' in str(exc_info.value)
        assert exc_info.value.__cause__ is exc

    def test_failures_are_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_version(monkeypatch, None)
        with pytest.raises(UpstreamUnavailableError):
            load_upstream()
        _patch_version(monkeypatch, '0.0.1')
        with pytest.raises(UpstreamVersionError):
            load_upstream()


# ---------------------------------------------------------------------------
# Adapter construction errors
# ---------------------------------------------------------------------------
#
# Every check runs in the adapter constructor before ``load_upstream()``; the loader is replaced with a failing stub,
# so these tests never reach the upstream package or a model.


@pytest.fixture
def _no_upstream_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agentdojo_adapter, 'load_upstream', _raise(AssertionError('upstream loaded too early')))


def _construct(subset_list: Optional[list] = None, agent_config: Any = None, **extra_params: Any) -> Any:
    # An empty ``extra_params`` dict would replace the declared params instead of merging into them.
    dataset_args: Dict[str, Any] = {'extra_params': extra_params} if extra_params else {}
    if subset_list is not None:
        dataset_args['subset_list'] = subset_list
    config = TaskConfig(
        model='gpt-4o-2024-05-13',
        datasets=['agentdojo'],
        dataset_args={'agentdojo': dataset_args},
        agent_config=agent_config,
    )
    return get_benchmark('agentdojo', config=config)


@pytest.mark.usefixtures('_no_upstream_access')
class TestConstructionErrors:
    @pytest.mark.parametrize('attack', ['dos', 'Direct', ''])
    def test_unsupported_attack(self, attack: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            _construct(attack=attack)
        message = str(exc_info.value)
        assert f'Value {attack} not in choices' in message and 'agentdojo' in message
        assert all(repr(name) in message for name in SUPPORTED_ATTACKS)

    def test_invalid_subset(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _construct(subset_list=['banking', 'workspace_plus'])
        assert "['workspace_plus']" in str(exc_info.value) and repr(list(SUITES)) in str(exc_info.value)

    @pytest.mark.parametrize('filter_name', ['user_task_ids', 'injection_task_ids'])
    def test_task_id_filter_needs_a_single_suite(self, filter_name: str) -> None:
        with pytest.raises(ValueError, match='single suite'):
            _construct(subset_list=['banking', 'slack'], **{filter_name: ['user_task_0']})

    @pytest.mark.parametrize(
        ('filter_name', 'value'),
        [('user_task_ids', 'user_task_0'), ('injection_task_ids', ['injection_task_0', 1])],
    )
    def test_task_id_filter_must_be_a_list_of_strings(self, filter_name: str, value: Any) -> None:
        with pytest.raises(ValueError) as exc_info:
            _construct(subset_list=['banking'], **{filter_name: value})
        assert f'extra_params.{filter_name}' in str(exc_info.value) and repr(value) in str(exc_info.value)

    @pytest.mark.parametrize('value', ['', 123])
    def test_invalid_victim_model_name(self, value: Any) -> None:
        with pytest.raises(ValueError, match='victim_model_name must be a non-empty string'):
            _construct(victim_model_name=value)

    @pytest.mark.parametrize(
        ('agent_config', 'fragment'),
        [
            (ExternalAgentConfig(framework='claude-code'), 'only supports the native AgentLoop'),
            (NativeAgentConfig(strategy='react'), "the only supported strategy is 'function_calling'"),
            (NativeAgentConfig(tools=['bash']), "NativeAgentConfig.tools must be empty, got ['bash']"),
        ],
    )
    def test_unsupported_agent_config(self, agent_config: Any, fragment: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            _construct(agent_config=agent_config)
        assert fragment in str(exc_info.value)

    def test_non_positive_max_steps_fails_at_construction(self) -> None:
        # The field validator only runs on creation, so a later assignment reaches the adapter.
        agent_config = NativeAgentConfig()
        agent_config.max_steps = 0
        with pytest.raises(ValueError, match='max_steps must be greater than 0, got 0'):
            _construct(agent_config=agent_config)


# ===========================================================================
# Upstream-dependent tests
# ===========================================================================


@requires_upstream
class TestUpstreamBehavior:
    """Upstream behavior the benchmark relies on that the equivalence tests below do not already cover."""

    def test_empty_function_name_deviation(self) -> None:
        from agentdojo.agent_pipeline.tool_execution import EMPTY_FUNCTION_NAME, ToolsExecutor
        from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
        from agentdojo.types import ChatAssistantMessage

        assert EMPTY_FUNCTION_NAME == UPSTREAM_EMPTY_FUNCTION_NAME
        suite = load_upstream().suites[SUITE]
        calls = [FunctionCall(function=name, args={}, id=name) for name in ('', EMPTY_FUNCTION_NAME)]
        assistant = ChatAssistantMessage(role='assistant', content=None, tool_calls=calls)
        environment = suite.load_and_inject_default_environment({})
        messages = ToolsExecutor().query('', FunctionsRuntime(suite.tools), environment, [assistant], {})[3]
        # Upstream answers a literal '' with the invalid-tool text; the binding deliberately treats it as empty.
        assert [message['error'] for message in messages[1:]] == [
            INVALID_TOOL_ERROR.format(name=''),
            EMPTY_FUNCTION_NAME_ERROR,
        ]
        binding = AgentDojoToolBinding(suite, environment)
        for name in ('', EMPTY_FUNCTION_NAME):
            call = ToolCall(id=name, function=ToolFunction(name=name, arguments={}))
            assert binding.handle_invalid_function(call) == EMPTY_FUNCTION_NAME_ERROR

    def test_model_names_match_upstream(self, upstream: UpstreamContext) -> None:
        from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME as UPSTREAM_DEFAULT_MODEL_NAME
        from agentdojo.attacks.important_instructions_attacks import get_wrong_model_name_from_pipeline

        from evalscope.benchmarks.agentdojo.injections import _pipeline_stub_class

        assert UPSTREAM_DEFAULT_MODEL_NAME == DEFAULT_MODEL_NAME
        for key in upstream.model_names:
            model_id = f'provider/{key}-suffix'
            victim, used_default = resolve_victim_model_name(model_id, None, upstream.model_names)
            assert not used_default
            effective = resolve_effective_model_name(
                'important_instructions_wrong_model_name', victim, upstream.model_names
            )
            assert effective == get_wrong_model_name_from_pipeline(_pipeline_stub_class()(model_id))

    def test_suite_tools_convert_to_tool_infos(self, upstream: UpstreamContext) -> None:
        infos_by_suite: Dict[str, Dict[str, ToolInfo]] = {}
        for suite_name, suite in upstream.suites.items():
            handlers, infos = AgentDojoToolBinding(suite, suite.load_and_inject_default_environment({})).build()
            assert [info.name for info in infos] == [fn.name for fn in suite.tools]
            assert set(handlers) == _tool_names(suite_name)
            infos_by_suite[suite_name] = {info.name: info for info in infos}
        # The upstream schema references ``#/$defs/SharingPermission``; the bridge inlines it with its description.
        permission = infos_by_suite['workspace']['share_file'].parameters.properties['permission']
        assert (permission.type, permission.enum, permission.description) == (
            'string',
            ['r', 'rw'],
            'The permission level to grant the user.',
        )


# ---------------------------------------------------------------------------
# Injection planning
# ---------------------------------------------------------------------------

_ATTACKS = [name for name in SUPPORTED_ATTACKS if name != BENIGN_ATTACK]


def _reference_injections(
    attack_name: str, sample: Tuple[str, str, str], pipeline_name: str, model_name: Optional[str]
) -> Dict[str, str]:
    """Injection map of the upstream attack class; ``model_name=None`` keeps upstream's own name resolution."""
    from evalscope.benchmarks.agentdojo.injections import _pipeline_stub_class

    upstream = load_upstream()
    suite_name, user_task_id, injection_task_id = sample
    suite = upstream.suites[suite_name]
    attack = upstream.attacks[attack_name](suite, _pipeline_stub_class()(pipeline_name))
    if model_name is not None:
        attack.model_name = model_name
    return dict(attack.attack(suite.user_tasks[user_task_id], suite.injection_tasks[injection_task_id]))


@requires_upstream
class TestInjections:
    @pytest.mark.parametrize('attack_name', _ATTACKS)
    def test_every_attack_matches_upstream(self, attack_name: str) -> None:
        upstream = load_upstream()
        model_key = next(iter(upstream.model_names))
        victim, _ = resolve_victim_model_name(model_key, None, upstream.model_names)
        planner = InjectionPlanner(upstream, attack_name, victim)
        for sample in [(SUITE, USER_TASK_ID, INJECTION_TASK_ID), ('workspace', 'user_task_13', 'injection_task_3')]:
            actual = planner.injections_for(*sample)
            assert actual and all(
                load_upstream().suites[sample[0]].injection_tasks[sample[2]].GOAL in t for t in actual.values()
            )
            # A fully upstream construction from the model id agrees with the planner.
            assert actual == _reference_injections(attack_name, sample, model_key, None)

    def test_planning_is_deterministic(self) -> None:
        upstream = load_upstream()
        samples = [
            (suite_name, user_task_id, injection_task_id)
            for suite_name in ('banking', 'slack')
            for user_task_id in ('user_task_0', 'user_task_1')
            for injection_task_id in sorted(upstream.suites[suite_name].injection_tasks)[:3]
        ]
        planner = InjectionPlanner(upstream, DEFAULT_ATTACK, VICTIM_MODEL_NAME)
        first = [planner.injections_for(*sample) for sample in samples]
        assert first == [planner.injections_for(*sample) for sample in samples]
        assert first == [_default_planner().injections_for(*sample) for sample in samples]

    def test_benign_planner_writes_no_injections(self) -> None:
        planner = InjectionPlanner(load_upstream(), BENIGN_ATTACK, VICTIM_MODEL_NAME)
        assert planner.injections_for(SUITE, USER_TASK_ID, None) == {}

    def test_user_task_without_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        context = load_upstream()
        monkeypatch.setattr(type(context.suites[SUITE].user_tasks[USER_TASK_ID]), 'ground_truth', lambda self, env: [])
        planner = InjectionPlanner(context, DEFAULT_ATTACK, VICTIM_MODEL_NAME)
        with pytest.raises(UpstreamDataError, match="suite 'banking'.*user task 'user_task_0'") as exc_info:
            planner.injections_for(SUITE, USER_TASK_ID, INJECTION_TASK_ID)
        assert 'is not injectable' in str(exc_info.value.__cause__)

    def test_tool_knowledge_needs_placeholder_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentdojo.functions_runtime import FunctionCall

        context = load_upstream()
        injection_task = context.suites[SUITE].injection_tasks[INJECTION_TASK_ID]
        ground_truth = [FunctionCall(function='get_balance', args={}, placeholder_args=None)]
        monkeypatch.setattr(type(injection_task), 'ground_truth', lambda self, env: ground_truth)
        planner = InjectionPlanner(context, 'tool_knowledge', VICTIM_MODEL_NAME)
        with pytest.raises(UpstreamDataError, match="suite 'banking'.*injection task 'injection_task_0'") as exc_info:
            planner.injections_for(SUITE, USER_TASK_ID, INJECTION_TASK_ID)
        assert 'placeholder' in str(exc_info.value.__cause__)


# ---------------------------------------------------------------------------
# Tool binding and environments
# ---------------------------------------------------------------------------


def _fresh_environment(suite_name: str, user_task_id: str, injections: Optional[Mapping[str, str]] = None) -> Any:
    suite = load_upstream().suites[suite_name]
    environment = suite.load_and_inject_default_environment(dict(injections or {}))
    return suite.user_tasks[user_task_id].init_environment(environment)


async def _dispatch(
    binding: AgentDojoToolBinding, calls: Sequence[ScriptedToolCall], prefix: str = 'call'
) -> List[str]:
    """Dispatch like the agent loop: suite tools go to the handlers, other names to ``handle_invalid_function``."""
    handlers, _ = binding.build()
    observations = []
    for index, call in enumerate(calls):
        tool_call = ToolCall(
            id=f'{prefix}_{index}',
            function=ToolFunction(name=call.function, arguments=copy.deepcopy(call.arguments)),
            parse_error=call.parse_error,
        )
        if call.function in handlers:
            observations.append(await handlers[call.function](tool_call, None))
        elif call.parse_error is not None:
            observations.append(ARGUMENT_PARSE_ERROR.format(error=call.parse_error))
        else:
            observations.append(binding.handle_invalid_function(tool_call))
    return observations


@st.composite
def _binding_scenarios(draw: st.DrawFn) -> Tuple[str, str, List[ScriptedToolCall]]:
    suite_name = draw(st.sampled_from(SUITES))
    user_task_id = draw(st.sampled_from(sorted(load_upstream().suites[suite_name].user_tasks)))
    return suite_name, user_task_id, draw(st.lists(_mutated_calls(suite_name), min_size=1, max_size=8))


@requires_upstream
@pytest.mark.usefixtures('frozen_tool_clock')
class TestToolBindingAndEnvironment:
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(scenario=_binding_scenarios())
    def test_binding_matches_upstream_tools_executor(self, scenario: Tuple[str, str, List[ScriptedToolCall]]) -> None:
        from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
        from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
        from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages
        from agentdojo.types import ChatAssistantMessage

        suite_name, user_task_id, calls = scenario
        suite = load_upstream().suites[suite_name]
        binding = AgentDojoToolBinding(suite, _fresh_environment(suite_name, user_task_id))
        observations = asyncio.run(_dispatch(binding, calls))
        assert all(EVALSCOPE_UNKNOWN_TOOL_TEXT not in text for text in observations)

        # Parse-error calls never reach the runtime and are not traced, so they have no upstream counterpart.
        compared = [(index, call) for index, call in enumerate(calls) if call.parse_error is None]
        for index, call in enumerate(calls):
            if call.parse_error is not None:
                assert observations[index] == ARGUMENT_PARSE_ERROR.format(error=call.parse_error)
        upstream_calls = [
            FunctionCall(
                function=UPSTREAM_EMPTY_FUNCTION_NAME if call.function == '' else call.function,
                args=copy.deepcopy(call.arguments),
                id=f'call_{index}',
            )
            for index, call in compared
        ]
        upstream_env = _fresh_environment(suite_name, user_task_id)
        assistant = ChatAssistantMessage(role='assistant', content=None, tool_calls=upstream_calls)
        try:
            _, _, upstream_env, messages, _ = ToolsExecutor().query(
                '', FunctionsRuntime(suite.tools), upstream_env, [assistant], {}
            )
        except TypeError:
            # Upstream aborts when tool_result_to_str rejects a result; the binding reports it instead.
            assert any(item.formatter_error for item in binding.trace)
            return

        # Upstream LLM clients send ``error or content`` back to the model.
        expected = [m['error'] or ''.join(b['content'] or '' for b in m['content']) for m in messages[1:]]
        assert [observations[index] for index, _ in compared] == expected
        assert binding.environment == upstream_env
        upstream_trace = functions_stack_trace_from_messages(messages)
        assert len(binding.trace) == len(upstream_trace) == len(compared)
        for item, upstream_item, (index, call) in zip(binding.trace, upstream_trace, compared):
            assert item.function == call.function
            assert item.args == to_jsonable(dict(upstream_item.args))
            assert (item.observation, item.call_id) == (observations[index], f'call_{index}')
            assert item.valid_function == (call.function in binding.tool_names)

    @pytest.mark.parametrize(
        ('suite_name', 'user_task_id'),
        [
            ('workspace', 'user_task_13'),
            ('travel', 'user_task_0'),
            ('banking', 'user_task_0'),
            ('slack', 'user_task_1'),
        ],
    )
    def test_environments_are_isolated(self, suite_name: str, user_task_id: str) -> None:
        upstream = load_upstream()
        suite = upstream.suites[suite_name]
        meta = _meta(suite_name, user_task_id)
        active = AgentDojoToolBinding(suite, create_environment(upstream, meta))
        idle = AgentDojoToolBinding(suite, create_environment(upstream, meta))
        before = copy.deepcopy(idle.environment)
        calls = [
            ScriptedToolCall(function=call.function, arguments=dict(call.args))
            for call in suite.user_tasks[user_task_id].ground_truth(active.environment)
        ]

        async def run_both() -> None:
            await asyncio.gather(_dispatch(active, calls), _dispatch(idle, []))

        asyncio.run(run_both())

        assert active.environment != before, 'the control task must actually mutate its environment'
        assert idle.environment == before and idle.trace == []
        assert create_environment(upstream, meta) == before

    def test_invalid_injection_vector_fails_environment_creation(self) -> None:
        meta = _meta(SUITE, USER_TASK_ID, INJECTION_TASK_ID, {'not_an_injection_vector': 'x'})
        with pytest.raises(EnvironmentCreationError, match="suite='banking', user_task_id='user_task_0'"):
            create_environment(load_upstream(), meta)


def _persisted(suite: Any, environment: Any) -> Any:
    """Persist an environment like the run record does and restore it like scoring does."""
    data = json.loads(json.dumps(environment.model_dump(mode='json', round_trip=True)))
    return restore_environment(suite.environment_type, data)


@requires_upstream
class TestGroundTruth:
    """Every task's ground truth, run through the binding, passes its own upstream check (as ``TaskSuite.check``).

    The final environment survives the JSON round trip of the run record unchanged, so the restored environment
    scores exactly like the live one.
    """

    @pytest.mark.parametrize('suite_name', SUITES)
    def test_ground_truth_solves_its_task(self, upstream: UpstreamContext, suite_name: str) -> None:
        suite = upstream.suites[suite_name]

        def run(meta: AgentDojoSampleMeta, task: Any, pre_environment: Any, environment: Any) -> Tuple[Any, ...]:
            binding = AgentDojoToolBinding(suite, environment)
            for call in task.ground_truth(environment):
                binding.execute(call.function, dict(call.args), call.id)
            restored = _persisted(suite, environment)
            assert restored == environment and restored is not environment
            output, trace = task.GROUND_TRUTH_OUTPUT, binding.trace
            result = check_results(upstream, meta, output, pre_environment, environment, trace)
            assert check_results(upstream, meta, output, pre_environment, restored, trace) == result
            return result

        failing_user = []
        for user_task_id, user_task in suite.user_tasks.items():
            meta = _meta(suite_name, user_task_id)
            pre_environment = create_environment(upstream, meta)
            if run(meta, user_task, pre_environment, create_environment(upstream, meta)) != (True, None):
                failing_user.append(user_task_id)

        failing_injection = []
        # Upstream pairs every injection task with the first user task on the default environment.
        first_user_task_id = next(iter(suite.user_tasks))
        for injection_task_id, injection_task in suite.injection_tasks.items():
            meta = _meta(suite_name, first_user_task_id, injection_task_id)
            environment = suite.load_and_inject_default_environment({})
            if not run(meta, injection_task, copy.deepcopy(environment), environment)[1]:
                failing_injection.append(injection_task_id)

        assert failing_user == []
        # Upstream workspace injection tasks 6-13 ship an empty ground truth; upstream's own check fails them too.
        expected = {f'injection_task_{index}' for index in range(6, 14)} if suite_name == 'workspace' else set()
        assert set(failing_injection) == expected

    def test_model_validate_would_undo_the_run(self, upstream: UpstreamContext) -> None:
        # Upstream validators rebuild ``emails`` from ``initial_emails``, so a plain model_validate drops sent mail.
        suite = upstream.suites['workspace']
        environment = suite.load_and_inject_default_environment({})
        AgentDojoToolBinding(suite, environment).execute(
            'send_email', {'recipients': ['a@x.com'], 'subject': 'Hi', 'body': 'Hello'}, 'call_0'
        )
        data = environment.model_dump(mode='json', round_trip=True)
        assert suite.environment_type.model_validate(data) != environment
        assert _persisted(suite, environment) == environment

    # ``share_file`` writes its ``email`` argument into an ``EmailStr`` key without validation; validating it on
    # restore would lowercase the domain, strip the display name or reject the value.
    @pytest.mark.parametrize(
        'email, utility',
        [
            ('john.doe@gmail.com', True),
            ('john.doe@GMAIL.com', False),
            ('John <john.doe@gmail.com>', False),
            ('john', False),
        ],
    )
    def test_unvalidated_shared_email_is_restored_exactly(
        self, upstream: UpstreamContext, email: str, utility: bool
    ) -> None:
        suite = upstream.suites['workspace']
        task = suite.user_tasks['user_task_32']
        meta = _meta('workspace', 'user_task_32')
        pre_environment, environment = create_environment(upstream, meta), create_environment(upstream, meta)
        binding = AgentDojoToolBinding(suite, environment)
        for call in task.ground_truth(environment):
            args = {**call.args, 'email': email} if call.function == 'share_file' else dict(call.args)
            binding.execute(call.function, args, call.id)
        restored = _persisted(suite, environment)
        assert restored == environment
        assert any(list(file.shared_with) == [email] for file in restored.cloud_drive.files.values())
        output, trace = task.GROUND_TRUTH_OUTPUT, binding.trace
        assert check_results(upstream, meta, output, pre_environment, restored, trace) == (utility, None)
        assert check_results(upstream, meta, output, pre_environment, environment, trace) == (utility, None)

    def test_unvalidated_participants_are_restored_exactly(self, upstream: UpstreamContext) -> None:
        suite = upstream.suites['workspace']
        meta = _meta('workspace', 'user_task_8')
        pre_environment, environment = create_environment(upstream, meta), create_environment(upstream, meta)
        binding = AgentDojoToolBinding(suite, environment)
        participants = ['john', 'John@EXAMPLE.com']
        binding.execute('add_calendar_event_participants', {'event_id': '24', 'participants': participants}, 'c0')
        restored = _persisted(suite, environment)
        assert restored == environment
        assert restored.calendar.events['24'].participants[-2:] == participants
        results = [
            check_results(upstream, meta, '', pre_environment, post, binding.trace) for post in (environment, restored)
        ]
        assert results == [(False, None), (False, None)]

    def test_invalid_environment_data_is_rejected(self, upstream: UpstreamContext) -> None:
        suite = upstream.suites[SUITE]
        data = suite.load_and_inject_default_environment({}).model_dump(mode='json', round_trip=True)
        for broken in ([], {**data, 'bank_account': None}, {k: v for k, v in data.items() if k != 'filesystem'}):
            with pytest.raises(ValueError):
                restore_environment(suite.environment_type, broken)


# ---------------------------------------------------------------------------
# Loading through the adapter
# ---------------------------------------------------------------------------


def _number(task_id: str) -> int:
    return int(task_id.rsplit('_', 1)[1])


def _expected_row_ids(
    suite_name: str,
    attack: str,
    user_task_ids: Optional[List[str]] = None,
    injection_task_ids: Optional[List[str]] = None,
) -> List[str]:
    """Rows in the documented order (user task number, benign row first, injection task number), then filtered."""
    suite = load_upstream().suites[suite_name]
    pairs = [(u, None) for u in suite.user_tasks] + [(u, i) for u in suite.user_tasks for i in suite.injection_tasks]
    pairs.sort(key=lambda pair: (_number(pair[0]), pair[1] is not None, -1 if pair[1] is None else _number(pair[1])))
    benign = attack == BENIGN_ATTACK
    return [
        row_id(suite_name, u, i)
        for u, i in pairs
        if (i is None) == benign
        and (not user_task_ids or u in user_task_ids)
        and (benign or not injection_task_ids or i in injection_task_ids)
    ]


@requires_upstream
class TestLoading:
    def test_docs_statistics_path_generates_every_attack_row(self, upstream: UpstreamContext) -> None:
        from evalscope.utils.doc_utils.benchmark_stats import compute_benchmark_statistics

        # Built without a TaskConfig, like the docs generator; the runtime is loaded by load_subset().
        adapter = get_benchmark('agentdojo')
        statistics = compute_benchmark_statistics(adapter)

        assert statistics.total_samples == 949
        counts = {s: len(upstream.suites[s].user_tasks) * len(upstream.suites[s].injection_tasks) for s in SUITES}
        assert {subset.name: subset.sample_count for subset in statistics.subset_stats} == counts
        for suite_name in SUITES:
            samples = list(adapter.test_dataset[suite_name])
            assert [s.metadata[SAMPLE_META_KEY]['row_id'] for s in samples] == _expected_row_ids(
                suite_name, DEFAULT_ATTACK
            )
            assert all(s.metadata[SAMPLE_META_KEY]['injections'] for s in samples)
            prompts = {s.metadata[SAMPLE_META_KEY]['user_task_id']: s.input[-1].text for s in samples}
            assert prompts == {u: task.PROMPT for u, task in upstream.suites[suite_name].user_tasks.items()}
            assert all(s.input[0].text == upstream.system_message for s in samples)

    @pytest.mark.parametrize(
        ('attack', 'limit', 'repeats'),
        [(DEFAULT_ATTACK, 2, 2), (DEFAULT_ATTACK, 0.5, 1), (BENIGN_ATTACK, 1, 2), (BENIGN_ATTACK, 0.5, 1)],
    )
    def test_filters_apply_before_limit_and_repeats(
        self, work_dir: str, attack: str, limit: Union[int, float], repeats: int
    ) -> None:
        config = make_task_config(
            work_dir,
            attack=attack,
            user_task_ids=['user_task_2', 'user_task_0'],
            injection_task_ids=['injection_task_1', 'injection_task_0'],
            limit=limit,
            repeats=repeats,
        )
        samples = list(get_benchmark('agentdojo', config=config).load_dataset()[SUITE])
        expected = (
            [row_id(SUITE, 'user_task_0', None)]
            if attack == BENIGN_ATTACK
            else [row_id(SUITE, 'user_task_0', f'injection_task_{i}') for i in (0, 1)]
        )
        assert [s.metadata[SAMPLE_META_KEY]['row_id'] for s in samples] == [r for r in expected for _ in range(repeats)]
        assert [s.group_id for s in samples] == [i for i in range(len(expected)) for _ in range(repeats)]
        if attack == BENIGN_ATTACK:
            assert all(s.metadata[SAMPLE_META_KEY]['injections'] == {} for s in samples)

    @pytest.mark.parametrize('filter_name', ['user_task_ids', 'injection_task_ids'])
    def test_unknown_task_id_fails_before_inference(self, tmp_path: Path, filter_name: str) -> None:
        reset_model_calls()
        config = make_task_config(tmp_path, attack=DEFAULT_ATTACK, **{filter_name: ['task_99']})
        with pytest.raises(ValueError, match=f"suite 'banking' has no task\\(s\\) matching extra_params.{filter_name}"):
            run_task(config)
        assert model_calls() == []


# ---------------------------------------------------------------------------
# Step budget and termination mapping
# ---------------------------------------------------------------------------


def _scripted_model() -> Any:
    return get_model(SCRIPTED_MODEL_NAME, SCRIPTED_EVAL_TYPE, memoize=False)


@requires_upstream
@pytest.mark.usefixtures('loop_runner_cleanup')
class TestStepBudget:
    @pytest.mark.parametrize('attack', [BENIGN_ATTACK, DEFAULT_ATTACK])
    def test_budget_counts_generations_and_executes_every_call(self, work_dir: str, attack: str) -> None:
        config = make_task_config(work_dir, attack=attack, agent_config=NativeAgentConfig(max_steps=2))
        adapter = get_benchmark('agentdojo', config=config)
        sample = _make_sample(adapter, SUITE, USER_TASK_ID, None if attack == BENIGN_ATTACK else INJECTION_TASK_ID)
        turn = tool_calls_turn(('get_balance', {}), ('get_most_recent_transactions', {'n': 1}), text='Working.')
        with installed_scripts({(SUITE, USER_TASK_ID): [turn, turn, text_turn(text='Unreachable')]}):
            state = adapter.run_inference(_scripted_model(), sample, work_dir)
            score = adapter.calculate_metrics(state).score

        assert model_calls() == [(SUITE, USER_TASK_ID, 0), (SUITE, USER_TASK_ID, 1)]
        record = state.metadata[RUN_RECORD_META_KEY]
        assert record['executed_steps'] == 2 and record['termination_reason'] == 'max_steps'
        assert len(record['tool_call_trace']) == 4
        assert record['final_output'] == state.output.completion == 'Working.'
        assert score.status == ScoreStatus.SUCCESS
        assert score.value == ({'utility': 0.0} if attack == BENIGN_ATTACK else {'utility': 0.0, 'targeted_asr': 0.0})


# ---------------------------------------------------------------------------
# Scoring from the prediction cache
# ---------------------------------------------------------------------------
#
# The clock is real: upstream email and cloud-drive tools stamp ``datetime.now()`` into the environment, and scoring
# must use the persisted final environment rather than anything it recomputes.

_SCRIPT_MAX_STEPS = 32


def _user_then_injection_turns(
    suite_name: str, user_task_id: str, injection_task_id: str, injections: Mapping[str, str]
) -> List[ModelOutput]:
    turns = ground_truth_turns(suite_name, user_task_id, injections)
    suite = load_upstream().suites[suite_name]
    task = suite.get_injection_task_by_id(injection_task_id)
    environment = _fresh_environment(suite_name, user_task_id, injections)
    calls = [tool_calls_turn((call.function, dict(call.args))) for call in task.ground_truth(environment)]
    return [*turns[:-1], *calls, text_turn(text=task.GROUND_TRUTH_OUTPUT)]


@requires_upstream
@pytest.mark.usefixtures('loop_runner_cleanup')
class TestScoringFromCache:
    @pytest.mark.parametrize('attack', [BENIGN_ATTACK, DEFAULT_ATTACK])
    @pytest.mark.parametrize(
        ('suite_name', 'user_task_id', 'injection_task_id'),
        [
            ('workspace', 'user_task_13', 'injection_task_0'),
            ('travel', 'user_task_3', 'injection_task_0'),
            ('banking', 'user_task_0', 'injection_task_0'),
            ('slack', 'user_task_1', 'injection_task_1'),
        ],
    )
    def test_cached_sample_scores_like_the_live_run(
        self, work_dir: str, attack: str, suite_name: str, user_task_id: str, injection_task_id: str
    ) -> None:
        benign = attack == BENIGN_ATTACK
        config = make_task_config(
            work_dir, subset=suite_name, attack=attack, agent_config=NativeAgentConfig(max_steps=_SCRIPT_MAX_STEPS)
        )
        inference_adapter = get_benchmark('agentdojo', config=config)
        review_adapter = get_benchmark('agentdojo', config=config)
        sample = _make_sample(inference_adapter, suite_name, user_task_id, None if benign else injection_task_id)
        meta = AgentDojoSampleMeta.model_validate(sample.metadata[SAMPLE_META_KEY])
        environments: List[Any] = []

        def capture_environment(upstream: UpstreamContext, sample_meta: AgentDojoSampleMeta) -> Any:
            environments.append(create_environment(upstream, sample_meta))
            return environments[-1]

        turns = (
            ground_truth_turns(suite_name, user_task_id)
            if benign
            else _user_then_injection_turns(suite_name, user_task_id, injection_task_id, meta.injections)
        )
        with pytest.MonkeyPatch.context() as patch, installed_scripts({(suite_name, user_task_id): turns}):
            patch.setattr(agentdojo_adapter, 'create_environment', capture_environment)
            state = inference_adapter.run_inference(_scripted_model(), sample, work_dir)
            calls_after_inference = len(model_calls())
            patch.setattr(AgentDojoToolBinding, 'execute', _raise(AssertionError('tool executed while scoring')))
            score = inference_adapter.calculate_metrics(state).score
            restored = _round_trip(state)
            review_score = review_adapter.calculate_metrics(restored).score
            assert len(model_calls()) == calls_after_inference

        live_environment, *pre_environments = environments
        payload = ScoringPayload.from_metadata(restored.metadata, load_upstream())
        assert payload.post_environment == live_environment
        assert len(pre_environments) == 2 and all(
            pre == create_environment(load_upstream(), meta) for pre in pre_environments
        )
        assert score.status == review_score.status == ScoreStatus.SUCCESS, review_score.explanation
        assert (review_score.value, review_score.metadata) == (score.value, score.metadata)
        if benign:
            assert score.value == {'utility': 1.0}
        else:
            assert score.value['targeted_asr'] == 1.0


# Upstream ``Calendar.create_event`` stores ``list(set(participants))``, so the participant order of a new event
# depends on the string hash seed and differs between processes. A workspace attack sample creates such events,
# reads them back and is then re-scored with ``rerun_review`` in fresh interpreters with other hash seeds.
_CALENDAR_USER_TASK_ID = 'user_task_9'
_CALENDAR_INJECTION_TASK_ID = 'injection_task_2'
_EXTRA_EVENT = {
    'title': 'Cross-process sync',
    'start_time': '2024-05-20 10:00',
    'end_time': '2024-05-20 11:00',
    'participants': ['a@x.com', 'b@x.com', 'c@x.com', 'd@x.com', 'e@x.com'],
}
_CROSS_PROCESS_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    from pathlib import Path

    from tests.benchmark import test_agentdojo as t

    mode, work_dir = sys.argv[1], Path(sys.argv[2])
    config = dict(
        subset='workspace',
        attack=t.DEFAULT_ATTACK,
        user_task_ids=[t._CALENDAR_USER_TASK_ID],
        injection_task_ids=[t._CALENDAR_INJECTION_TASK_ID],
        victim_model_name=t.VICTIM_MODEL_NAME,
        agent_config=t.NativeAgentConfig(max_steps=t._SCRIPT_MAX_STEPS),
    )
    if mode == 'record':
        report = t._run({('workspace', t._CALENDAR_USER_TASK_ID): t._calendar_script()}, work_dir, **config)
    else:
        report = t._run({}, work_dir, use_cache=str(work_dir), rerun_review=True, **config)
        assert t.model_calls() == []
    (row,) = t._reviews(work_dir, 'workspace').values()
    print(json.dumps({
        'score': {key: t._score(row)[key] for key in ('status', 'value', 'metadata')},
        'report': {m.identity.name: [m.score, m.num] for m in report.metrics},
        'set_order': list(set(t._EXTRA_EVENT['participants'])),
    }))
    """
)
_RECORD_HASH_SEED = '1'
_RESCORE_HASH_SEEDS = ('2', '3', '4', '5')


def _calendar_script() -> List[ModelOutput]:
    """User then injection ground truth, plus an event with five participants that is read back."""
    upstream = load_upstream()
    planner = InjectionPlanner(upstream, DEFAULT_ATTACK, VICTIM_MODEL_NAME)
    injections = planner.injections_for('workspace', _CALENDAR_USER_TASK_ID, _CALENDAR_INJECTION_TASK_ID)
    turns = _user_then_injection_turns('workspace', _CALENDAR_USER_TASK_ID, _CALENDAR_INJECTION_TASK_ID, injections)
    read_back = [
        tool_calls_turn(('create_calendar_event', _EXTRA_EVENT)),
        tool_calls_turn(('search_calendar_events', {'query': _EXTRA_EVENT['title']})),
        tool_calls_turn(('get_day_calendar_events', {'day': '2024-05-20'})),
    ]
    return [*turns[:-1], *read_back, turns[-1]]


def _cross_process(mode: str, work_dir: Path, hash_seed: str) -> 'subprocess.Popen[str]':
    repo_root = str(EVALSCOPE_ROOT.parent)
    python_path = os.pathsep.join(filter(None, [repo_root, os.environ.get('PYTHONPATH')]))
    return subprocess.Popen(
        [sys.executable, '-c', _CROSS_PROCESS_SCRIPT, mode, str(work_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=repo_root,
        env={**os.environ, 'PYTHONHASHSEED': hash_seed, 'PYTHONPATH': python_path},
    )


def _result(process: 'subprocess.Popen[str]') -> Dict[str, Any]:
    stdout, stderr = process.communicate(timeout=600)
    assert process.returncode == 0, f'stdout:\n{stdout}\nstderr:\n{stderr}'
    return json.loads(stdout.strip().splitlines()[-1])


@requires_upstream
class TestCrossProcessScoring:
    def test_rerun_review_under_other_hash_seeds_reproduces_the_scores(self, tmp_path: Path) -> None:
        recorded = _result(_cross_process('record', tmp_path / 'record', _RECORD_HASH_SEED))
        assert recorded['score']['status'] == ScoreStatus.SUCCESS
        # The extra events break the strict user-task diff; the attack succeeds only if the injected event persists.
        assert recorded['score']['value'] == {'utility': 0.0, 'targeted_asr': 1.0}
        predictions = tmp_path / 'record' / 'predictions' / SCRIPTED_MODEL_NAME / 'agentdojo_workspace.jsonl'
        (row,) = jsonl_to_list(str(predictions))
        trace = row['metadata'][RUN_RECORD_META_KEY]['tool_call_trace']
        assert sum(item['function'] == 'create_calendar_event' for item in trace) == 3
        assert _EXTRA_EVENT['title'] in trace[-1]['observation']

        # Each process scores its own copy of the outputs, so concurrent reviews never share a file.
        copies = [shutil.copytree(tmp_path / 'record', tmp_path / f'seed_{seed}') for seed in _RESCORE_HASH_SEEDS]
        processes = [_cross_process('rescore', Path(path), seed) for path, seed in zip(copies, _RESCORE_HASH_SEEDS)]
        rescored = [_result(process) for process in processes]

        assert all(result['score'] == recorded['score'] for result in rescored)
        assert all(result['report'] == recorded['report'] for result in rescored)
        # At least one seed orders the participants differently, which the persisted environment makes irrelevant.
        assert any(result['set_order'] != recorded['set_order'] for result in rescored)


# ---------------------------------------------------------------------------
# Incomplete or tampered payloads
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _captured_errors(level: int = logging.ERROR) -> Iterator[List[str]]:
    """Collect records of ``level`` and above from the ``evalscope`` logger, which does not propagate to root."""
    messages: List[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = Collect(level=level)
    logging.getLogger('evalscope').addHandler(handler)
    try:
        yield messages
    finally:
        logging.getLogger('evalscope').removeHandler(handler)


@pytest.fixture(scope='class')
def valid_state(work_dir: str, loop_runner_cleanup: None) -> Tuple[Any, TaskState]:
    """A scored attack sample used as the positive control for invalid-cache tests."""
    adapter = get_benchmark('agentdojo', config=make_task_config(work_dir, attack=DEFAULT_ATTACK))
    sample = _make_sample(adapter, SUITE, USER_TASK_ID, INJECTION_TASK_ID)
    with installed_scripts({(SUITE, USER_TASK_ID): ground_truth_turns(SUITE, USER_TASK_ID)}):
        state = adapter.run_inference(_scripted_model(), sample, work_dir)
    assert adapter.calculate_metrics(copy.deepcopy(state)).score.status == ScoreStatus.SUCCESS
    return adapter, state


@requires_upstream
class TestPayloadExclusion:
    @pytest.mark.parametrize(
        ('path', 'value'),
        [
            ((SAMPLE_META_KEY,), None),
            ((RUN_RECORD_META_KEY,), None),
            ((SAMPLE_META_KEY, 'user_task_id'), None),
            ((RUN_RECORD_META_KEY, 'final_output'), None),
            ((SAMPLE_META_KEY, 'upstream_version'), '0.0'),
            ((SAMPLE_META_KEY, 'benchmark_version'), 'v0'),
            ((SAMPLE_META_KEY, 'suite'), 'unknown'),
            ((RUN_RECORD_META_KEY, 'post_environment'), []),
            ((RUN_RECORD_META_KEY, 'post_environment', 'bank_account'), None),
            ((RUN_RECORD_META_KEY, 'post_environment', 'bank_account', 'balance'), 'not a number'),
        ],
    )
    def test_incomplete_or_tampered_payload_is_excluded(
        self, valid_state: Tuple[Any, TaskState], path: Tuple[str, ...], value: Any
    ) -> None:
        adapter, base = valid_state
        state = copy.deepcopy(base)
        part = state.metadata
        for key in path[:-1]:
            part = part[key]
        if value is None:
            del part[path[-1]]
        else:
            part[path[-1]] = value
        persisted = copy.deepcopy(state.metadata)
        bad_field = '.'.join(path[:2] if path[:2] == (RUN_RECORD_META_KEY, 'post_environment') else path)
        reset_model_calls()

        with _captured_errors() as errors:
            score = adapter.calculate_metrics(state).score

        assert (score.status, score.value) == (ScoreStatus.EXCLUDED, {})
        assert score.metadata['excluded_reason'] == f'invalid_scoring_payload: {[bad_field]}'
        assert any(bad_field in message for message in errors)
        assert state.metadata == persisted and model_calls() == []

    @pytest.mark.parametrize(
        ('config_kwargs', 'drifted'),
        [
            ({'attack': 'direct'}, ['attack']),
            (
                {'attack': DEFAULT_ATTACK, 'victim_model_name': 'Other Victim'},
                ['victim_model_name', 'effective_model_name'],
            ),
        ],
    )
    def test_cached_sample_built_with_another_config_is_excluded(
        self, valid_state: Tuple[Any, TaskState], work_dir: str, config_kwargs: Dict[str, Any], drifted: List[str]
    ) -> None:
        _, state = valid_state
        adapter = get_benchmark('agentdojo', config=make_task_config(work_dir, **config_kwargs))

        score = adapter.calculate_metrics(copy.deepcopy(state)).score

        assert (score.status, score.value) == (ScoreStatus.EXCLUDED, {})
        assert score.metadata['excluded_reason'] == (
            f'invalid_scoring_payload: {[f"{SAMPLE_META_KEY}.{name}" for name in drifted]}'
        )


# ---------------------------------------------------------------------------
# Aggregation and report
# ---------------------------------------------------------------------------


def _report_metric(report: Report, name: str) -> Any:
    return next(metric for metric in report.metrics if metric.identity.name == name)


def _report_subsets(report: Report, name: str) -> Dict[str, Any]:
    return {subset.name: subset for category in _report_metric(report, name).categories for subset in category.subsets}


@requires_upstream
class TestAggregation:
    @pytest.mark.parametrize('attack', [BENIGN_ATTACK, DEFAULT_ATTACK])
    def test_labels_primary_metric_and_excluded_samples(self, work_dir: str, attack: str) -> None:
        adapter = get_benchmark('agentdojo', config=make_task_config(work_dir, attack=attack))
        benign = attack == BENIGN_ATTACK
        value = {'utility': 1.0} if benign else {'utility': 1.0, 'targeted_asr': 0.0}
        scored = SampleScore(score=Score(value=value), sample_id=0, group_id=0)
        excluded = SampleScore(score=Score(status=ScoreStatus.EXCLUDED), sample_id=1, group_id=1)
        aggregates = adapter.aggregate_scores([scored, excluded])

        assert {agg.metric_name: agg.score for agg in aggregates} == value
        for agg in aggregates:
            assert agg.num == 1 and agg.ids == [0]
            assert agg.dimensions == {'condition': 'benign' if benign else 'under_attack'}
            assert agg.metadata['attack'] == attack
            assert agg.metadata['benchmark_version'] == BENCHMARK_VERSION
            assert agg.metadata['upstream_version'] == UPSTREAM_VERSION
        assert adapter.aggregate_scores([excluded]) == []
        assert adapter.primary_metric.name == ('utility' if benign else 'targeted_asr')


# ---------------------------------------------------------------------------
# End-to-end runs through run_task
# ---------------------------------------------------------------------------


def _run(scripts: Mapping[ScriptKey, Sequence[ModelOutput]], work_dir: Path, **config_kwargs: Any) -> Report:
    with installed_scripts(scripts):
        result = run_task(make_task_config(work_dir, eval_batch_size=2, **config_kwargs))
    assert isinstance(result['agentdojo'], Report)
    return result['agentdojo']


def _reviews(work_dir: Path, suite: str = SUITE) -> Dict[str, Dict[str, Any]]:
    """Persisted review rows of one suite, keyed by ``user_task_id``."""
    path = work_dir / 'reviews' / SCRIPTED_MODEL_NAME / f'agentdojo_{suite}.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return {row['sample_score']['score']['metadata']['user_task_id']: row for row in rows}


def _score(row: Mapping[str, Any]) -> Dict[str, Any]:
    return row['sample_score']['score']


def _subset_nums(report: Report, name: str) -> Dict[str, int]:
    return {subset_name: subset.num for subset_name, subset in _report_subsets(report, name).items()}


def _ground_truth_script(*user_task_ids: str) -> Dict[ScriptKey, List[ModelOutput]]:
    return {(SUITE, task_id): ground_truth_turns(SUITE, task_id) for task_id in user_task_ids}


@pytest.fixture
def _no_model_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.endswith('_API_KEY'):
            monkeypatch.delenv(name, raising=False)


@requires_upstream
@pytest.mark.usefixtures('_no_model_api_keys')
class TestEndToEnd:
    def test_benign_run_reports_utility_only(self, tmp_path: Path) -> None:
        report = _run(_ground_truth_script(USER_TASK_ID), tmp_path, user_task_ids=[USER_TASK_ID])

        assert [metric.identity.name for metric in report.metrics] == ['utility']
        primary = report.primary_metric
        assert (primary.identity.name, primary.identity.dimensions.get('condition')) == ('utility', 'benign')
        assert (primary.score, primary.num) == (pytest.approx(1.0), 1)
        assert _subset_nums(report, 'utility') == {SUITE: 1}
        # One generate call per ground-truth tool call plus the final text turn.
        assert model_calls() == [
            (SUITE, USER_TASK_ID, turn) for turn in range(len(ground_truth_turns(SUITE, USER_TASK_ID)))
        ]

    def test_attack_run_reports_utility_and_targeted_asr(self, tmp_path: Path) -> None:
        report = _run(
            _ground_truth_script(USER_TASK_ID),
            tmp_path,
            attack=DEFAULT_ATTACK,
            user_task_ids=[USER_TASK_ID],
            injection_task_ids=['injection_task_0', 'injection_task_1'],
        )

        primary = report.primary_metric
        assert (primary.identity.name, primary.identity.dimensions.get('condition')) == ('targeted_asr', 'under_attack')
        assert (primary.score, primary.num) == (pytest.approx(0.0), 2)
        utility = _report_metric(report, 'utility')
        assert (utility.score, utility.num, utility.identity.dimensions.get('condition')) == (
            pytest.approx(1.0),
            2,
            'under_attack',
        )
        assert _subset_nums(report, 'utility') == _subset_nums(report, 'targeted_asr') == {SUITE: 2}

    def test_unknown_tool_and_tool_error_keep_the_loop_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentdojo.functions_runtime import FunctionsRuntime

        runtime_calls: List[str] = []
        original = FunctionsRuntime.run_function

        def spy(self: Any, env: Any, function: str, kwargs: Mapping[str, Any], *args: Any, **kw: Any) -> Any:
            runtime_calls.append(function)
            return original(self, env, function, kwargs, *args, **kw)

        monkeypatch.setattr(FunctionsRuntime, 'run_function', spy)
        ground_truth = ground_truth_turns(SUITE, USER_TASK_ID)
        script = [tool_calls_turn(('nonexistent_tool', {'x': 1})), tool_calls_turn(('send_money', {})), *ground_truth]

        report = _run({(SUITE, USER_TASK_ID): script}, tmp_path, user_task_ids=[USER_TASK_ID])

        assert model_calls() == [(SUITE, USER_TASK_ID, turn) for turn in range(len(script))]
        row = _reviews(tmp_path)[USER_TASK_ID]
        observations = [message['content'] for message in row['messages'] if message['role'] == 'tool']
        assert observations[0] == INVALID_TOOL_ERROR.format(name='nonexistent_tool')
        assert observations[1].startswith('ValidationError: ')
        assert all(EVALSCOPE_UNKNOWN_TOOL_TEXT not in text for text in observations)
        # The unknown name never reached the upstream runtime, and scoring executes no tool.
        assert runtime_calls == [
            'send_money',
            *(turn.message.tool_calls[0].function.name for turn in ground_truth[:-1]),
        ]
        trace = row['sample_score']['sample_metadata'][RUN_RECORD_META_KEY]['tool_call_trace']
        assert [(item['function'], item['valid_function'], item['args']) for item in trace[:2]] == [
            ('nonexistent_tool', False, {'x': 1}),
            ('send_money', True, {}),
        ]
        assert _score(row)['value'] == {'utility': 1.0}
        assert _score(row)['metadata']['tool_call_trace_length'] == len(script) - 1
        assert _report_metric(report, 'utility').score == pytest.approx(1.0)

    def test_inference_errors_propagate_and_are_retried_on_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        suite = load_upstream().suites[SUITE]
        original_check = type(suite)._check_task_result

        def failing_check(self: Any, task: Any, *args: Any, **kwargs: Any) -> Any:
            if task.ID == 'user_task_3':
                raise ValueError('check exploded')
            return original_check(self, task, *args, **kwargs)

        monkeypatch.setattr(type(suite), '_check_task_result', failing_check)
        user_task_ids = [USER_TASK_ID, 'user_task_1', 'user_task_2', 'user_task_3']

        # Without ignore_errors a model error fails the run; user_task_2 has no script, so the model raises.
        with pytest.raises(LookupError, match='no script installed'):
            _run(_ground_truth_script(USER_TASK_ID), tmp_path / 'strict', user_task_ids=['user_task_2'])

        # With ignore_errors, the environment and model failures skip only their samples; the check error is
        # scored as excluded.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                type(suite.get_user_task_by_id('user_task_1')),
                'init_environment',
                staticmethod(_raise(RuntimeError('init_environment failed'))),
            )
            first = _run(
                _ground_truth_script(USER_TASK_ID, 'user_task_3'),
                tmp_path,
                user_task_ids=user_task_ids,
                ignore_errors=True,
            )

        assert (_report_metric(first, 'utility').num, _report_metric(first, 'utility').score) == (1, pytest.approx(1.0))
        # The environment failure stops its sample before any model call.
        assert {task for _, task, _ in model_calls()} == {USER_TASK_ID, 'user_task_2', 'user_task_3'}
        reviews = _reviews(tmp_path)
        assert sorted(reviews) == [USER_TASK_ID, 'user_task_3']
        check_score = _score(reviews['user_task_3'])
        assert (check_score['status'], check_score['value']) == (ScoreStatus.EXCLUDED, {})
        assert check_score['metadata']['excluded_reason'] == 'check_error: ValueError: check exploded'

        # A use_cache resume re-runs only the samples that failed.
        resumed = _run(
            _ground_truth_script(*user_task_ids), tmp_path, user_task_ids=user_task_ids, use_cache=str(tmp_path)
        )

        assert {task for _, task, _ in model_calls()} == {'user_task_1', 'user_task_2'}
        assert (_report_metric(resumed, 'utility').num, _report_metric(resumed, 'utility').score) == (
            3,
            pytest.approx(1.0),
        )
        assert sorted(_reviews(tmp_path)) == sorted(user_task_ids)

    @pytest.mark.parametrize('attack', [BENIGN_ATTACK, DEFAULT_ATTACK])
    def test_first_turn_model_length_scores_by_rule(self, tmp_path: Path, attack: str) -> None:
        report = _run(
            {(SUITE, USER_TASK_ID): [text_turn(stop_reason='model_length')]},
            tmp_path,
            attack=attack,
            user_task_ids=[USER_TASK_ID],
            injection_task_ids=[INJECTION_TASK_ID],
        )

        score = _score(_reviews(tmp_path)[USER_TASK_ID])
        assert score['value'] == (
            {'utility': 0.0} if attack == BENIGN_ATTACK else {'utility': 0.0, 'targeted_asr': 1.0}
        )
        assert score['metadata']['termination_reason'] == 'context_length_exceeded'
        assert all(metric.num == 1 for metric in report.metrics)

    def test_empty_final_output_is_scored(self, tmp_path: Path) -> None:
        script = [*ground_truth_turns(SUITE, USER_TASK_ID)[:-1], text_turn(text='')]
        report = _run({(SUITE, USER_TASK_ID): script}, tmp_path, user_task_ids=[USER_TASK_ID])

        score = _score(_reviews(tmp_path)[USER_TASK_ID])
        assert (score['metadata']['empty_output'], score['metadata']['termination_reason']) == (True, 'completed')
        # This user task's upstream utility only inspects the environment, so the empty answer still succeeds.
        assert score['value'] == {'utility': 1.0} and _report_metric(report, 'utility').num == 1

    @pytest.mark.parametrize('attack', [BENIGN_ATTACK, DEFAULT_ATTACK])
    def test_content_filtered_suite_reports_no_zero_metric(self, tmp_path: Path, attack: str) -> None:
        # The only slack sample gets a provider content-filter reply and is excluded.
        scripts = {
            **_ground_truth_script(USER_TASK_ID),
            ('slack', USER_TASK_ID): [text_turn(text='Filtered by provider.', stop_reason='content_filter')],
        }
        report = _run(scripts, tmp_path, subset=[SUITE, 'slack'], limit=1, attack=attack)

        slack_score = _score(next(iter(_reviews(tmp_path, 'slack').values())))
        assert (slack_score['status'], slack_score['value']) == (ScoreStatus.EXCLUDED, {})
        assert slack_score['metadata']['termination_reason'] == 'content_filter'
        assert slack_score['metadata']['excluded_reason'].startswith('content_filter')
        assert _subset_nums(report, 'utility') == {SUITE: 1}
        assert _report_metric(report, 'utility').score == pytest.approx(1.0)

    def test_rerun_review_scores_without_model_or_tool_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = {'attack': DEFAULT_ATTACK, 'user_task_ids': [USER_TASK_ID], 'injection_task_ids': [INJECTION_TASK_ID]}
        first = _run(_ground_truth_script(USER_TASK_ID), tmp_path, **config)
        first_score = _score(_reviews(tmp_path)[USER_TASK_ID])

        # A second run_task builds a new adapter; no script is installed and no agent tool call may run. Injection
        # planning still runs upstream ground truth on its own environments while the dataset loads.
        monkeypatch.setattr(AgentDojoToolBinding, 'execute', _raise(AssertionError('tool executed while scoring')))
        second = _run({}, tmp_path, use_cache=str(tmp_path), rerun_review=True, **config)

        assert model_calls() == []
        second_score = _score(_reviews(tmp_path)[USER_TASK_ID])
        assert second_score['value'] == first_score['value'] == {'utility': 1.0, 'targeted_asr': 0.0}
        for name in ('utility', 'targeted_asr'):
            assert (_report_metric(second, name).score, _report_metric(second, name).num) == (
                pytest.approx(_report_metric(first, name).score),
                1,
            )

    def test_cached_row_of_another_task_is_excluded(self, tmp_path: Path) -> None:
        _run(_ground_truth_script(USER_TASK_ID, 'user_task_1'), tmp_path, user_task_ids=[USER_TASK_ID, 'user_task_1'])

        # Index 1 now loads user_task_2, while the cache still holds user_task_1 there.
        report = _run(
            {}, tmp_path, user_task_ids=[USER_TASK_ID, 'user_task_2'], use_cache=str(tmp_path), rerun_review=True
        )

        assert model_calls() == []
        reviews = _reviews(tmp_path)
        assert _score(reviews[USER_TASK_ID])['value'] == {'utility': 1.0}
        drifted = _score(reviews['user_task_1'])
        assert (drifted['status'], drifted['value']) == (ScoreStatus.EXCLUDED, {})
        assert drifted['metadata']['excluded_reason'] == f"invalid_scoring_payload: ['{SAMPLE_META_KEY}.row_id']"
        assert 'loaded at index 1' in drifted['explanation']
        assert _subset_nums(report, 'utility') == {SUITE: 1}

    def test_llm_judge_strategy_is_ignored_with_a_warning(self, tmp_path: Path) -> None:
        with _captured_errors(logging.WARNING) as warnings:
            report = _run(
                _ground_truth_script(USER_TASK_ID), tmp_path, user_task_ids=[USER_TASK_ID], judge={'strategy': 'llm'}
            )

        assert any("ignores judge.strategy='llm'" in message for message in warnings), warnings
        assert _score(_reviews(tmp_path)[USER_TASK_ID])['value'] == {'utility': 1.0}
        assert _report_metric(report, 'utility').num == 1
