"""Bind the upstream AgentDojo tools of one suite to one sample's task environment.

Tool execution mirrors upstream ``agentdojo.agent_pipeline.tool_execution.ToolsExecutor`` (0.1.35):

1. Empty or unknown function names never reach the runtime and get the upstream error text.
2. String arguments that ``literal_eval`` to a list are replaced by that list.
3. ``FunctionsRuntime.run_function`` runs on the bound environment; an error text is returned verbatim, otherwise
   the result is formatted with ``tool_result_to_str``.

Every executed or rejected call is appended to the tool call trace so scoring can replay it later. Calls whose
arguments could not be parsed are answered with ``ARGUMENT_PARSE_ERROR`` and are not traced.

Upstream ``send_email`` and ``create_file`` / ``append_to_file`` write ``datetime.datetime.now()`` into the
environment and the observation. Each executed call records its instant and runs with the tool clock pinned to it,
so a replay in another process reproduces the same observation and environment.

The upstream package is imported inside functions only.
"""

import contextlib
import contextvars
import datetime
import importlib
import types
from ast import literal_eval
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from pydantic import BaseModel

from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
from evalscope.api.agent import AgentContext, AgentLoopResult, ParsedAction
from evalscope.api.agent.environment import AgentEnvironment
from evalscope.api.agent.tool_executor import ToolHandler
from evalscope.api.messages import ChatMessage, ChatMessageTool
from evalscope.api.tool import ToolCall, ToolCallError, ToolInfo, ToolParams
from evalscope.utils.json_schema import json_schema

EMPTY_FUNCTION_NAME_ERROR = 'Empty function name provided. Provide a valid function name.'
INVALID_TOOL_ERROR = 'Invalid tool {name} provided.'
ARGUMENT_PARSE_ERROR = 'Failed to parse tool call arguments as a JSON object: {error}'

# Upstream ``agentdojo.agent_pipeline.llms.google_llm.EMPTY_FUNCTION_NAME``.
UPSTREAM_EMPTY_FUNCTION_NAME = '<empty-function-name>'

# Deliberate deviation: upstream ``ToolsExecutor`` answers with ``EMPTY_FUNCTION_NAME_ERROR`` only for the sentinel
# above, which only its Google client emits after normalizing an empty name; a literal ``''`` would get
# ``Invalid tool  provided.``. EvalScope's clients pass ``''`` through unchanged, so the binding emulates the
# Google-client normalization and treats both values as empty.
_EMPTY_FUNCTION_NAMES = frozenset({'', UPSTREAM_EMPTY_FUNCTION_NAME})

# ---------------------------------------------------------------------------
# Pinnable tool clock
# ---------------------------------------------------------------------------

# Upstream modules whose tools call ``datetime.datetime.now()`` through their ``import datetime`` attribute.
# ``types.Email.timestamp`` also defaults to ``now``, but no tool relies on that default.
CLOCK_STAMPING_MODULES = (
    'agentdojo.default_suites.v1.tools.email_client',
    'agentdojo.default_suites.v1.tools.cloud_drive_client',
)

_REAL_DATETIME = datetime.datetime

# A context variable rather than a global patch: samples run tool calls concurrently in evaluator worker threads,
# and each call must only see its own instant.
_PINNED_NOW: contextvars.ContextVar[Optional[datetime.datetime]] = contextvars.ContextVar(
    'agentdojo_pinned_now', default=None
)


class _PinnableDatetime(datetime.datetime):
    """``datetime.datetime`` whose ``now()`` returns the instant pinned in the current context, if any."""

    @classmethod
    def now(cls, tz: Optional[datetime.tzinfo] = None) -> datetime.datetime:
        pinned = _PINNED_NOW.get()
        if pinned is None:
            return _REAL_DATETIME.now(tz)
        return pinned if tz is None else pinned.astimezone(tz)


_CLOCK_PROXY = types.ModuleType('datetime', 'AgentDojo clock proxy of the datetime module.')
_CLOCK_PROXY.__dict__.update({key: value for key, value in vars(datetime).items() if not key.startswith('__')})
_CLOCK_PROXY.datetime = _PinnableDatetime


def install_clock_proxy() -> None:
    """Point the ``datetime`` attribute of every stamping module at the clock proxy (idempotent).

    Only an attribute that still is the real ``datetime`` module is replaced, so a clock frozen by someone else is
    left alone. Without a pinned instant the proxy behaves exactly like the real module.
    """
    for module_name in CLOCK_STAMPING_MODULES:
        module = importlib.import_module(module_name)
        if getattr(module, 'datetime', None) is datetime:
            module.datetime = _CLOCK_PROXY


@contextlib.contextmanager
def pinned_clock(instant: datetime.datetime) -> Iterator[None]:
    """Make the proxied ``datetime.datetime.now()`` return ``instant`` within the current context."""
    token = _PINNED_NOW.set(instant)
    try:
        yield
    finally:
        _PINNED_NOW.reset(token)


# ---------------------------------------------------------------------------
# Tool binding
# ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Convert ``literal_eval`` output (tuples, sets, non-str keys) into JSON-native values.

    Keeps trace arguments identical before and after a JSON round trip through the prediction cache.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {key if isinstance(key, str) else str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    return str(value)


class ToolCallTraceItem(BaseModel):
    """One tool call issued by the agent."""

    function: str
    """Function name as issued by the model; may be ``''`` or an unknown name."""

    args: Dict[str, Any]
    """Converted arguments for a valid name, raw arguments for an invalid name (JSON-native)."""

    observation: str
    """Exact observation text returned to the model."""

    call_id: Optional[str] = None
    valid_function: bool
    """Whether ``function`` is a tool of the suite."""

    formatter_error: bool = False
    """``tool_result_to_str`` raised; ``observation`` is the formatted ``TypeError``."""

    called_at: Optional[datetime.datetime] = None
    """Clock instant the call ran with; ``None`` for rejected calls."""


class AgentDojoToolBinding:
    """Bind the upstream functions of one suite to one sample's environment, with its own ``FunctionsRuntime``."""

    def __init__(self, suite: Any, environment: Any) -> None:
        from agentdojo.functions_runtime import FunctionsRuntime

        install_clock_proxy()
        self._suite = suite
        self.environment = environment
        self.tool_names = frozenset(fn.name for fn in suite.tools)
        self.trace: List[ToolCallTraceItem] = []
        self._runtime = FunctionsRuntime(suite.tools)

    def build(self) -> Tuple[Dict[str, ToolHandler], List[ToolInfo]]:
        """Return the handlers and ToolInfos of the suite tools, in ``suite.tools`` order.

        Tool calls run inline on the sample's event loop: each evaluator worker thread has its own loop and the
        calls of one sample are executed sequentially, like upstream ``ToolsExecutor``.
        """

        async def handle(call: ToolCall, env: Optional[AgentEnvironment]) -> str:
            if call.parse_error is not None:
                # No runtime call and no trace item: the environment stays unchanged.
                return ARGUMENT_PARSE_ERROR.format(error=call.parse_error)
            return self.execute(call.function.name, call.function.arguments, call.id)

        tool_infos = []
        for fn in self._suite.tools:
            schema = json_schema(fn.parameters)
            parameters = ToolParams(properties=schema.properties or {}, required=schema.required or [])
            tool_infos.append(ToolInfo(name=fn.name, description=fn.description, parameters=parameters))
        return {info.name: handle for info in tool_infos}, tool_infos

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        call_id: Optional[str],
        called_at: Optional[datetime.datetime] = None,
    ) -> str:
        """Run one call of a suite tool on the bound environment, trace it and return the observation.

        ``called_at`` is the clock instant to run with; ``None`` takes the current wall-clock time.
        """
        # ``tool_execution`` imports the ``agent_pipeline`` package; the adapter has already checked the version.
        from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str

        converted = {
            key: literal_eval(value) if isinstance(value, str) and is_string_list(value) else value
            for key, value in arguments.items()
        }
        instant = called_at if called_at is not None else datetime.datetime.now()
        with pinned_clock(instant):
            result, error = self._runtime.run_function(self.environment, name, converted)
        formatter_error = False
        if error is not None:
            observation = error
        else:
            try:
                observation = tool_result_to_str(result)
            except TypeError as exc:
                # Upstream aborts the pipeline here; the binding reports it as a tool error instead.
                observation = f'TypeError: {exc}'
                formatter_error = True
        self.trace.append(
            ToolCallTraceItem(
                function=name,
                args=to_jsonable(converted),
                observation=observation,
                call_id=call_id,
                valid_function=True,
                formatter_error=formatter_error,
                called_at=instant,
            )
        )
        return observation

    def handle_invalid_function(self, call: ToolCall) -> str:
        """Answer an empty or unknown name without touching the runtime; the call is traced with raw arguments."""
        return self._reject(call.function.name, call.function.arguments, call.id)

    def replay(self, item: ToolCallTraceItem) -> str:
        """Re-execute one recorded item (with its recorded clock instant) and return the observation.

        Arguments of valid items are already converted; the conversion is idempotent.
        """
        if item.valid_function:
            return self.execute(item.function, item.args, item.call_id, item.called_at)
        return self._reject(item.function, item.args, item.call_id)

    def _reject(self, name: str, arguments: Mapping[str, Any], call_id: Optional[str]) -> str:
        observation = (
            EMPTY_FUNCTION_NAME_ERROR if name in _EMPTY_FUNCTION_NAMES else INVALID_TOOL_ERROR.format(name=name)
        )
        self.trace.append(
            ToolCallTraceItem(
                function=name,
                args=to_jsonable(dict(arguments)),
                observation=observation,
                call_id=call_id,
                valid_function=False,
            )
        )
        return observation


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class AgentDojoStrategy(FunctionCallingStrategy):
    """``function_calling`` without submit tool or nudges, with upstream-compatible observations.

    * The exposed tools are exactly the suite tools; a ``submit`` call is an invalid name.
    * The loop ends as soon as the model replies without tool calls, like upstream ``ToolsExecutionLoop``.
    * Unknown function names get the upstream ``ToolsExecutor`` texts instead of EvalScope's "not registered" text,
      and tool errors are never flagged on the message.

    Built per sample around one binding and deliberately not registered, so ``get_strategy('function_calling')``
    keeps returning the generic strategy.
    """

    max_nudges: int = 0
    max_parse_error_nudges: int = 0

    def __init__(self, binding: AgentDojoToolBinding) -> None:
        super().__init__(system_prompt=None, include_submit_tool=False)
        self._binding = binding

    def format_observation(
        self,
        call: ToolCall,
        observation: str,
        error: Optional[ToolCallError],
        parsed: ParsedAction,
        ctx: AgentContext,
    ) -> ChatMessage:
        """Return the tool observation, replacing the executor's text for names without a handler."""
        if call.function.name not in self._binding.tool_names:
            if call.parse_error is not None:
                # Argument parse failures take precedence and are not traced.
                observation = ARGUMENT_PARSE_ERROR.format(error=call.parse_error)
            else:
                observation = self._binding.handle_invalid_function(call)
        return ChatMessageTool(content=observation, tool_call_id=call.id, function=call.function.name, error=None)

    def extract_final_answer(self, result: AgentLoopResult) -> str:
        """Return the text of the last assistant message, or ``''`` when it has no text content."""
        for message in reversed(result.messages):
            if message.role == 'assistant':
                return message.text if message.content else ''
        return ''
