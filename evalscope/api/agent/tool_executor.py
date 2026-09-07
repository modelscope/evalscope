"""Route a ``ToolCall`` to its backing Python function or environment command.

Tool handlers are registered under ``evalscope/agent/tools/`` via
``@register_agent_tool('name')`` and must expose an async
``run(call: ToolCall, env: AgentEnvironment | None) -> ToolObservation`` callable.

With ``validate_arguments=True`` the executor also checks every call against
the ``ToolInfo`` schema the model was shown and refuses to dispatch a call
that violates it, returning a ``parsing`` :class:`ToolCallError` the same way
an unregistered tool yields an ``unknown`` one.
"""

import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from evalscope.api.tool import ToolCall, ToolCallError, ToolInfo, validate_tool_arguments

from .environment import AgentEnvironment
from .types import ToolExecutionOutput

# Signature for an async tool handler.  Returns the textual observation
# that will be attached to the next user/tool message.
ToolObservation = str | ToolExecutionOutput
ToolHandler = Callable[[ToolCall, Optional[AgentEnvironment]], Awaitable[ToolObservation]]


class ToolExecutor:
    """Dispatches ``ToolCall`` instances to registered async handlers.

    Kept stateless on purpose: the AgentLoop creates one per sample with
    a prebuilt ``{name: handler}`` mapping derived from ``NativeAgentConfig.tools``
    or ``AgentAdapter.build_tools``.
    """

    def __init__(
        self,
        handlers: Dict[str, ToolHandler],
        environment: Optional[AgentEnvironment] = None,
        *,
        tool_infos: Optional[List[ToolInfo]] = None,
        validate_arguments: bool = False,
    ) -> None:
        self._handlers = handlers
        self._environment = environment
        self._tool_infos: Dict[str, ToolInfo] = {tool.name: tool for tool in tool_infos or []}
        self._validate_arguments = validate_arguments

    @property
    def environment(self) -> Optional[AgentEnvironment]:
        return self._environment

    @property
    def tool_names(self) -> List[str]:
        return list(self._handlers.keys())

    async def execute(self, call: ToolCall) -> Tuple[ToolObservation, Optional[ToolCallError], float]:
        """Run one tool call.

        Returns ``(observation, error, duration_seconds)``. Existing handlers
        return strings; attachment-producing tools may return
        :class:`ToolExecutionOutput`.

        When argument validation is enabled and the call violates the schema
        advertised for its tool, the handler is not invoked: the violation is
        returned as a ``parsing`` :class:`ToolCallError` so the model can
        correct the call on its next turn.
        """
        started = time.time()
        handler = self._handlers.get(call.function.name)
        if handler is None:
            err = ToolCallError(
                type='unknown',
                message=f"Tool '{call.function.name}' is not registered. Available: {sorted(self._handlers.keys())}",
            )
            return err.message, err, time.time() - started

        if self._validate_arguments:
            violation = self._schema_violation(call)
            if violation is not None:
                err = ToolCallError(type='parsing', message=violation)
                return err.message, err, time.time() - started

        try:
            observation = await handler(call, self._environment)
            return observation, None, time.time() - started
        except TimeoutError as exc:
            return str(exc), ToolCallError(type='timeout', message=str(exc)), time.time() - started
        except PermissionError as exc:
            return str(exc), ToolCallError(type='permission', message=str(exc)), time.time() - started
        except Exception as exc:  # noqa: BLE001 - generic boundary
            return str(exc), ToolCallError(type='unknown', message=str(exc)), time.time() - started

    def _schema_violation(self, call: ToolCall) -> Optional[str]:
        """Describe how ``call`` violates its tool's schema, or ``None``.

        Tools without a registered :class:`ToolInfo` cannot be validated and
        are dispatched as-is.
        """
        tool = self._tool_infos.get(call.function.name)
        if tool is None:
            return None
        violation = validate_tool_arguments(call, tool)
        if violation is None:
            return None
        return (
            f"Invalid arguments for tool '{call.function.name}': {violation}. "
            "Arguments must match the tool's parameter schema."
        )


__all__ = ['ToolExecutor', 'ToolHandler', 'ToolObservation']
