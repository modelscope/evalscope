"""Route a ``ToolCall`` to its backing Python function or environment command.

Tool handlers are registered under ``evalscope/agent/tools/`` via
``@register_agent_tool('name')`` and must expose an async
``run(call: ToolCall, env: AgentEnvironment | None) -> str`` callable.
"""

import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from evalscope.api.tool import ToolCall, ToolCallError
from .environment import AgentEnvironment

# Signature for an async tool handler.  Returns the textual observation
# that will be attached to the next user/tool message.
ToolHandler = Callable[[ToolCall, Optional[AgentEnvironment]], Awaitable[str]]


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
    ) -> None:
        self._handlers = handlers
        self._environment = environment

    @property
    def environment(self) -> Optional[AgentEnvironment]:
        return self._environment

    @property
    def tool_names(self) -> List[str]:
        return list(self._handlers.keys())

    async def execute(self, call: ToolCall) -> Tuple[str, Optional[ToolCallError], float]:
        """Run one tool call.

        Returns ``(observation, error, duration_seconds)``.  ``observation``
        is always a string so it can be appended to a ``ChatMessageTool``.
        """
        started = time.time()
        handler = self._handlers.get(call.function.name)
        if handler is None:
            err = ToolCallError(
                type='unknown',
                message=f"Tool '{call.function.name}' is not registered. "
                f'Available: {sorted(self._handlers.keys())}',
            )
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


__all__ = ['ToolExecutor', 'ToolHandler']
