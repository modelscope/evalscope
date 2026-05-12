"""AgentLoop: the model-agnostic orchestration core.

Takes a model + strategy + environment + tool executor and drives the
generate → parse → tool_call → observe loop.  It does NOT decide prompt
formats or termination semantics; those belong to :class:`AgentStrategy`.

The loop is async so it can naturally express parallel tool execution and
streaming generation in the future.  ``AgentAdapter._on_inference`` bridges
it into the synchronous evaluation pipeline through ``AsyncioLoopRunner``.
"""

import time
from typing import List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageTool
from evalscope.api.model import Model, ModelOutput
from evalscope.utils.logger import get_logger
from .environment import AgentEnvironment
from .strategy import AgentStrategy
from .tool_executor import ToolExecutor
from .trace import AgentTrace, EventType
from .types import AgentContext, AgentLoopResult

logger = get_logger()


class AgentLoop:
    """Generic multi-turn tool-use loop.

    Intended for use from ``AgentAdapter`` (or ``DefaultDataAdapter`` when a
    global ``agent_config`` is set).  The caller owns the lifecycle of the
    environment (create before ``run``, close after).
    """

    def __init__(
        self,
        model: Model,
        strategy: AgentStrategy,
        tool_executor: ToolExecutor,
        *,
        environment: Optional[AgentEnvironment] = None,
        max_steps: int = 10,
        trace: Optional[AgentTrace] = None,
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.tool_executor = tool_executor
        self.environment = environment
        self.max_steps = max_steps
        self.trace = trace or AgentTrace(
            strategy=getattr(strategy, 'name', None),
            environment=environment.name if environment else None,
            max_steps=max_steps,
        )

    async def run(
        self,
        ctx: AgentContext,
    ) -> AgentLoopResult:
        """Drive the loop until the strategy signals completion or ``max_steps``.

        The provided ``ctx.messages`` is mutated in place: each iteration
        appends the new assistant reply and any tool observations, so the
        caller can inspect the final conversation afterwards.
        """
        # Inject the system prompt once at step 0 (if provided).
        system_prompt = self.strategy.build_system_prompt(ctx)
        if system_prompt and not any(m.role == 'system' for m in ctx.messages):
            ctx.messages.insert(0, ChatMessageSystem(content=system_prompt))

        final_output: Optional[ModelOutput] = None

        while ctx.step < self.max_steps:
            # ---- generate ----
            generate_messages = self.strategy.prepare_messages(ctx)
            mode = self.strategy.tool_schema_mode()
            tools = self.strategy.tools(ctx) if mode == 'function_calling' else []

            started = time.time()
            output = self.model.generate(input=generate_messages, tools=tools or None)
            latency_ms = (time.time() - started) * 1000
            ctx.last_output = output
            final_output = output

            assistant_msg = output.message
            ctx.messages.append(assistant_msg)

            self.trace.add_event(
                step=ctx.step,
                type=EventType.MODEL_GENERATE,
                latency_ms=latency_ms,
                token_usage=_extract_usage(output),
                payload={'stop_reason': output.stop_reason},
            )

            # ---- parse ----
            parsed = self.strategy.parse_output(output, ctx)
            if parsed.error:
                self.trace.add_event(
                    step=ctx.step,
                    type=EventType.ERROR,
                    payload={
                        'source': 'parse',
                        'message': parsed.error
                    },
                )

            # ---- terminate? ----
            if self.strategy.is_done(parsed, ctx):
                if parsed.final_answer is not None:
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.SUBMIT,
                        payload={'final_answer': parsed.final_answer},
                    )
                break

            # ---- tool execution ----
            if parsed.tool_calls:
                for call in parsed.tool_calls:
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.TOOL_CALL,
                        payload={
                            'name': call.function.name,
                            'arguments': call.function.arguments,
                            'id': call.id,
                        },
                    )
                    observation, error, duration = await self.tool_executor.execute(call)
                    ctx.messages.append(
                        ChatMessageTool(
                            content=observation,
                            tool_call_id=call.id,
                            function=call.function.name,
                            error=error,
                        )
                    )
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.TOOL_RESULT,
                        latency_ms=duration * 1000,
                        payload={
                            'name': call.function.name,
                            'id': call.id,
                            'error': error.type if error else None,
                            'preview': observation[:500] if isinstance(observation, str) else None,
                        },
                    )

            ctx.step += 1

        if ctx.step >= self.max_steps:
            logger.info(f'AgentLoop reached max_steps={self.max_steps} '
                        f'for sample {ctx.sample_id}; terminating.')
            self.trace.add_event(
                step=ctx.step,
                type=EventType.ERROR,
                payload={
                    'source': 'loop',
                    'message': 'max_steps_exceeded'
                },
            )

        assert final_output is not None, 'AgentLoop.run ended without any generate() call'
        return AgentLoopResult(
            messages=ctx.messages,
            final_output=final_output,
            trace=self.trace,
        )


def _extract_usage(output: ModelOutput) -> Optional[dict]:
    usage = output.usage
    if usage is None:
        return None
    try:
        return {
            'input': int(getattr(usage, 'input_tokens', 0) or 0),
            'output': int(getattr(usage, 'output_tokens', 0) or 0),
            'total': int(getattr(usage, 'total_tokens', 0) or 0),
        }
    except Exception:  # noqa: BLE001
        return None


__all__ = ['AgentLoop']
