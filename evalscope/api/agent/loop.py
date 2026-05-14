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

from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageTool, ChatMessageUser
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

            logger.debug(
                f'[AgentLoop] sample={ctx.sample_id} step={ctx.step}/{self.max_steps} '
                f'strategy={getattr(self.strategy, "name", "?")} mode={mode} '
                f'tools=[{", ".join(t.name for t in tools)}] '
                f'messages_count={len(generate_messages)}'
            )

            started = time.time()
            output = self.model.generate(input=generate_messages, tools=tools or None)
            latency_ms = (time.time() - started) * 1000
            ctx.last_output = output
            final_output = output

            # Deep-copy before appending to ctx.messages so that downstream
            # adapter code (e.g. ``output.completion = final_text`` in
            # ``DefaultDataAdapter._extract_final_answer``) cannot retro-
            # actively mutate the message we are about to persist to JSONL.
            # Without this, the assistant's reasoning text gets overwritten
            # by the extracted final answer because ``output.message`` and
            # ``ctx.messages[-1]`` would otherwise share the same object.
            assistant_msg = output.message.model_copy(deep=True)
            ctx.messages.append(assistant_msg)

            logger.debug(
                f'[AgentLoop] sample={ctx.sample_id} step={ctx.step} generate done '
                f'latency={latency_ms:.0f}ms stop_reason={output.stop_reason} '
                f'tool_calls={assistant_msg.tool_calls} '
                f'content={assistant_msg.text}'
            )

            self.trace.add_event(
                step=ctx.step,
                type=EventType.MODEL_GENERATE,
                message_id=assistant_msg.id,
                latency_ms=latency_ms,
                token_usage=_extract_usage(output),
                payload={'stop_reason': output.stop_reason},
            )

            # ---- parse ----
            parsed = self.strategy.parse_output(output, ctx)
            if parsed.error:
                logger.debug(f'[AgentLoop] sample={ctx.sample_id} step={ctx.step} parse_error: {parsed.error}')
                self.trace.add_event(
                    step=ctx.step,
                    type=EventType.ERROR,
                    message_id=assistant_msg.id,
                    payload={
                        'source': 'parse',
                        'message': parsed.error
                    },
                )

            # ---- terminate? ----
            if self.strategy.is_done(parsed, ctx):
                logger.debug(
                    f'[AgentLoop] sample={ctx.sample_id} step={ctx.step} is_done=True '
                    f'final_answer={str(parsed.final_answer)[:100]!r}'
                )
                if parsed.final_answer is not None:
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.SUBMIT,
                        message_id=assistant_msg.id,
                        payload={'final_answer': parsed.final_answer},
                    )
                break

            # ---- no tool calls but not done → nudge ----
            # The model didn't call any tool and didn't submit a final
            # answer.  Ask the strategy whether a nudge is appropriate.
            if not parsed.tool_calls:
                logger.debug(
                    f'[AgentLoop] sample={ctx.sample_id} step={ctx.step} '
                    f'no_tool_calls, should_nudge={self.strategy.should_nudge(parsed, ctx)}'
                )
                if self.strategy.should_nudge(parsed, ctx):
                    nudge = ChatMessageUser(
                        content='No tool was called. Please use an available tool '
                        'or call the submit tool with your final answer.',
                    )
                    ctx.messages.append(nudge)
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.NUDGE,
                        message_id=nudge.id,
                        payload={
                            'source': 'nudge',
                            'message': 'no_tool_call_reminder'
                        },
                    )
                    ctx.step += 1
                    continue
                else:
                    # Strategy declines nudge; treat as implicit final answer.
                    final = parsed.raw_text or ''
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.SUBMIT,
                        message_id=None,
                        payload={
                            'final_answer': final,
                            'source': 'implicit_no_nudge'
                        },
                    )
                    break

            # ---- tool execution ----
            if parsed.tool_calls:
                logger.debug(
                    f'[AgentLoop] sample={ctx.sample_id} step={ctx.step} '
                    f'executing {len(parsed.tool_calls)} tool call(s): '
                    f'{[c.function.name for c in parsed.tool_calls]}'
                )
                last_obs_id: Optional[str] = None
                for call in parsed.tool_calls:
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.TOOL_CALL,
                        message_id=assistant_msg.id,
                        payload={
                            'name': call.function.name,
                            'arguments': call.function.arguments,
                            'id': call.id,
                        },
                    )
                    observation, error, duration = await self.tool_executor.execute(call)
                    logger.debug(
                        f'[AgentLoop] sample={ctx.sample_id} step={ctx.step} '
                        f'tool={call.function.name} duration={duration*1000:.0f}ms '
                        f'error={error.type if error else None} '
                        f'obs_len={len(observation) if isinstance(observation, str) else 0}'
                    )
                    # Let the strategy decide how to format the observation
                    # (ChatMessageTool for FC, ChatMessageUser for textual_block).
                    obs_msg = self.strategy.format_observation(call, observation, error)
                    ctx.messages.append(obs_msg)
                    last_obs_id = obs_msg.id
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.TOOL_RESULT,
                        message_id=obs_msg.id,
                        latency_ms=duration * 1000,
                        payload={
                            'name': call.function.name,
                            'id': call.id,
                            'error': error.type if error else None,
                            'preview': observation[:500] if isinstance(observation, str) else None,
                        },
                    )

                # Post-execution termination check.  Strategies like
                # ``swe_bench_backticks`` detect completion sentinels in the
                # tool observation, which is only available *after*
                # execution.  FC strategies return False here because
                # parsed.tool_calls is still non-empty.
                if self.strategy.is_done(parsed, ctx):
                    self.trace.add_event(
                        step=ctx.step,
                        type=EventType.SUBMIT,
                        message_id=last_obs_id,
                        payload={'source': 'post_execution_check'},
                    )
                    break

            ctx.step += 1

        if ctx.step >= self.max_steps:
            logger.info(f'AgentLoop reached max_steps={self.max_steps} '
                        f'for sample {ctx.sample_id}; terminating.')
            if logger.isEnabledFor(10):  # DEBUG level
                logger.debug(
                    f'[AgentLoop] sample={ctx.sample_id} max_steps_exceeded '
                    f'total_messages={len(ctx.messages)}'
                )
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
