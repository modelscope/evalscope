"""AgentLoop: the model-agnostic orchestration core.

Takes a model + strategy + environment + tool executor and drives the
generate → parse → tool_call → observe loop.  It does NOT decide prompt
formats or termination semantics; those belong to :class:`AgentStrategy`.

Termination has a **single** signal: ``ParsedAction.final_answer`` set by
the strategy (in either ``parse_output`` or ``format_observation``).
The loop checks ``strategy.is_done(parsed, ctx)`` after each phase and
breaks when it returns True.  No exceptions are used for control flow.

The loop is async so it can naturally express parallel tool execution and
streaming generation in the future.  ``AgentAdapter._on_inference`` bridges
it into the synchronous evaluation pipeline through ``AsyncioLoopRunner``.
"""

import logging
import time
from typing import Any, List, Optional, Tuple

from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.api.model import Model, ModelOutput, ModelUsage
from evalscope.utils.logger import get_logger
from .constants import NUDGE_PROMPT, LoopMessages, MetadataKeys, SubmissionSources, ToolSchemaModes, TraceSources
from .environment import AgentEnvironment
from .strategy import AgentStrategy
from .tool_executor import ToolExecutor
from .trace import AgentTrace, EventType
from .types import AgentContext, AgentLoopResult, ParsedAction

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
            framework='native',
            strategy=getattr(strategy, 'name', None),
            environment=environment.name if environment else None,
            max_steps=max_steps,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, ctx: AgentContext) -> AgentLoopResult:
        """Drive the loop until the strategy signals completion or ``max_steps``.

        The provided ``ctx.messages`` is mutated in place: each iteration
        appends the new assistant reply and any tool observations.  The
        returned ``AgentLoopResult.messages`` is a shallow copy so that
        post-run mutations on either side cannot pollute the other.
        """
        self._inject_system_prompt(ctx)

        final_output: Optional[ModelOutput] = None
        terminated_by_strategy = False

        while ctx.step < self.max_steps:
            # ---- generate ----
            output, latency_ms = await self._generate(ctx)
            final_output = output
            assistant_msg = self._snapshot_assistant_message(output)
            ctx.messages.append(assistant_msg)
            self._emit_generate(ctx, assistant_msg, output, latency_ms)

            # ---- terminate on model context overflow ----
            # Provider layer (openai_handle_bad_request / anthropic_handle_bad_request)
            # converts BadRequestError → stop_reason='model_length' instead of raising;
            # we surface that here as a graceful loop end so the rest of the
            # evaluation continues. Compaction / message-trimming recovery
            # (cf. inspect_ai's _handle_overflow) is intentionally out of scope.
            if output.stop_reason == 'model_length':
                self._emit_context_overflow(ctx)
                terminated_by_strategy = True
                break

            # ---- parse ----
            parsed = self.strategy.parse_output(output, ctx)
            if parsed.error:
                self._emit_parse_error(ctx, assistant_msg, parsed)

            # ---- terminate from parse? ----
            if self.strategy.is_done(parsed, ctx):
                self._emit_submit(ctx, assistant_msg, parsed)
                terminated_by_strategy = True
                break

            # ---- no tool calls but not done → nudge or implicit submit ----
            if not parsed.tool_calls:
                if self._try_nudge(parsed, ctx):
                    continue
                self._emit_implicit_submit(ctx, parsed)
                terminated_by_strategy = True
                break

            # ---- tool execution (may set parsed.final_answer post-hoc) ----
            if await self._run_tools(parsed, ctx, assistant_msg):
                terminated_by_strategy = True
                break

            ctx.step += 1

        if not terminated_by_strategy:
            self._emit_max_steps_exceeded(ctx)

        if final_output is None:
            raise RuntimeError('AgentLoop.run ended without any generate() call')

        return AgentLoopResult(
            messages=list(ctx.messages),
            final_output=final_output,
            trace=self.trace,
        )

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    def _inject_system_prompt(self, ctx: AgentContext) -> None:
        """Insert the strategy's system prompt at index 0 if absent."""
        system_prompt = self.strategy.build_system_prompt(ctx)
        if system_prompt and not any(m.role == 'system' for m in ctx.messages):
            ctx.messages.insert(0, ChatMessageSystem(content=system_prompt))

    async def _generate(self, ctx: AgentContext) -> Tuple[ModelOutput, float]:
        """Run one ``model.generate_async`` round and return (output, latency_ms)."""
        generate_messages = self.strategy.prepare_messages(ctx)
        mode = self.strategy.tool_schema_mode()
        tools = self.strategy.tools(ctx) if mode == ToolSchemaModes.FUNCTION_CALLING else []

        self._dbg(
            ctx,
            f'{ctx.step}/{self.max_steps} '
            f'strategy={getattr(self.strategy, "name", "?")} mode={mode} '
            f'tools=[{", ".join(t.name for t in tools)}] '
            f'messages_count={len(generate_messages)}',
        )

        started = time.monotonic()
        output = await self.model.generate_async(input=generate_messages, tools=tools or None)
        latency_ms = (time.monotonic() - started) * 1000
        ctx.last_output = output
        return output, latency_ms

    def _snapshot_assistant_message(self, output: ModelOutput) -> ChatMessage:
        """Deep-copy the assistant message before appending to ``ctx.messages``.

        Without this, ``DefaultDataAdapter._extract_final_answer`` would later
        overwrite the persisted reasoning text by mutating the same
        ``output.message`` object that lives in ``ctx.messages``.
        """
        return output.message.model_copy(deep=True)

    def _try_nudge(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        """Inject a 'please call a tool' reminder if the strategy allows it.

        Returns True when a nudge was injected (caller should ``continue``);
        False when the strategy declined and the caller should treat the
        current output as an implicit final answer.
        """
        should_nudge = self.strategy.should_nudge(parsed, ctx)
        self._dbg(ctx, f'no_tool_calls, should_nudge={should_nudge}')
        if not should_nudge:
            return False

        nudge = ChatMessageUser(content=NUDGE_PROMPT)
        ctx.messages.append(nudge)
        self.trace.add_event(
            step=ctx.step,
            type=EventType.NUDGE,
            message_id=nudge.id,
            payload={
                'source': TraceSources.NUDGE,
                'message': LoopMessages.NO_TOOL_CALL_REMINDER,
            },
        )
        ctx.step += 1
        return True

    async def _run_tools(
        self,
        parsed: ParsedAction,
        ctx: AgentContext,
        assistant_msg: ChatMessage,
    ) -> bool:
        """Execute every tool call in ``parsed.tool_calls`` sequentially.

        Returns True when ``strategy.is_done`` becomes True after a tool
        observation (post-tool termination), signalling the outer loop to
        break without re-checking ``is_done``.
        """
        self._dbg(
            ctx,
            f'executing {len(parsed.tool_calls)} tool call(s): '
            f'{[c.function.name for c in parsed.tool_calls]}',
        )
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
            self._dbg(
                ctx,
                f'tool={call.function.name} duration={duration*1000:.0f}ms '
                f'error={error.type if error else None} '
                f'obs_len={len(observation) if isinstance(observation, str) else 0}',
            )
            # ``format_observation`` may signal completion by mutating
            # ``parsed.final_answer``; the post-execution ``is_done`` check
            # below picks it up.
            obs_msg = self.strategy.format_observation(call, observation, error, parsed, ctx)
            ctx.messages.append(obs_msg)
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

            if self.strategy.is_done(parsed, ctx):
                self._emit_post_tool_submit(ctx, obs_msg, call, parsed, duration)
                return True

        return False

    # ------------------------------------------------------------------
    # Trace event emitters
    # ------------------------------------------------------------------

    def _emit_generate(
        self,
        ctx: AgentContext,
        assistant_msg: ChatMessage,
        output: ModelOutput,
        latency_ms: float,
    ) -> None:
        self._dbg(
            ctx,
            f'generate done latency={latency_ms:.0f}ms '
            f'stop_reason={output.stop_reason} '
            f'tool_calls={assistant_msg.tool_calls} '
            f'content={assistant_msg.text}',
        )
        if output.usage is not None:
            self.trace.total_usage = (
                output.usage if self.trace.total_usage is None else self.trace.total_usage + output.usage
            )
        self.trace.add_event(
            step=ctx.step,
            type=EventType.MODEL_GENERATE,
            message_id=assistant_msg.id,
            latency_ms=latency_ms,
            token_usage=_extract_usage(output),
            payload={'stop_reason': output.stop_reason},
        )

    def _emit_parse_error(
        self,
        ctx: AgentContext,
        assistant_msg: ChatMessage,
        parsed: ParsedAction,
    ) -> None:
        self._dbg(ctx, f'parse_error: {parsed.error}')
        self.trace.add_event(
            step=ctx.step,
            type=EventType.ERROR,
            message_id=assistant_msg.id,
            payload={
                'source': TraceSources.PARSE,
                'message': parsed.error,
            },
        )

    def _emit_submit(
        self,
        ctx: AgentContext,
        assistant_msg: ChatMessage,
        parsed: ParsedAction,
    ) -> None:
        self._dbg(
            ctx,
            f'is_done=True final_answer={str(parsed.final_answer)[:100]!r}',
        )
        if parsed.final_answer is not None:
            self.trace.add_event(
                step=ctx.step,
                type=EventType.SUBMIT,
                message_id=assistant_msg.id,
                payload={'final_answer': parsed.final_answer},
            )

    def _emit_implicit_submit(self, ctx: AgentContext, parsed: ParsedAction) -> None:
        final = parsed.raw_text or ''
        self.trace.add_event(
            step=ctx.step,
            type=EventType.SUBMIT,
            message_id=None,
            payload={
                'final_answer': final,
                'source': SubmissionSources.IMPLICIT_NO_NUDGE,
            },
        )

    def _emit_post_tool_submit(
        self,
        ctx: AgentContext,
        obs_msg: ChatMessage,
        call: Any,
        parsed: ParsedAction,
        duration: float,
    ) -> None:
        source = ctx.metadata.get(MetadataKeys.SUBMISSION_SOURCE, SubmissionSources.POST_TOOL)
        self.trace.add_event(
            step=ctx.step,
            type=EventType.SUBMIT,
            message_id=obs_msg.id,
            latency_ms=duration * 1000,
            payload={
                'source': source,
                'name': call.function.name,
                'id': call.id,
                'final_answer_preview': str(parsed.final_answer)[:120],
            },
        )
        self._dbg(
            ctx,
            f'is_done after tool {call.function.name}; '
            f'final_answer_len={len(str(parsed.final_answer or ""))}',
        )

    def _emit_max_steps_exceeded(self, ctx: AgentContext) -> None:
        logger.info(f'AgentLoop reached max_steps={self.max_steps} '
                    f'for sample {ctx.sample_id}; terminating.')
        if logger.isEnabledFor(logging.DEBUG):
            self._dbg(ctx, f'max_steps_exceeded total_messages={len(ctx.messages)}')
        self.trace.add_event(
            step=ctx.step,
            type=EventType.ERROR,
            payload={
                'source': TraceSources.LOOP,
                'message': LoopMessages.MAX_STEPS_EXCEEDED,
            },
        )

    def _emit_context_overflow(self, ctx: AgentContext) -> None:
        logger.warning(
            f'AgentLoop sample={ctx.sample_id} step={ctx.step}: '
            'model context window exceeded; terminating gracefully.'
        )
        if logger.isEnabledFor(logging.DEBUG):
            self._dbg(ctx, f'context_overflow total_messages={len(ctx.messages)}')
        self.trace.add_event(
            step=ctx.step,
            type=EventType.ERROR,
            payload={
                'source': TraceSources.LOOP,
                'message': LoopMessages.MODEL_CONTEXT_OVERFLOW,
            },
        )

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _dbg(self, ctx: AgentContext, msg: str) -> None:
        """Emit a DEBUG log line with a uniform sample/step prefix."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'[AgentLoop] sample={ctx.sample_id} step={ctx.step} {msg}')


def _extract_usage(output: ModelOutput) -> Optional[dict]:
    usage = output.usage
    if usage is None:
        return None
    return {
        'input': int(getattr(usage, 'input_tokens', 0) or 0),
        'output': int(getattr(usage, 'output_tokens', 0) or 0),
        'total': int(getattr(usage, 'total_tokens', 0) or 0),
    }


__all__ = ['AgentLoop']
