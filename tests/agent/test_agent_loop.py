# Copyright (c) Alibaba, Inc. and its affiliates.
"""T1 骨架 - AgentLoop 核心循环 + FunctionCallingStrategy.

Plan 覆盖点:
- async AgentLoop: generate → parse → tool_call → observe → terminate
- 每步 trace 打点 (MODEL_GENERATE / TOOL_CALL / TOOL_RESULT / SUBMIT / ERROR)
- max_steps 强制终止
- 未知工具 → ToolCallError 而非中断循环
- FunctionCallingStrategy.parse_output / is_done / tool_schema_mode
"""

import asyncio
import itertools
import unittest
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import evalscope  # noqa: F401 - trigger strategy registration
from evalscope.api.agent import (
    AgentContext,
    AgentLoop,
    AgentStrategy,
    AgentTrace,
    EventType,
    ParsedAction,
    ToolExecutionOutput,
    ToolExecutor,
)
from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ContentImage
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.registry import get_strategy
from evalscope.api.tool import ToolCall, ToolCallError
from evalscope.api.tool.tool_call import ToolFunction


def _make_output(
    content: str = '',
    tool_calls: Optional[List[ToolCall]] = None,
    stop_reason: str = 'stop',
) -> ModelOutput:
    msg = ChatMessageAssistant(content=content, tool_calls=tool_calls)
    return ModelOutput(
        model='mock',
        choices=[ChatCompletionChoice(message=msg, stop_reason=stop_reason)],
    )


def _tool_call(name: str = 'echo', args: Optional[dict] = None, call_id: str = 'c1') -> ToolCall:
    return ToolCall(id=call_id, function=ToolFunction(name=name, arguments=args or {'x': 1}))


class TestFunctionCallingStrategy(unittest.TestCase):
    """FC 策略: 无 tool_calls 即停, 有 tool_calls 则继续."""

    def setUp(self):
        self.strategy = get_strategy('function_calling')()
        self.ctx = AgentContext(
            sample_id='s',
            messages=[ChatMessageUser(content='hi')],
        )

    def test_parse_output_without_tool_calls(self):
        """No tool calls → no final_answer; model must call submit to finish."""
        parsed = self.strategy.parse_output(_make_output(content='final!'), self.ctx)
        self.assertIsNone(parsed.final_answer)
        self.assertEqual(parsed.tool_calls, [])
        self.assertFalse(self.strategy.is_done(parsed, self.ctx))

    def test_parse_output_with_tool_calls(self):
        parsed = self.strategy.parse_output(
            _make_output(content='', tool_calls=[_tool_call()]),
            self.ctx,
        )
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertIsNone(parsed.final_answer)
        self.assertFalse(self.strategy.is_done(parsed, self.ctx))

    def test_tool_schema_mode_is_function_calling(self):
        self.assertEqual(self.strategy.tool_schema_mode(), 'function_calling')

    def test_optional_submit_tool(self):
        strategy = get_strategy('function_calling')(include_submit_tool=False)
        self.assertEqual(strategy.tools(self.ctx), [])
        parsed = strategy.parse_output(
            _make_output(tool_calls=[_tool_call(name='submit', args={'answer': 'done'})]),
            self.ctx,
        )
        self.assertIsNone(parsed.final_answer)
        self.assertEqual(parsed.tool_calls[0].function.name, 'submit')

    def test_max_tool_calls_per_turn(self):
        strategy = get_strategy('function_calling')(max_tool_calls_per_turn=1)
        parsed = strategy.parse_output(
            _make_output(tool_calls=[_tool_call(call_id='one'), _tool_call(call_id='two')]),
            self.ctx,
        )
        self.assertEqual(parsed.tool_calls, [])
        self.assertEqual(parsed.error, 'Call at most 1 tool call per turn.')


class TestAgentLoopCore(unittest.TestCase):
    """AgentLoop 主循环语义."""

    def _build_loop(self, model, *, handlers=None, max_steps=5, trace=None):
        strategy = get_strategy('function_calling')()
        executor = ToolExecutor(handlers=handlers or {}, environment=None)
        return AgentLoop(
            model=model,
            strategy=strategy,
            tool_executor=executor,
            max_steps=max_steps,
            trace=trace,
        )

    def test_submit_tool_terminates_loop(self):
        """Model must call submit to finish; nudge injected when no tool used."""
        submit_call = ToolCall(id='sc1', function=ToolFunction(name='submit', arguments={'answer': '42'}))
        model = MagicMock()
        model.generate_async = AsyncMock(side_effect=[
            _make_output(content='the answer is 42'),  # no tool call → nudge
            _make_output(tool_calls=[submit_call]),     # submit → done
        ])

        loop = self._build_loop(model)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(model.generate_async.call_count, 2)
        # user + assistant(text) + nudge(user) + assistant(submit)
        self.assertEqual(len(result.messages), 4)

        types = [e.type for e in result.trace.events]
        self.assertIn(EventType.MODEL_GENERATE, types)
        self.assertIn(EventType.SUBMIT, types)
        # Nudge event present
        nudge_events = [e for e in result.trace.events if e.payload and e.payload.get('source') == 'nudge']
        self.assertEqual(len(nudge_events), 1)

    def test_tool_call_then_submit(self):
        model = MagicMock()
        # 第 1 轮发起 tool_call; 第 2 轮调用 submit
        submit_call = ToolCall(id='sc1', function=ToolFunction(name='submit', arguments={'answer': 'done'}))
        model.generate_async = AsyncMock(side_effect=[
            _make_output(tool_calls=[_tool_call(name='echo', args={'x': 7})]),
            _make_output(tool_calls=[submit_call]),
        ])

        async def echo_handler(call, env):
            return f"echoed:{call.function.arguments['x']}"

        loop = self._build_loop(model, handlers={'echo': echo_handler})
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='run echo')])
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(model.generate_async.call_count, 2)
        # user + assistant(tool_call) + tool + assistant(submit)
        self.assertEqual(len(result.messages), 4)
        tool_msg = result.messages[2]
        self.assertIsInstance(tool_msg, ChatMessageTool)
        self.assertEqual(tool_msg.content, 'echoed:7')
        self.assertIsNone(tool_msg.error)

        types = [e.type for e in result.trace.events]
        self.assertEqual(
            types,
            [
                EventType.MODEL_GENERATE,
                EventType.TOOL_CALL,
                EventType.TOOL_RESULT,
                EventType.MODEL_GENERATE,
                EventType.SUBMIT,
            ],
        )

    def test_rich_tool_output_appends_attachment_and_terminates(self):
        model = MagicMock()
        model.generate_async = AsyncMock(
            return_value=_make_output(tool_calls=[_tool_call(name='browser_action', args={'action': 'click("1")'})])
        )

        async def browser_handler(call, env):
            return ToolExecutionOutput(
                text='reward=1',
                attachments=[ContentImage(image='/tmp/step-001.png')],
                metadata={'reward': 1.0},
                terminate=True,
                final_answer='1',
            )

        loop = self._build_loop(model, handlers={'browser_action': browser_handler})
        format_observation = loop.strategy.format_observation

        def format_with_strategy_metadata(*args, **kwargs):
            message = format_observation(*args, **kwargs)
            message.metadata = {'strategy': 'preserved'}
            return message

        loop.strategy.format_observation = MagicMock(side_effect=format_with_strategy_metadata)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='click')])
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(model.generate_async.call_count, 1)
        self.assertIsInstance(result.messages[2], ChatMessageTool)
        self.assertEqual(result.messages[2].content, 'reward=1')
        self.assertEqual(result.messages[2].metadata, {'strategy': 'preserved', 'reward': 1.0})
        self.assertIsInstance(result.messages[3], ChatMessageUser)
        self.assertEqual(result.messages[3].content[0].image, '/tmp/step-001.png')
        self.assertEqual(result.messages[3].tool_call_id, ['c1'])
        tool_event = next(event for event in result.trace.events if event.type == EventType.TOOL_RESULT)
        self.assertEqual(tool_event.payload['metadata']['reward'], 1.0)
        self.assertEqual(tool_event.payload['attachments'], ['/tmp/step-001.png'])

    def test_rich_tool_attachments_follow_all_tool_results(self):
        model = MagicMock()
        submit_call = ToolCall(id='submit', function=ToolFunction(name='submit', arguments={'answer': 'done'}))
        model.generate_async = AsyncMock(side_effect=[
            _make_output(tool_calls=[
                _tool_call(name='echo', call_id='one'),
                _tool_call(name='echo', call_id='two'),
            ]),
            _make_output(tool_calls=[submit_call]),
        ])

        async def rich_handler(call, env):
            return ToolExecutionOutput(
                text=f'output:{call.id}',
                attachments=[ContentImage(image=f'/tmp/{call.id}.png')],
            )

        loop = self._build_loop(model, handlers={'echo': rich_handler})
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='run both')])
        result = asyncio.run(loop.run(ctx))

        self.assertEqual([message.role for message in result.messages[:6]], [
            'user',
            'assistant',
            'tool',
            'tool',
            'user',
            'user',
        ])
        self.assertEqual(result.messages[2].tool_call_id, 'one')
        self.assertEqual(result.messages[3].tool_call_id, 'two')
        self.assertEqual(result.messages[4].tool_call_id, ['one'])
        self.assertEqual(result.messages[5].tool_call_id, ['two'])

    def test_unknown_tool_yields_error_observation_without_aborting(self):
        model = MagicMock()
        submit_call = ToolCall(id='sc1', function=ToolFunction(name='submit', arguments={'answer': 'recovered'}))
        model.generate_async = AsyncMock(side_effect=[
            _make_output(tool_calls=[_tool_call(name='missing')]),
            _make_output(tool_calls=[submit_call]),
        ])
        loop = self._build_loop(model, handlers={})
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='go')])
        result = asyncio.run(loop.run(ctx))

        # 第三条消息是 tool 观察, 含 error
        tool_msg = result.messages[2]
        self.assertIsInstance(tool_msg, ChatMessageTool)
        self.assertIsInstance(tool_msg.error, ToolCallError)
        self.assertEqual(tool_msg.error.type, 'unknown')
        # Loop 没被打断, 第二轮成功 submit

    def test_max_steps_exhaustion_emits_error_event(self):
        model = MagicMock()
        # 每轮都返回 tool_call → 循环永远不收敛 → 触发 max_steps
        model.generate_async = AsyncMock(return_value=_make_output(tool_calls=[_tool_call(name='echo')]))

        async def echo_handler(call, env):
            return 'obs'

        loop = self._build_loop(model, handlers={'echo': echo_handler}, max_steps=2)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')], max_steps=2)
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(model.generate_async.call_count, 2)

        # 最末事件应为 ERROR + max_steps_exceeded
        last = result.trace.events[-1]
        self.assertEqual(last.type, EventType.ERROR)
        self.assertEqual(last.payload.get('message'), 'max_steps_exceeded')

    def test_system_prompt_injected_once(self):
        # 使用自定义策略返回 system prompt
        class _SysStrategy:
            name = 'sys'

            def build_system_prompt(self, ctx):
                return 'SYSTEM_PROMPT_X'

            def prepare_messages(self, ctx):
                return ctx.messages

            def parse_output(self, output, ctx):
                return ParsedAction(final_answer=output.choices[0].message.content)

            def is_done(self, parsed, ctx):
                return True

            def tool_schema_mode(self):
                return 'none'

            def tools(self, ctx):
                return []

        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_make_output(content='ok'))
        executor = ToolExecutor(handlers={}, environment=None)
        loop = AgentLoop(
            model=model,
            strategy=_SysStrategy(),
            tool_executor=executor,
            max_steps=1,
        )
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(result.messages[0].role, 'system')
        self.assertEqual(result.messages[0].content, 'SYSTEM_PROMPT_X')


class _AlwaysTextStrategy(AgentStrategy):
    """Minimal strategy stub that never yields tool calls.

    Inherits ``should_nudge`` / ``nudge_message`` from :class:`AgentStrategy`
    rather than restating them, so these tests exercise the real defaults
    instead of a copy that could drift from them.
    """

    name: str = 'always_text'

    def __init__(self, max_nudges: int) -> None:
        self.max_nudges = max_nudges

    def build_system_prompt(self, ctx: AgentContext) -> None:
        return None

    def prepare_messages(self, ctx: AgentContext) -> List[ChatMessage]:
        return ctx.messages

    def parse_output(self, output: ModelOutput, ctx: AgentContext) -> ParsedAction:
        return ParsedAction(raw_text=output.choices[0].message.content)

    def is_done(self, parsed: ParsedAction, ctx: AgentContext) -> bool:
        return parsed.final_answer is not None

    def tool_schema_mode(self) -> str:
        return 'none'

    def tools(self, ctx: AgentContext) -> list:
        return []

    def format_observation(self, call, observation, error, parsed, ctx) -> ChatMessageTool:
        return ChatMessageTool(content=str(observation), tool_call_id=call.id, function=call.function.name)


#: Per-strategy nudge budgets: (idle, malformed).  Pinned here rather than read
#: back from the strategy, so widening a budget fails a test.
EXPECTED_NUDGE_BUDGETS = {
    'function_calling': (2, 3),
    'react': (2, 3),
    'swe_bench_toolcall': (1, 3),
    'swe_bench_backticks': (2, 3),
}


class TestAgentLoopNudgeBudget(unittest.TestCase):
    """The loop owns the nudge count; a stuck model stops after max_nudges.

    Regression for #1577: the SWE-bench strategy counted a reminder string that
    the loop never injected, so ``should_nudge`` always saw 0 and nudged on
    every no-tool turn up to ``max_steps`` (250 for SWE-bench).
    """

    def _run_all_text(self, strategy, *, max_steps=20):
        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_make_output(content='thinking, no tool'))
        executor = ToolExecutor(handlers={}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=max_steps)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))
        return model, ctx, result

    def test_every_registered_strategy_has_a_reviewed_budget(self):
        from evalscope.api.registry import STRATEGY_REGISTRY

        self.assertEqual(set(STRATEGY_REGISTRY.list_keys()), set(EXPECTED_NUDGE_BUDGETS))

    def test_registered_strategies_stop_after_their_budget(self):
        # Parametrized over the live registry so a new strategy is covered
        # automatically. On the pre-fix code swe_bench_* would reach max_steps.
        from evalscope.api.registry import STRATEGY_REGISTRY

        for name in STRATEGY_REGISTRY.list_keys():
            with self.subTest(strategy=name):
                strategy = get_strategy(name)()
                budget, error_budget = EXPECTED_NUDGE_BUDGETS[name]
                self.assertEqual(strategy.max_nudges, budget)
                self.assertEqual(strategy.max_parse_error_nudges, error_budget)
                model, ctx, result = self._run_all_text(strategy)

                # One initial generate plus exactly ``budget`` retries.
                self.assertEqual(model.generate_async.call_count, budget + 1)
                self.assertEqual(ctx.nudge_count, budget)
                # An all-text run is IDLE throughout, so the malformed budget
                # must be untouched.
                self.assertEqual(ctx.parse_error_nudge_count, 0)

                nudge_events = [e for e in result.trace.events
                                if e.payload and e.payload.get('source') == 'nudge']
                # Invariant: the loop-owned counter equals the observable nudges.
                self.assertEqual(len(nudge_events), budget)
                # An all-text turn carries no parse error, so the trace must
                # tag these as the no-tool-call variant.
                self.assertEqual(
                    {e.payload.get('message') for e in nudge_events},
                    {'no_tool_call_reminder'},
                )

                submit = [e for e in result.trace.events if e.type == EventType.SUBMIT]
                self.assertEqual(len(submit), 1)
                self.assertEqual(submit[0].payload.get('source'), 'implicit_no_nudge')

    def test_tool_call_resets_the_nudge_streak(self):
        # P3 semantics: nudges bound a *consecutive* silent streak. A tool call
        # mid-episode resets the counter, so a later stray text turn does not
        # inherit earlier nudges. Sequence: text, echo, text, text, text.
        strategy = _AlwaysTextStrategy(max_nudges=2)
        strategy.parse_output = MagicMock(side_effect=[
            ParsedAction(raw_text='t1'),
            ParsedAction(tool_calls=[_tool_call(name='echo')]),
            ParsedAction(raw_text='t3'),
            ParsedAction(raw_text='t4'),
            ParsedAction(raw_text='t5'),
        ])

        async def echo_handler(call, env):
            return 'echoed'

        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_make_output(content='x'))
        executor = ToolExecutor(handlers={'echo': echo_handler}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=20)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))

        # nudge(t1) → tool(reset) → nudge(t3) → nudge(t4) → implicit submit(t5).
        # Without the reset the budget would be spent by t4 (call_count 4).
        self.assertEqual(model.generate_async.call_count, 5)
        nudge_events = [e for e in result.trace.events
                        if e.payload and e.payload.get('source') == 'nudge']
        self.assertEqual(len(nudge_events), 3)
        self.assertEqual(ctx.nudge_count, 2)


class TestAgentLoopNudgeContent(unittest.TestCase):
    """The reminder injected on a nudge reflects what the model did wrong."""

    def _nudge_messages(self, result):
        # Identify nudges by the trace's NUDGE message_ids rather than by
        # position: a strategy that injects a system prompt shifts the initial
        # user message, and positional filtering would report the task
        # description as a nudge. Same principle as the fix under test — read
        # the authoritative record instead of inferring from the transcript.
        nudge_ids = {
            e.message_id
            for e in result.trace.events if e.type == EventType.NUDGE and e.message_id is not None
        }
        return [m for m in result.messages if m.id in nudge_ids]

    def test_parse_error_reaches_the_model(self):
        # function_calling with a per-turn cap: two tool calls -> ParsedAction
        # with an error and no tool_calls. The model must see that error, not
        # the generic "no tool was called" text (it did call tools).
        strategy = get_strategy('function_calling')(max_tool_calls_per_turn=1)
        submit_call = ToolCall(id='s', function=ToolFunction(name='submit', arguments={'answer': 'done'}))
        model = MagicMock()
        model.generate_async = AsyncMock(side_effect=[
            _make_output(tool_calls=[_tool_call(call_id='a'), _tool_call(call_id='b')]),
            _make_output(tool_calls=[submit_call]),
        ])
        executor = ToolExecutor(handlers={}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=10)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))

        nudges = self._nudge_messages(result)
        self.assertEqual(len(nudges), 1)
        self.assertEqual(str(nudges[0].content), 'Call at most 1 tool call per turn.')

        nudge_events = [e for e in result.trace.events
                        if e.payload and e.payload.get('source') == 'nudge']
        self.assertEqual(nudge_events[0].payload.get('message'), 'parse_error_reminder')

    def test_no_submit_tool_reminder_omits_submit(self):
        # BrowserGym-style config: no submit tool. The reminder must not tell
        # the model to call a submit tool that is not exposed.
        strategy = get_strategy('function_calling')(include_submit_tool=False)
        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_make_output(content='thinking'))
        executor = ToolExecutor(handlers={}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=6)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))

        nudges = self._nudge_messages(result)
        self.assertTrue(nudges)
        self.assertNotIn('submit', str(nudges[0].content).lower())

    def test_reminder_identified_correctly_with_system_prompt(self):
        # swe_bench_toolcall always injects a system prompt, so messages[0] is
        # the system message and the task description sits at index 1. The
        # reminder must still be identified correctly (and must not be the task
        # description) — positional filtering would get this wrong.
        strategy = get_strategy('swe_bench_toolcall')()
        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_make_output(content='thinking about the repo'))
        executor = ToolExecutor(handlers={}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=6)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='Fix the failing test.')])
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(result.messages[0].role, 'system')
        nudges = self._nudge_messages(result)
        self.assertEqual(len(nudges), 1)
        self.assertIn('bash', str(nudges[0].content).lower())
        self.assertNotIn('Fix the failing test.', str(nudges[0].content))


class TestSplitNudgeBudgets(unittest.TestCase):
    """Malformed and idle turns are budgeted independently.

    ``tool_calls`` being empty is a symptom shared by two different failures,
    so one shared counter let either of them consume the other's retries.
    """

    def _loop(self, strategy, parse_results, *, handlers=None, max_steps=30):
        strategy.parse_output = MagicMock(side_effect=parse_results)
        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_make_output(content='x'))
        executor = ToolExecutor(handlers=handlers or {}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=max_steps)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))
        return model, ctx, result

    def test_malformed_streak_uses_its_own_budget(self):
        # 3 malformed retries (not 2) then the loop gives up: 1 + 3 generates.
        strategy = _AlwaysTextStrategy(max_nudges=2)
        model, ctx, _ = self._loop(strategy, [ParsedAction(error='bad format', raw_text='prose')] * 6)
        self.assertEqual(model.generate_async.call_count, 4)
        self.assertEqual(ctx.parse_error_nudge_count, 3)
        self.assertEqual(ctx.nudge_count, 0)

    def test_idle_streak_does_not_consume_the_malformed_budget(self):
        # Two malformed turns, then idle: the idle budget is still full, so the
        # run continues. A shared counter would already be exhausted (2+2 > 3).
        strategy = _AlwaysTextStrategy(max_nudges=2)
        model, ctx, _ = self._loop(
            strategy,
            [
                ParsedAction(error='bad format', raw_text='p1'),
                ParsedAction(error='bad format', raw_text='p2'),
                ParsedAction(raw_text='t3'),
                ParsedAction(raw_text='t4'),
                ParsedAction(raw_text='t5'),
            ],
        )
        # malformed, malformed, idle, idle → nudged 4 times, 5th turn is final.
        self.assertEqual(model.generate_async.call_count, 5)
        self.assertEqual(ctx.parse_error_nudge_count, 2)
        self.assertEqual(ctx.nudge_count, 2)

    def test_act_turn_resets_both_counters(self):
        async def echo_handler(call, env):
            return 'echoed'

        strategy = _AlwaysTextStrategy(max_nudges=2)
        model, ctx, _ = self._loop(
            strategy,
            [
                ParsedAction(error='bad format', raw_text='p1'),
                ParsedAction(raw_text='t2'),
                ParsedAction(tool_calls=[_tool_call(name='echo')]),
                ParsedAction(raw_text='t4'),
                ParsedAction(raw_text='t5'),
                ParsedAction(raw_text='t6'),
            ],
            handlers={'echo': echo_handler},
        )
        # After the ACT turn both streaks restart, so the idle budget of 2 is
        # available again: t4 and t5 are nudged, t6 is final.
        self.assertEqual(model.generate_async.call_count, 6)
        self.assertEqual(ctx.nudge_count, 2)
        self.assertEqual(ctx.parse_error_nudge_count, 0)


class TestTerminalSource(unittest.TestCase):
    """The SUBMIT event must say *why* the episode ended."""

    def _run(self, parse_result, *, strategy=None):
        strategy = strategy or _AlwaysTextStrategy(max_nudges=1)
        strategy.parse_output = MagicMock(return_value=parse_result)
        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_make_output(content='x'))
        executor = ToolExecutor(handlers={}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=30)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))
        submits = [e for e in result.trace.events if e.type == EventType.SUBMIT]
        self.assertEqual(len(submits), 1)
        return submits[0]

    def test_malformed_exhausted_gets_its_own_source(self):
        submit = self._run(ParsedAction(error='Call at most 1 tool call per turn.', raw_text='here you go'))
        self.assertEqual(submit.payload.get('source'), 'parse_error_exhausted')
        self.assertEqual(submit.payload.get('outcome'), 'malformed')
        # The error must travel with it: "kept breaking the protocol" is a
        # different diagnosis from "stopped calling tools".
        self.assertEqual(submit.payload.get('error'), 'Call at most 1 tool call per turn.')

    def test_idle_exhausted_keeps_implicit_no_nudge(self):
        submit = self._run(ParsedAction(raw_text='the answer is 42'))
        self.assertEqual(submit.payload.get('source'), 'implicit_no_nudge')
        self.assertEqual(submit.payload.get('outcome'), 'idle')
        self.assertIsNone(submit.payload.get('error'))

    def test_loop_does_not_publish_a_final_answer_it_does_not_own(self):
        # The reported prediction is resolved after the loop returns, by the
        # adapter hook; a competing ``final_answer`` here already disagreed
        # with it for swe_bench_backticks.
        submit = self._run(ParsedAction(raw_text='THOUGHT: the patch would be trivial'))
        self.assertNotIn('final_answer', submit.payload)
        self.assertIn('THOUGHT:', submit.payload.get('raw_text_preview', ''))


class TestNudgeIsNotMistakenForASubmission(unittest.TestCase):
    """A reminder must never be reported as the model's submission.

    ``swe_bench_backticks`` archives observations as ``ChatMessageUser``, and so
    is a nudge — a run ending on one would return the reminder as the patch.
    """

    def test_backticks_extract_ignores_a_trailing_nudge(self):
        # Alternate a valid fenced block with prose so the nudge streak never
        # exhausts the budget; max_steps then runs out on a nudge turn.
        strategy = get_strategy('swe_bench_backticks')()
        outputs = itertools.cycle([
            _make_output(content='```mswea_bash_command\nls\n```'),
            _make_output(content='THOUGHT: let me think without emitting a command'),
        ])

        async def bash_handler(call, env):
            return 'file1\nfile2'

        model = MagicMock()
        model.generate_async = AsyncMock(side_effect=lambda *a, **kw: next(outputs))
        executor = ToolExecutor(handlers={'bash': bash_handler}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=10)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='fix the bug')])
        result = asyncio.run(loop.run(ctx))

        # Precondition: the run really did end on a nudge, otherwise this test
        # would pass without exercising the guard.
        nudge_ids = {e.message_id for e in result.trace.events if e.type == EventType.NUDGE}
        self.assertIn(result.messages[-1].id, nudge_ids)

        # The sentinel never fired, so there is no submission to recover.
        answer = strategy.extract_final_answer(result)
        self.assertEqual(answer, '')
        self.assertNotIn('mswea_bash_command', answer)


if __name__ == '__main__':
    unittest.main()
