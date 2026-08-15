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
import unittest
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import evalscope  # noqa: F401 - trigger strategy registration
from evalscope.api.agent import (
    AgentContext,
    AgentLoop,
    AgentTrace,
    EventType,
    ParsedAction,
    ToolExecutionOutput,
    ToolExecutor,
)
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ContentImage
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


class _AlwaysTextStrategy:
    """Minimal strategy stub that never yields tool calls.

    Used to exercise the loop's nudge budget without depending on any real
    strategy's parsing rules. ``max_nudges`` is read by the default
    ``should_nudge`` logic mirrored below.
    """

    name = 'always_text'

    def __init__(self, max_nudges: int) -> None:
        self.max_nudges = max_nudges

    def build_system_prompt(self, ctx):
        return None

    def prepare_messages(self, ctx):
        return ctx.messages

    def parse_output(self, output, ctx):
        return ParsedAction(raw_text=output.choices[0].message.content)

    def is_done(self, parsed, ctx):
        return parsed.final_answer is not None

    def should_nudge(self, parsed, ctx):
        return ctx.nudge_count < self.max_nudges

    def nudge_message(self, parsed, ctx):
        return parsed.error or 'please call a tool'

    def tool_schema_mode(self):
        return 'none'

    def tools(self, ctx):
        return []

    def format_observation(self, call, observation, error, parsed, ctx):
        return ChatMessageTool(content=str(observation), tool_call_id=call.id, function=call.function.name)


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

    def test_registered_strategies_stop_after_their_budget(self):
        # Parametrized over the live registry so a new strategy is covered
        # automatically. On the pre-fix code swe_bench_* would reach max_steps.
        from evalscope.api.registry import STRATEGY_REGISTRY

        for name in STRATEGY_REGISTRY.list_keys():
            with self.subTest(strategy=name):
                strategy = get_strategy(name)()
                budget = strategy.max_nudges
                model, ctx, result = self._run_all_text(strategy)

                # One initial generate plus exactly ``budget`` retries.
                self.assertEqual(model.generate_async.call_count, budget + 1)
                self.assertEqual(ctx.nudge_count, budget)

                nudge_events = [e for e in result.trace.events
                                if e.payload and e.payload.get('source') == 'nudge']
                # Invariant: the loop-owned counter equals the observable nudges.
                self.assertEqual(len(nudge_events), budget)

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

    def test_backticks_all_text_is_bounded_not_max_steps(self):
        # Behavioral guard independent of the nudge_count field: the backticks
        # strategy used to count a reminder string ('must contain exactly one')
        # that the loop never injects, so an all-text model nudged until
        # max_steps. Assert the run is bounded well below the step budget.
        strategy = get_strategy('swe_bench_backticks')()
        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_make_output(content='just prose, no fenced block'))
        executor = ToolExecutor(handlers={}, environment=None)
        loop = AgentLoop(model=model, strategy=strategy, tool_executor=executor, max_steps=8)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='fix the bug')])
        asyncio.run(loop.run(ctx))

        # budget 2 → one initial generate + two nudges, then implicit submit.
        self.assertEqual(model.generate_async.call_count, 3)


class TestAgentLoopNudgeContent(unittest.TestCase):
    """The reminder injected on a nudge reflects what the model did wrong."""

    def _nudge_messages(self, result):
        return [m for m in result.messages if m.role == 'user' and m is not result.messages[0]]

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


if __name__ == '__main__':
    unittest.main()
