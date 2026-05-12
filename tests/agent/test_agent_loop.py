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
from unittest.mock import MagicMock

import evalscope  # noqa: F401 - trigger strategy registration
from evalscope.api.agent import AgentContext, AgentLoop, AgentTrace, EventType, ParsedAction, ToolExecutor
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
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
        parsed = self.strategy.parse_output(_make_output(content='final!'), self.ctx)
        self.assertEqual(parsed.final_answer, 'final!')
        self.assertEqual(parsed.tool_calls, [])
        self.assertTrue(self.strategy.is_done(parsed, self.ctx))

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

    def test_single_step_terminates_when_no_tool_calls(self):
        model = MagicMock()
        model.generate.return_value = _make_output(content='the answer is 42')

        loop = self._build_loop(model)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')])
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(model.generate.call_count, 1)
        # user + assistant 两条消息
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(result.final_output.choices[0].message.content, 'the answer is 42')

        types = [e.type for e in result.trace.events]
        self.assertEqual(types, [EventType.MODEL_GENERATE, EventType.SUBMIT])

    def test_tool_call_then_final_answer(self):
        model = MagicMock()
        # 第 1 轮发起 tool_call; 第 2 轮直接给 final answer
        model.generate.side_effect = [
            _make_output(tool_calls=[_tool_call(name='echo', args={'x': 7})]),
            _make_output(content='done'),
        ]

        async def echo_handler(call, env):
            return f"echoed:{call.function.arguments['x']}"

        loop = self._build_loop(model, handlers={'echo': echo_handler})
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='run echo')])
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(model.generate.call_count, 2)
        # user + assistant(tool_call) + tool + assistant(final) 四条
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

    def test_unknown_tool_yields_error_observation_without_aborting(self):
        model = MagicMock()
        model.generate.side_effect = [
            _make_output(tool_calls=[_tool_call(name='missing')]),
            _make_output(content='recovered'),
        ]
        loop = self._build_loop(model, handlers={})
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='go')])
        result = asyncio.run(loop.run(ctx))

        # 第三条消息是 tool 观察, 含 error
        tool_msg = result.messages[2]
        self.assertIsInstance(tool_msg, ChatMessageTool)
        self.assertIsInstance(tool_msg.error, ToolCallError)
        self.assertEqual(tool_msg.error.type, 'unknown')
        # Loop 没被打断, 第二轮成功给出 final answer
        self.assertEqual(result.final_output.choices[0].message.content, 'recovered')

    def test_max_steps_exhaustion_emits_error_event(self):
        model = MagicMock()
        # 每轮都返回 tool_call → 循环永远不收敛 → 触发 max_steps
        model.generate.return_value = _make_output(tool_calls=[_tool_call(name='echo')])

        async def echo_handler(call, env):
            return 'obs'

        loop = self._build_loop(model, handlers={'echo': echo_handler}, max_steps=2)
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='q')], max_steps=2)
        result = asyncio.run(loop.run(ctx))

        self.assertEqual(model.generate.call_count, 2)

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
        model.generate.return_value = _make_output(content='ok')
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


if __name__ == '__main__':
    unittest.main()
