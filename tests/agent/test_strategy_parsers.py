# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for agent strategy parsers.

Tests the parse_output / is_done / format_observation / extract_final_answer
logic for ReAct, SWE-bench strategies, and FunctionCallingStrategy (submit
interception) without making any API calls.
"""

import unittest
from unittest.mock import MagicMock

from evalscope.api.agent import AgentContext, AgentLoopResult, ParsedAction
from evalscope.api.agent.trace import AgentTrace
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from evalscope.api.model import ModelOutput
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction, ToolInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(text: str = '', tool_calls: list | None = None) -> ModelOutput:
    """Build a minimal ModelOutput for parse_output tests."""
    msg = ChatMessageAssistant(
        content=text,
        tool_calls=tool_calls,
        model='test',
        source='generate',
    )
    return ModelOutput(model='test', choices=[{'message': msg}])


def _make_ctx(messages: list | None = None, tools: list | None = None) -> AgentContext:
    return AgentContext(
        sample_id='test',
        messages=messages or [],
        tools=tools or [],
        max_steps=10,
    )


def _tool_call(name: str, args: dict | None = None, call_id: str = 'tc-1') -> ToolCall:
    return ToolCall(id=call_id, function=ToolFunction(name=name, arguments=args or {}))


# ---------------------------------------------------------------------------
# SweBenchBackticksStrategy parse_output
# ---------------------------------------------------------------------------

class TestSweBenchBackticksParseOutput(unittest.TestCase):
    """Tests for SweBenchBackticksStrategy.parse_output."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_backticks import SweBenchBackticksStrategy
        self.strategy = SweBenchBackticksStrategy()
        self.ctx = _make_ctx()

    def test_single_bash_block(self):
        output = _make_output(text='THOUGHT: I need to list files\n```mswea_bash_command\nls -la\n```')
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertEqual(parsed.tool_calls[0].function.name, 'bash')
        self.assertIn('ls -la', parsed.tool_calls[0].function.arguments.get('command', ''))

    def test_no_bash_block_returns_raw_text(self):
        output = _make_output(text='The answer is 42')
        parsed = self.strategy.parse_output(output, self.ctx)
        # No fenced block → raw_text set, final_answer is None (nudge path).
        self.assertIsNone(parsed.final_answer)
        self.assertEqual(parsed.raw_text, 'The answer is 42')

    def test_multiple_bash_blocks_error(self):
        output = _make_output(
            text='THOUGHT: try two\n```mswea_bash_command\necho first\n```\n```mswea_bash_command\necho second\n```'
        )
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertIsNotNone(parsed.error)
        self.assertIn('exactly one', parsed.error)

    def test_bash_block_with_multiline_command(self):
        cmd = 'for i in $(seq 1 5); do\n  echo $i\ndone'
        output = _make_output(text=f'THOUGHT: loop\n```mswea_bash_command\n{cmd}\n```')
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertIn('seq 1 5', parsed.tool_calls[0].function.arguments.get('command', ''))


# ---------------------------------------------------------------------------
# SweBenchBackticksStrategy is_done
# ---------------------------------------------------------------------------

class TestSweBenchBackticksIsDone(unittest.TestCase):
    """Tests for SweBenchBackticksStrategy.is_done (sentinel detection)."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_backticks import SweBenchBackticksStrategy
        from evalscope.agent.strategies.swe_bench.swe_bench_toolcall import SUBMIT_SENTINEL
        self.strategy = SweBenchBackticksStrategy()
        self.sentinel = SUBMIT_SENTINEL

    def test_sentinel_on_own_line_is_done(self):
        ctx = _make_ctx(messages=[
            ChatMessageUser(content=f'<output>\n{self.sentinel}\npatch content\n</output>'),
        ])
        parsed = ParsedAction(tool_calls=[_tool_call('bash')])
        self.assertTrue(self.strategy.is_done(parsed, ctx))

    def test_sentinel_substring_not_done(self):
        ctx = _make_ctx(messages=[
            ChatMessageUser(content=f'please run {self.sentinel} at the end'),
        ])
        parsed = ParsedAction(tool_calls=[_tool_call('bash')])
        self.assertFalse(self.strategy.is_done(parsed, ctx))

    def test_no_sentinel_not_done(self):
        ctx = _make_ctx(messages=[
            ChatMessageUser(content='normal output'),
        ])
        parsed = ParsedAction(tool_calls=[_tool_call('bash')])
        self.assertFalse(self.strategy.is_done(parsed, ctx))

    def test_tool_message_ignored(self):
        """Backticks uses user messages, not tool messages."""
        ctx = _make_ctx(messages=[
            ChatMessageTool(content=f'{self.sentinel}\npatch', tool_call_id='tc-1', function='bash'),
        ])
        parsed = ParsedAction(tool_calls=[_tool_call('bash')])
        self.assertFalse(self.strategy.is_done(parsed, ctx))


# ---------------------------------------------------------------------------
# SweBenchBackticksStrategy format_observation
# ---------------------------------------------------------------------------

class TestSweBenchBackticksFormatObservation(unittest.TestCase):
    """Tests for SweBenchBackticksStrategy.format_observation."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_backticks import SweBenchBackticksStrategy
        self.strategy = SweBenchBackticksStrategy()

    def test_normal_observation_returns_user_message(self):
        call = _tool_call('bash', {'command': 'ls'})
        msg = self.strategy.format_observation(call, 'file1.txt\nfile2.txt', None)
        self.assertIsInstance(msg, ChatMessageUser)
        self.assertEqual(msg.role, 'user')
        self.assertIn('file1.txt', msg.content)
        self.assertIn('<output>', msg.content)

    def test_error_observation(self):
        call = _tool_call('bash', {'command': 'ls'})
        error = ToolCallError(type='timeout', message='command timed out')
        msg = self.strategy.format_observation(call, '', error)
        self.assertIsInstance(msg, ChatMessageUser)
        self.assertIn('<error>', msg.content)


# ---------------------------------------------------------------------------
# SweBenchBackticksStrategy extract_final_answer
# ---------------------------------------------------------------------------

class TestSweBenchBackticksExtractFinalAnswer(unittest.TestCase):
    """Tests for SweBenchBackticksStrategy.extract_final_answer."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_backticks import SweBenchBackticksStrategy
        from evalscope.agent.strategies.swe_bench.swe_bench_toolcall import SUBMIT_SENTINEL
        self.strategy = SweBenchBackticksStrategy()
        self.sentinel = SUBMIT_SENTINEL

    def _make_result(self, messages):
        output = _make_output(text='done')
        return AgentLoopResult(messages=messages, final_output=output, trace=AgentTrace())

    def test_sentinel_answer_extracted(self):
        messages = [
            ChatMessageUser(content=f'{self.sentinel}\n42'),
        ]
        result = self._make_result(messages)
        answer = self.strategy.extract_final_answer(result)
        self.assertEqual(answer, '42')

    def test_no_sentinel_returns_empty(self):
        messages = [
            ChatMessageAssistant(content='I think the answer is 7', model='test', source='generate'),
        ]
        result = self._make_result(messages)
        answer = self.strategy.extract_final_answer(result)
        self.assertEqual(answer, '')

    def test_no_messages_returns_empty(self):
        result = self._make_result([])
        answer = self.strategy.extract_final_answer(result)
        self.assertEqual(answer, '')


# ---------------------------------------------------------------------------
# SweBenchToolcallStrategy parse_output / is_done / extract
# ---------------------------------------------------------------------------

class TestSweBenchToolcallParseOutput(unittest.TestCase):
    """Tests for SweBenchToolcallStrategy.parse_output."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_toolcall import SweBenchToolcallStrategy
        self.strategy = SweBenchToolcallStrategy()
        self.ctx = _make_ctx()

    def test_bash_tool_call_parsed(self):
        bash_call = _tool_call('bash', {'command': 'ls'})
        output = _make_output(text='Let me check', tool_calls=[bash_call])
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertEqual(len(parsed.tool_calls), 1)

    def test_non_bash_tool_filtered(self):
        submit_call = _tool_call('submit', {'answer': '42'})
        output = _make_output(text='', tool_calls=[submit_call])
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertEqual(len(parsed.tool_calls), 0)

    def test_no_tool_calls_raw_text(self):
        output = _make_output(text='just thinking')
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertIsNone(parsed.final_answer)
        self.assertEqual(parsed.raw_text, 'just thinking')


class TestSweBenchToolcallIsDone(unittest.TestCase):
    """Tests for SweBenchToolcallStrategy.is_done (sentinel in tool message)."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_toolcall import SUBMIT_SENTINEL, SweBenchToolcallStrategy
        self.strategy = SweBenchToolcallStrategy()
        self.sentinel = SUBMIT_SENTINEL

    def test_sentinel_in_tool_message_is_done(self):
        ctx = _make_ctx(messages=[
            ChatMessageTool(
                content=f'<output>\n{self.sentinel}\npatch_line\n</output>',
                tool_call_id='tc-1', function='bash'),
        ])
        parsed = ParsedAction(tool_calls=[_tool_call('bash')])
        self.assertTrue(self.strategy.is_done(parsed, ctx))

    def test_user_message_ignored(self):
        """Toolcall mode scans tool messages only."""
        ctx = _make_ctx(messages=[
            ChatMessageUser(content=f'{self.sentinel}\npatch'),
        ])
        parsed = ParsedAction(tool_calls=[_tool_call('bash')])
        self.assertFalse(self.strategy.is_done(parsed, ctx))

    def test_no_sentinel_not_done(self):
        ctx = _make_ctx(messages=[
            ChatMessageTool(content='normal output', tool_call_id='tc-1', function='bash'),
        ])
        parsed = ParsedAction(tool_calls=[_tool_call('bash')])
        self.assertFalse(self.strategy.is_done(parsed, ctx))

    def test_sentinel_strict_line_match(self):
        """Sentinel embedded in prose must NOT trigger completion."""
        ctx = _make_ctx(messages=[
            ChatMessageTool(
                content=f'run echo {self.sentinel} to submit',
                tool_call_id='tc-1', function='bash'),
        ])
        parsed = ParsedAction(tool_calls=[_tool_call('bash')])
        self.assertFalse(self.strategy.is_done(parsed, ctx))


class TestSweBenchToolcallExtractFinalAnswer(unittest.TestCase):
    """Tests for SweBenchToolcallStrategy.extract_final_answer."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_toolcall import SUBMIT_SENTINEL, SweBenchToolcallStrategy
        self.strategy = SweBenchToolcallStrategy()
        self.sentinel = SUBMIT_SENTINEL

    def _make_result(self, messages):
        output = _make_output(text='done')
        return AgentLoopResult(messages=messages, final_output=output, trace=AgentTrace())

    def test_patch_extracted_from_tool_message(self):
        messages = [
            ChatMessageTool(
                content=f'{self.sentinel}\ndiff --git a/foo.py\n+fix',
                tool_call_id='tc-1', function='bash'),
        ]
        result = self._make_result(messages)
        answer = self.strategy.extract_final_answer(result)
        self.assertIn('diff --git', answer)

    def test_no_sentinel_returns_empty(self):
        messages = [
            ChatMessageTool(content='just output', tool_call_id='tc-1', function='bash'),
        ]
        result = self._make_result(messages)
        self.assertEqual(self.strategy.extract_final_answer(result), '')


# ---------------------------------------------------------------------------
# ReAct parse_output + submit interception
# ---------------------------------------------------------------------------

class TestReactParseOutput(unittest.TestCase):
    """Tests for ReactStrategy.parse_output including submit interception."""

    def setUp(self):
        from evalscope.agent.strategies.react import ReactStrategy
        self.strategy = ReactStrategy()
        self.ctx = _make_ctx()

    def test_submit_interception(self):
        submit_call = _tool_call('submit', {'answer': '42'})
        output = _make_output(text='I am confident', tool_calls=[submit_call])
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertEqual(parsed.final_answer, '42')
        self.assertEqual(len(parsed.tool_calls), 0)

    def test_regular_tool_call_not_intercepted(self):
        bash_call = _tool_call('bash', {'command': 'ls'})
        output = _make_output(text='Let me check', tool_calls=[bash_call])
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertIsNone(parsed.final_answer)

    def test_no_tool_calls_not_done(self):
        """No tool calls → final_answer is None, loop should continue (nudge)."""
        output = _make_output(text='The answer is 42')
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertIsNone(parsed.final_answer)
        self.assertEqual(len(parsed.tool_calls), 0)
        self.assertFalse(self.strategy.is_done(parsed, self.ctx))


# ---------------------------------------------------------------------------
# ReAct tools() auto-injection
# ---------------------------------------------------------------------------

class TestReactToolsInjection(unittest.TestCase):
    """Tests for ReactStrategy.tools() auto-injection of submit."""

    def setUp(self):
        from evalscope.agent.strategies.react import ReactStrategy
        self.strategy = ReactStrategy()

    def test_submit_auto_injected(self):
        ctx = _make_ctx(tools=[])
        tool_list = self.strategy.tools(ctx)
        names = [t.name for t in tool_list]
        self.assertIn('submit', names)

    def test_submit_not_duplicated(self):
        from evalscope.agent.tools.submit import SUBMIT_TOOL_INFO
        ctx = _make_ctx(tools=[SUBMIT_TOOL_INFO])
        tool_list = self.strategy.tools(ctx)
        submit_count = sum(1 for t in tool_list if t.name == 'submit')
        self.assertEqual(submit_count, 1)

    def test_existing_tools_preserved(self):
        python_info = ToolInfo(name='python_exec', description='run python')
        ctx = _make_ctx(tools=[python_info])
        tool_list = self.strategy.tools(ctx)
        names = [t.name for t in tool_list]
        self.assertIn('python_exec', names)
        self.assertIn('submit', names)


# ---------------------------------------------------------------------------
# ReAct build_system_prompt
# ---------------------------------------------------------------------------

class TestReactSystemPrompt(unittest.TestCase):
    """Tests for ReactStrategy.build_system_prompt."""

    def setUp(self):
        from evalscope.agent.strategies.react import ReactStrategy
        self.strategy = ReactStrategy()

    def test_default_prompt_contains_tool_descriptions(self):
        python_info = ToolInfo(name='python_exec', description='Execute Python code')
        ctx = _make_ctx(tools=[python_info])
        prompt = self.strategy.build_system_prompt(ctx)
        self.assertIn('python_exec', prompt)
        self.assertIn('submit', prompt)  # submit is auto-injected

    def test_custom_prompt_overrides_default(self):
        strategy = self.strategy.__class__(system_prompt='Custom prompt')
        ctx = _make_ctx()
        prompt = strategy.build_system_prompt(ctx)
        self.assertEqual(prompt, 'Custom prompt')


# ---------------------------------------------------------------------------
# FunctionCallingStrategy submit interception
# ---------------------------------------------------------------------------

class TestFCSubmitInterception(unittest.TestCase):
    """Tests for FunctionCallingStrategy submit interception."""

    def setUp(self):
        from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
        self.strategy = FunctionCallingStrategy()
        self.ctx = _make_ctx()

    def test_submit_interception(self):
        submit_call = _tool_call('submit', {'answer': 'hello'})
        output = _make_output(text='Done', tool_calls=[submit_call])
        parsed = self.strategy.parse_output(output, self.ctx)
        self.assertEqual(parsed.final_answer, 'hello')
        self.assertEqual(len(parsed.tool_calls), 0)

    def test_fc_format_observation_returns_tool_message(self):
        call = _tool_call('bash', {'command': 'ls'})
        msg = self.strategy.format_observation(call, 'output', None)
        self.assertIsInstance(msg, ChatMessageTool)
        self.assertEqual(msg.role, 'tool')


# ---------------------------------------------------------------------------
# submit tool registration
# ---------------------------------------------------------------------------

class TestSubmitToolRegistration(unittest.TestCase):
    """Tests for submit tool handler registration."""

    def test_submit_registered(self):
        from evalscope.api.registry import AGENT_TOOL_INFO_REGISTRY, AGENT_TOOL_REGISTRY
        self.assertIn('submit', AGENT_TOOL_REGISTRY)
        self.assertIn('submit', AGENT_TOOL_INFO_REGISTRY)

    def test_submit_tool_info_schema(self):
        from evalscope.agent.tools.submit import SUBMIT_TOOL_INFO
        self.assertEqual(SUBMIT_TOOL_INFO.name, 'submit')
        self.assertIn('answer', SUBMIT_TOOL_INFO.parameters.properties)
        self.assertIn('answer', SUBMIT_TOOL_INFO.parameters.required)


# ---------------------------------------------------------------------------
# ReAct extract_final_answer
# ---------------------------------------------------------------------------

class TestReactExtractFinalAnswer(unittest.TestCase):
    """Tests for ReactStrategy.extract_final_answer."""

    def setUp(self):
        from evalscope.agent.strategies.react import ReactStrategy
        self.strategy = ReactStrategy()

    def _make_result(self, messages):
        # Build final_output from the last assistant message so
        # result.final_output.message.content matches expectations.
        last_asst_content = ''
        last_asst_tool_calls = None
        for msg in reversed(messages):
            if msg.role == 'assistant':
                last_asst_content = msg.content or ''
                last_asst_tool_calls = msg.tool_calls
                break
        output = _make_output(text=last_asst_content, tool_calls=last_asst_tool_calls)
        return AgentLoopResult(messages=messages, final_output=output, trace=AgentTrace())

    def test_submit_tool_call_answer_extracted(self):
        submit_call = _tool_call('submit', {'answer': '18'})
        messages = [
            ChatMessageUser(content='Solve: 16-3-4=?'),
            ChatMessageAssistant(content='', tool_calls=[submit_call], model='test', source='generate'),
        ]
        result = self._make_result(messages)
        self.assertEqual(self.strategy.extract_final_answer(result), '18')

    def test_submit_among_multiple_tool_calls(self):
        submit_call = _tool_call('submit', {'answer': '42'})
        other_call = _tool_call('python_exec', {'code': 'print(6*7)'})
        messages = [
            ChatMessageAssistant(content='', tool_calls=[other_call, submit_call], model='test', source='generate'),
        ]
        result = self._make_result(messages)
        self.assertEqual(self.strategy.extract_final_answer(result), '42')

    def test_no_submit_falls_back_to_content(self):
        messages = [
            ChatMessageAssistant(content='I think 7', model='test', source='generate'),
        ]
        result = self._make_result(messages)
        self.assertEqual(self.strategy.extract_final_answer(result), 'I think 7')


# ---------------------------------------------------------------------------
# FC extract_final_answer + no-tool-call semantics
# ---------------------------------------------------------------------------

class TestFCExtractFinalAnswer(unittest.TestCase):
    """Tests for FunctionCallingStrategy.extract_final_answer."""

    def setUp(self):
        from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
        self.strategy = FunctionCallingStrategy()

    def _make_result(self, messages):
        # Build final_output from the last assistant message so
        # result.final_output.message.content matches expectations.
        last_asst_content = ''
        last_asst_tool_calls = None
        for msg in reversed(messages):
            if msg.role == 'assistant':
                last_asst_content = msg.content or ''
                last_asst_tool_calls = msg.tool_calls
                break
        output = _make_output(text=last_asst_content, tool_calls=last_asst_tool_calls)
        return AgentLoopResult(messages=messages, final_output=output, trace=AgentTrace())

    def test_submit_tool_call_answer_extracted(self):
        submit_call = _tool_call('submit', {'answer': '99'})
        messages = [
            ChatMessageAssistant(content='', tool_calls=[submit_call], model='test', source='generate'),
        ]
        result = self._make_result(messages)
        self.assertEqual(self.strategy.extract_final_answer(result), '99')

    def test_no_submit_falls_back_to_content(self):
        messages = [
            ChatMessageAssistant(content='raw text', model='test', source='generate'),
        ]
        result = self._make_result(messages)
        self.assertEqual(self.strategy.extract_final_answer(result), 'raw text')


class TestFCNoToolCallSemantics(unittest.TestCase):
    """Tests for FC strategy: no tool calls → not done (must use submit)."""

    def setUp(self):
        from evalscope.agent.strategies.function_calling import FunctionCallingStrategy
        self.strategy = FunctionCallingStrategy()
        self.ctx = _make_ctx()

    def test_no_tool_calls_not_done(self):
        parsed = self.strategy.parse_output(_make_output(text='just text'), self.ctx)
        self.assertIsNone(parsed.final_answer)
        self.assertFalse(self.strategy.is_done(parsed, self.ctx))

    def test_submit_is_done(self):
        submit_call = _tool_call('submit', {'answer': 'yes'})
        parsed = self.strategy.parse_output(_make_output(tool_calls=[submit_call]), self.ctx)
        self.assertEqual(parsed.final_answer, 'yes')
        self.assertTrue(self.strategy.is_done(parsed, self.ctx))


if __name__ == '__main__':
    unittest.main()
