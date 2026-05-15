# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for agent strategy parsers.

Covers:
* ReAct / FunctionCalling submit interception, tools(), system prompt.
* SWE-bench toolcall + backticks strategies after the unified termination
  refactor: sentinel detection in ``format_observation`` mutates
  ``ParsedAction.final_answer`` (no exceptions).
* The pure observation parser in
  ``evalscope.agent.strategies.swe_bench._observation`` (returncode
  reverse-engineering, sentinel detection, mini-style envelope).

No API calls; all tests are deterministic.
"""

import unittest

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
# SweBenchBackticksStrategy format_observation
# ---------------------------------------------------------------------------

class TestSweBenchBackticksFormatObservation(unittest.TestCase):
    """Tests for SweBenchBackticksStrategy.format_observation."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench._observation import SUBMIT_SENTINEL
        from evalscope.agent.strategies.swe_bench.swe_bench_backticks import SweBenchBackticksStrategy
        self.strategy = SweBenchBackticksStrategy()
        self.sentinel = SUBMIT_SENTINEL

    def test_normal_observation_returns_user_message(self):
        call = _tool_call('bash', {'command': 'ls'})
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, 'file1.txt\nfile2.txt', None, parsed, ctx)
        self.assertIsInstance(msg, ChatMessageUser)
        self.assertEqual(msg.role, 'user')
        self.assertIn('file1.txt', msg.content)
        self.assertIn('<returncode>0</returncode>', msg.content)
        self.assertIn('<output>', msg.content)
        # No sentinel: parsed.final_answer must remain None.
        self.assertIsNone(parsed.final_answer)

    def test_error_observation(self):
        call = _tool_call('bash', {'command': 'ls'})
        error = ToolCallError(type='timeout', message='command timed out')
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, '', error, parsed, ctx)
        self.assertIsInstance(msg, ChatMessageUser)
        self.assertIn('<error>', msg.content)
        self.assertIn('command timed out', msg.content)
        self.assertIsNone(parsed.final_answer)

    def test_sentinel_first_line_sets_final_answer(self):
        call = _tool_call('bash', {'command': 'cat patch.txt'})
        observation = f'{self.sentinel}\ndiff --git a/foo.py b/foo.py\n+fix'
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, observation, None, parsed, ctx)
        # parsed mutated in place — single termination signal.
        self.assertIsNotNone(parsed.final_answer)
        self.assertIn('diff --git a/foo.py', parsed.final_answer)
        self.assertNotIn(self.sentinel, parsed.final_answer)
        # Archived message carries the raw payload, NOT an XML envelope,
        # so downstream patch extraction never sees ``</output>``-style tags.
        self.assertIsInstance(msg, ChatMessageUser)
        self.assertNotIn('<returncode>', msg.content)
        self.assertNotIn('<output>', msg.content)
        self.assertEqual(ctx.metadata.get('submission_source'), 'sentinel')

    def test_sentinel_with_nonzero_exit_no_submitted(self):
        call = _tool_call('bash', {'command': 'cat patch.txt'})
        observation = f'{self.sentinel}\ndiff content\n[exit 2]'
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, observation, None, parsed, ctx)
        self.assertIsInstance(msg, ChatMessageUser)
        self.assertIn('<returncode>2</returncode>', msg.content)
        self.assertIn(self.sentinel, msg.content)
        self.assertIsNone(parsed.final_answer)


# ---------------------------------------------------------------------------
# SweBenchBackticksStrategy extract_final_answer
# ---------------------------------------------------------------------------

class TestSweBenchBackticksExtractFinalAnswer(unittest.TestCase):
    """``extract_final_answer`` recovers sentinel payload from messages.

    Backticks strategy archives the submission as a ``ChatMessageUser``
    whose content is the raw payload (no XML envelope).
    """

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_backticks import SweBenchBackticksStrategy
        self.strategy = SweBenchBackticksStrategy()

    def _make_result(self, *, messages=None):
        output = _make_output(text='done')
        return AgentLoopResult(
            messages=messages or [],
            final_output=output,
            trace=AgentTrace(),
        )

    def test_sentinel_payload_extracted_from_user_message(self):
        # Mirror what swe_bench_backticks.format_observation archives on
        # a sentinel hit: a clean user message right after the assistant
        # turn, with NO XML envelope.
        messages = [
            ChatMessageUser(content='task description'),
            ChatMessageAssistant(content='```mswea_bash_command\ncat patch.txt\n```',
                                 model='test', source='generate'),
            ChatMessageUser(content='diff --git a/foo.py\n+fix'),
        ]
        result = self._make_result(messages=messages)
        self.assertEqual(self.strategy.extract_final_answer(result), 'diff --git a/foo.py\n+fix')

    def test_no_observations_returns_empty(self):
        result = self._make_result(messages=[])
        self.assertEqual(self.strategy.extract_final_answer(result), '')

    def test_only_envelope_observation_returns_empty(self):
        # All observations are XML envelopes (no sentinel ever fired).
        messages = [
            ChatMessageAssistant(content='```mswea_bash_command\nls\n```',
                                 model='test', source='generate'),
            ChatMessageUser(content='<returncode>0</returncode>\n<output>\nfile1\n</output>'),
        ]
        result = self._make_result(messages=messages)
        self.assertEqual(self.strategy.extract_final_answer(result), '')


# ---------------------------------------------------------------------------
# SweBenchToolcallStrategy parse_output / extract
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


class TestSweBenchToolcallFormatObservation(unittest.TestCase):
    """``format_observation`` is the sentinel interception point."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench._observation import SUBMIT_SENTINEL
        from evalscope.agent.strategies.swe_bench.swe_bench_toolcall import SweBenchToolcallStrategy
        self.strategy = SweBenchToolcallStrategy()
        self.sentinel = SUBMIT_SENTINEL

    def test_normal_observation_returns_tool_message(self):
        call = _tool_call('bash', {'command': 'ls'})
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, 'file1.txt\nfile2.txt', None, parsed, ctx)
        self.assertIsInstance(msg, ChatMessageTool)
        self.assertEqual(msg.role, 'tool')
        self.assertEqual(msg.tool_call_id, 'tc-1')
        self.assertIn('<returncode>0</returncode>', msg.content)
        self.assertIn('file1.txt', msg.content)
        self.assertIsNone(parsed.final_answer)

    def test_sentinel_first_line_sets_final_answer(self):
        call = _tool_call('bash', {'command': 'cat patch.txt'})
        observation = f'{self.sentinel}\ndiff --git a/foo.py b/foo.py\n+fix'
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, observation, None, parsed, ctx)
        # parsed mutated in place — single termination signal.
        self.assertIsNotNone(parsed.final_answer)
        self.assertIn('diff --git a/foo.py', parsed.final_answer)
        self.assertNotIn(self.sentinel, parsed.final_answer)
        # Archived tool message carries the raw payload without XML envelope.
        self.assertIsInstance(msg, ChatMessageTool)
        self.assertEqual(msg.tool_call_id, 'tc-1')
        self.assertNotIn('<returncode>', msg.content)
        self.assertNotIn('<output>', msg.content)
        self.assertEqual(ctx.metadata.get('submission_source'), 'sentinel')

    def test_sentinel_in_middle_no_submitted(self):
        call = _tool_call('bash', {'command': 'echo'})
        observation = f'preamble line\n{self.sentinel}\nmore content'
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, observation, None, parsed, ctx)
        self.assertIsInstance(msg, ChatMessageTool)
        self.assertIn(self.sentinel, msg.content)
        self.assertIsNone(parsed.final_answer)

    def test_sentinel_with_nonzero_exit_no_submitted(self):
        call = _tool_call('bash', {'command': 'cat patch.txt'})
        observation = f'{self.sentinel}\ndiff content\n[exit 1]'
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, observation, None, parsed, ctx)
        self.assertIsInstance(msg, ChatMessageTool)
        self.assertIn('<returncode>1</returncode>', msg.content)
        self.assertIsNone(parsed.final_answer)

    def test_error_observation_uses_format_error_template(self):
        call = _tool_call('bash', {'command': 'ls'})
        error = ToolCallError(type='timeout', message='command timed out')
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, '', error, parsed, ctx)
        self.assertIsInstance(msg, ChatMessageTool)
        self.assertIn('<error>', msg.content)
        self.assertIn('command timed out', msg.content)
        self.assertIn('Tool call error', msg.content)
        self.assertIsNone(parsed.final_answer)

    def test_sentinel_strips_trailing_stderr(self):
        call = _tool_call('bash', {'command': 'cat patch.txt'})
        observation = (
            f'{self.sentinel}\n'
            'diff --git a/foo.py b/foo.py\n'
            '+fix\n'
            '[stderr]\n'
            'warning: trailing whitespace.'
        )
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, observation, None, parsed, ctx)
        self.assertIsNotNone(parsed.final_answer)
        self.assertIn('diff --git', parsed.final_answer)
        self.assertNotIn('[stderr]', parsed.final_answer)
        self.assertNotIn('trailing whitespace', parsed.final_answer)
        # Archived message also strips the stderr block.
        self.assertNotIn('[stderr]', msg.content)


class TestSweBenchToolcallExtractFinalAnswer(unittest.TestCase):
    """``extract_final_answer`` recovers sentinel payload from messages.

    Toolcall strategy archives the submission as a ``ChatMessageTool``
    whose content is the raw payload (no XML envelope, no error block).
    """

    def setUp(self):
        from evalscope.agent.strategies.swe_bench.swe_bench_toolcall import SweBenchToolcallStrategy
        self.strategy = SweBenchToolcallStrategy()

    def _make_result(self, *, messages=None):
        output = _make_output(text='done')
        return AgentLoopResult(
            messages=messages or [],
            final_output=output,
            trace=AgentTrace(),
        )

    def test_sentinel_payload_extracted_from_tool_message(self):
        bash_call = _tool_call('bash', {'command': 'cat patch.txt'})
        messages = [
            ChatMessageAssistant(content='', tool_calls=[bash_call], model='test', source='generate'),
            ChatMessageTool(content='diff --git a/foo.py\n+fix', tool_call_id='tc-1', function='bash'),
        ]
        result = self._make_result(messages=messages)
        self.assertEqual(self.strategy.extract_final_answer(result), 'diff --git a/foo.py\n+fix')

    def test_no_observations_returns_empty(self):
        result = self._make_result(messages=[])
        self.assertEqual(self.strategy.extract_final_answer(result), '')

    def test_envelope_observation_skipped(self):
        # Standard XML envelope (no sentinel) must not be misread as submission.
        bash_call = _tool_call('bash', {'command': 'ls'})
        messages = [
            ChatMessageAssistant(content='', tool_calls=[bash_call], model='test', source='generate'),
            ChatMessageTool(
                content='<returncode>0</returncode>\n<output>\nfile1\n</output>',
                tool_call_id='tc-1',
                function='bash',
            ),
        ]
        result = self._make_result(messages=messages)
        self.assertEqual(self.strategy.extract_final_answer(result), '')

    def test_error_observation_skipped(self):
        bash_call = _tool_call('bash', {'command': 'ls'})
        messages = [
            ChatMessageAssistant(content='', tool_calls=[bash_call], model='test', source='generate'),
            ChatMessageTool(
                content='Tool call error:\n\n<error>\nboom\n</error>',
                tool_call_id='tc-1',
                function='bash',
            ),
        ]
        result = self._make_result(messages=messages)
        self.assertEqual(self.strategy.extract_final_answer(result), '')


# ---------------------------------------------------------------------------
# _observation.py: pure parsers (returncode reverse-engineering, sentinel,
# mini-style envelope rendering).
# ---------------------------------------------------------------------------

class TestParseExecMetadata(unittest.TestCase):
    """``parse_exec_metadata`` reverses bash tool's tail markers."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench._observation import parse_exec_metadata
        self.parse = parse_exec_metadata

    def test_exit_marker_extracts_returncode(self):
        rc, timed_out, body = self.parse('hello\n[exit 1]')
        self.assertEqual(rc, 1)
        self.assertFalse(timed_out)
        self.assertEqual(body, 'hello')

    def test_negative_exit_marker(self):
        rc, _, body = self.parse('crashed\n[exit -9]')
        self.assertEqual(rc, -9)
        self.assertEqual(body, 'crashed')

    def test_no_marker_assumes_success(self):
        rc, timed_out, body = self.parse('plain output')
        self.assertEqual(rc, 0)
        self.assertFalse(timed_out)
        self.assertEqual(body, 'plain output')

    def test_no_output_literal(self):
        rc, _, body = self.parse('(no output)')
        self.assertEqual(rc, 0)
        self.assertEqual(body, '')

    def test_timeout_marker(self):
        rc, timed_out, body = self.parse('partial\n[TIMEOUT]')
        self.assertTrue(timed_out)
        self.assertEqual(rc, 124)
        self.assertEqual(body, 'partial')


class TestCheckSentinel(unittest.TestCase):
    """``check_sentinel`` mirrors ``DockerEnvironment._check_finished``."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench._observation import SUBMIT_SENTINEL, check_sentinel
        self.sentinel = SUBMIT_SENTINEL
        self.check = check_sentinel

    def test_first_line_match_returns_payload(self):
        observation = f'{self.sentinel}\ndiff --git a/foo.py'
        self.assertEqual(self.check(observation), 'diff --git a/foo.py')

    def test_returncode_nonzero_rejects(self):
        observation = f'{self.sentinel}\ndiff\n[exit 1]'
        self.assertIsNone(self.check(observation))

    def test_timeout_rejects(self):
        observation = f'{self.sentinel}\ndiff\n[TIMEOUT]'
        self.assertIsNone(self.check(observation))

    def test_sentinel_in_middle_rejected(self):
        observation = f'preamble\n{self.sentinel}\ndiff'
        self.assertIsNone(self.check(observation))

    def test_sentinel_substring_rejected(self):
        observation = f'echo {self.sentinel} done'
        self.assertIsNone(self.check(observation))

    def test_no_sentinel_returns_none(self):
        self.assertIsNone(self.check('regular output'))

    def test_empty_input_returns_none(self):
        self.assertIsNone(self.check(''))
        self.assertIsNone(self.check(None))

    def test_strips_trailing_stderr_block(self):
        observation = (
            f'{self.sentinel}\n'
            'diff body\n'
            '[stderr]\n'
            'a warning'
        )
        payload = self.check(observation)
        self.assertIsNotNone(payload)
        self.assertIn('diff body', payload)
        self.assertNotIn('[stderr]', payload)
        self.assertNotIn('a warning', payload)


class TestFormatExecObservation(unittest.TestCase):
    """``format_exec_observation`` renders mini-style envelopes."""

    def setUp(self):
        from evalscope.agent.strategies.swe_bench._observation import format_exec_observation
        self.fmt = format_exec_observation

    def test_short_success_envelope(self):
        out = self.fmt('hello world')
        self.assertIn('<returncode>0</returncode>', out)
        self.assertIn('<output>\nhello world\n</output>', out)

    def test_nonzero_exit_envelope(self):
        out = self.fmt('boom\n[exit 42]')
        self.assertIn('<returncode>42</returncode>', out)
        self.assertIn('<output>\nboom\n</output>', out)
        self.assertNotIn('[exit 42]', out)

    def test_timeout_envelope(self):
        out = self.fmt('abc\n[TIMEOUT]')
        self.assertIn('<returncode>124</returncode>', out)
        self.assertIn('<exception>Command timed out.</exception>', out)

    def test_long_output_truncates_with_head_tail(self):
        body = ('A' * 5000) + ('B' * 2000) + ('C' * 5000)
        out = self.fmt(body, max_chars=10000, head=5000, tail=5000)
        self.assertIn('<output_head>', out)
        self.assertIn('<output_tail>', out)
        self.assertIn('<elided_chars>', out)
        self.assertIn('2000 characters elided', out)

    def test_error_message_uses_error_template(self):
        out = self.fmt('', error_message='boom')
        self.assertIn('<error>', out)
        self.assertIn('boom', out)
        self.assertIn('Tool call error', out)


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
        """No tool calls -> final_answer is None, loop should continue (nudge)."""
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
        parsed = ParsedAction()
        ctx = _make_ctx()
        msg = self.strategy.format_observation(call, 'output', None, parsed, ctx)
        self.assertIsInstance(msg, ChatMessageTool)
        self.assertEqual(msg.role, 'tool')
        self.assertIsNone(parsed.final_answer)


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
    """Tests for FC strategy: no tool calls -> not done (must use submit)."""

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
