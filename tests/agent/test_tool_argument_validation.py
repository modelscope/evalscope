"""Tool-call argument validation against the schema advertised to the model.

``NativeAgentConfig.validate_tool_arguments`` makes :class:`ToolExecutor` check
every call against the ``ToolInfo.parameters`` the model was shown before
dispatching it. A violating call is not executed; the model receives the
violation as a ``parsing`` tool error, the same path an unregistered tool
already takes, so the loop keeps going and the trace shows what happened.
"""

import asyncio
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import evalscope  # noqa: F401 - trigger strategy registration
from evalscope.agent.runner import run_native_agent
from evalscope.api.agent import AgentContext, AgentLoop, AgentLoopResult, AgentTrace, EventType, ToolExecutor
from evalscope.api.agent.runner import run_agent_loop
from evalscope.api.agent.types import NativeAgentConfig
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.registry import get_strategy
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction, ToolInfo, ToolParams, validate_tool_arguments
from evalscope.config import TaskConfig
from evalscope.utils.json_schema import JSONSchema

LOOKUP = ToolInfo(
    name='lookup',
    description='Look something up.',
    parameters=ToolParams(
        properties={'query': JSONSchema(type='string'), 'limit': JSONSchema(type='integer')},
        required=['query'],
    ),
)


def _call(args: Dict[str, Any], *, name: str = 'lookup', call_id: str = 'c1') -> ToolCall:
    return ToolCall(id=call_id, function=ToolFunction(name=name, arguments=args))


def _submit(answer: str = 'done') -> ToolCall:
    return ToolCall(id='submit', function=ToolFunction(name='submit', arguments={'answer': answer}))


def _submit_with(args: Dict[str, Any]) -> ToolCall:
    return ToolCall(id='submit', function=ToolFunction(name='submit', arguments=args))


def _output(tool_calls: Optional[List[ToolCall]] = None, content: str = '') -> ModelOutput:
    msg = ChatMessageAssistant(content=content, tool_calls=tool_calls)
    return ModelOutput(model='mock', choices=[ChatCompletionChoice(message=msg, stop_reason='stop')])


class TestValidateToolArguments(unittest.TestCase):
    """The validator judges a call against exactly the schema the model was shown."""

    def test_valid_arguments_pass(self):
        self.assertIsNone(validate_tool_arguments(_call({'query': 'x', 'limit': 3}), LOOKUP))

    def test_missing_required_property_is_reported(self):
        self.assertEqual(validate_tool_arguments(_call({'limit': 3}), LOOKUP), "'query' is a required property")

    def test_wrong_type_reports_the_property_path(self):
        violation = validate_tool_arguments(_call({'query': 'x', 'limit': 'abc'}), LOOKUP)
        self.assertEqual(violation, "'abc' is not of type 'integer' (at 'limit')")

    def test_unexpected_property_is_rejected(self):
        """``ToolParams`` advertises ``additionalProperties: false``, so an extra key breaks the contract too."""
        violation = validate_tool_arguments(_call({'query': 'x', 'page': 2}), LOOKUP)
        self.assertIn("'page' was unexpected", violation)

    def test_uncompilable_schema_is_treated_as_unconstrained(self):
        """A duplicated ``required`` entry violates the metaschema; the call must still go through."""
        broken = ToolInfo(
            name='broken',
            description='d',
            parameters=ToolParams(properties={'query': JSONSchema(type='string')}, required=['query', 'query']),
        )
        self.assertIsNone(validate_tool_arguments(_call({'anything': 1}, name='broken'), broken))


class TestToolExecutorValidation(unittest.TestCase):
    """A violating call is refused before the handler runs; everything else is unchanged."""

    def setUp(self):
        self.calls: List[Dict[str, Any]] = []

        async def lookup(call: ToolCall, env: Any) -> str:
            self.calls.append(call.function.arguments)
            return 'found'

        async def echo(call: ToolCall, env: Any) -> str:
            self.calls.append(call.function.arguments)
            return 'echo'

        self.handlers = {'lookup': lookup, 'echo': echo}

    def _executor(self) -> ToolExecutor:
        return ToolExecutor(handlers=self.handlers, environment=None, tool_infos=[LOOKUP], validate_arguments=True)

    def test_valid_call_is_dispatched(self):
        observation, error, _ = asyncio.run(self._executor().execute(_call({'query': 'x'})))
        self.assertIsNone(error)
        self.assertEqual(observation, 'found')
        self.assertEqual(self.calls, [{'query': 'x'}])

    def test_violating_call_is_not_dispatched(self):
        observation, error, _ = asyncio.run(self._executor().execute(_call({'limit': 3})))
        self.assertIsInstance(error, ToolCallError)
        self.assertEqual(error.type, 'parsing')
        self.assertIn("Invalid arguments for tool 'lookup'", observation)
        self.assertIn("'query' is a required property", observation)
        self.assertEqual(self.calls, [])

    def test_tool_without_schema_is_dispatched_unvalidated(self):
        observation, error, _ = asyncio.run(self._executor().execute(_call({'whatever': 1}, name='echo')))
        self.assertIsNone(error)
        self.assertEqual(observation, 'echo')

    def test_validation_is_off_by_default(self):
        executor = ToolExecutor(handlers=self.handlers, environment=None, tool_infos=[LOOKUP])
        _, error, _ = asyncio.run(executor.execute(_call({'limit': 3})))
        self.assertIsNone(error)
        self.assertEqual(self.calls, [{'limit': 3}])

    def test_unknown_tool_still_reports_unknown(self):
        _, error, _ = asyncio.run(self._executor().execute(_call({'query': 'x'}, name='missing')))
        self.assertEqual(error.type, 'unknown')


class TestLoopIntegration(unittest.TestCase):

    def test_rejected_call_reaches_the_model_and_the_trace(self):
        seen: List[Dict[str, Any]] = []

        async def lookup(call: ToolCall, env: Any) -> str:
            seen.append(call.function.arguments)
            return 'found'

        model = MagicMock()
        model.generate_async = AsyncMock(
            side_effect=[
                _output([_call({'limit': 3}, call_id='bad')]),
                _output([_call({'query': 'x'}, call_id='good')]),
                _output([_submit('x')]),
            ]
        )
        executor = ToolExecutor(
            handlers={'lookup': lookup}, environment=None, tool_infos=[LOOKUP], validate_arguments=True
        )
        loop = AgentLoop(
            model=model, strategy=get_strategy('function_calling')(), tool_executor=executor, max_steps=5
        )
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='go')], tools=[LOOKUP])

        result = asyncio.run(loop.run(ctx))

        rejected = result.messages[2]
        self.assertIsInstance(rejected, ChatMessageTool)
        self.assertEqual(rejected.error.type, 'parsing')
        self.assertIn("'query' is a required property", rejected.text)
        # The handler only ever saw the corrected call.
        self.assertEqual(seen, [{'query': 'x'}])
        tool_results = [ev for ev in result.trace.events if ev.type == EventType.TOOL_RESULT]
        self.assertEqual([ev.payload['error'] for ev in tool_results], ['parsing', None])
        self.assertEqual(result.final_output.message.tool_calls[0].function.name, 'submit')

    def test_invalid_submit_is_rejected_before_termination(self):
        cases = [
            ({}, "'answer' is a required property"),
            ({'answer': 1}, "1 is not of type 'string' (at 'answer')"),
            ({'answer': 'done', 'extra': True}, "'extra' was unexpected"),
        ]
        for arguments, expected_error in cases:
            with self.subTest(arguments=arguments):
                model = MagicMock()
                model.generate_async = AsyncMock(side_effect=[_output([_submit_with(arguments)]), _output([_submit('done')])])
                result = run_agent_loop(
                    model=model,
                    strategy=get_strategy('function_calling')(),
                    handlers={},
                    environment=None,
                    initial_messages=[ChatMessageUser(content='go')],
                    all_tools=[],
                    max_steps=3,
                    sample_id='s',
                    trace_strategy_name='function_calling',
                    trace_env_name=None,
                    validate_tool_arguments=True,
                )

                tool_message = result.messages[2]
                self.assertIsInstance(tool_message, ChatMessageTool)
                self.assertEqual(tool_message.error.type, 'parsing')
                self.assertIn(expected_error, tool_message.text)
                tool_results = [event for event in result.trace.events if event.type == EventType.TOOL_RESULT]
                self.assertEqual([event.payload['error'] for event in tool_results], ['parsing'])
                self.assertEqual(result.final_output.message.tool_calls[0].function.arguments, {'answer': 'done'})


class TestConfigPlumbing(unittest.TestCase):
    """The flag travels from NativeAgentConfig to the executor on both entry points."""

    def test_native_agent_config_defaults_to_off(self):
        self.assertFalse(NativeAgentConfig().validate_tool_arguments)

    def test_run_agent_loop_builds_a_validating_executor(self):
        built: Dict[str, Any] = {}
        real_executor = ToolExecutor

        def recording_executor(*args: Any, **kwargs: Any) -> ToolExecutor:
            built.update(kwargs)
            return real_executor(*args, **kwargs)

        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_output([_submit()]))
        with patch('evalscope.api.agent.runner.ToolExecutor', side_effect=recording_executor):
            run_agent_loop(
                model=model,
                strategy=get_strategy('function_calling')(),
                handlers={},
                environment=None,
                initial_messages=[ChatMessageUser(content='go')],
                all_tools=[LOOKUP],
                max_steps=2,
                sample_id='s',
                trace_strategy_name='function_calling',
                trace_env_name=None,
                validate_tool_arguments=True,
            )
        self.assertTrue(built['validate_arguments'])
        self.assertEqual([tool.name for tool in built['tool_infos']], ['lookup', 'submit'])

    def test_run_native_agent_forwards_the_flag(self):
        seen: Dict[str, Any] = {}

        def fake_run_agent_loop(**kwargs: Any) -> AgentLoopResult:
            seen.update(kwargs)
            return AgentLoopResult(
                messages=[ChatMessageAssistant(content='raw')],
                final_output=_output(content='raw'),
                trace=AgentTrace(strategy='fake', max_steps=1),
            )

        class FakeStrategy:

            def __init__(self, **_: Any) -> None:
                pass

        with (
            patch('evalscope.agent.runner.get_strategy', lambda name: FakeStrategy),
            patch('evalscope.agent.runner.resolve_tools', lambda tools: {}),
            patch('evalscope.agent.runner.resolve_tool_infos', lambda tools: []),
            patch('evalscope.agent.runner.run_agent_loop', fake_run_agent_loop),
        ):
            run_native_agent(
                task_config=TaskConfig(
                    datasets=['demo'],
                    agent_config=NativeAgentConfig(strategy='fake', max_steps=1, validate_tool_arguments=True),
                ),
                model=object(),
                sample=Sample(id=1, input='do work', target='', metadata={}),
                build_sandbox_config=lambda _: None,
                extract_final_answer=lambda loop_result, strategy: 'final',
            )
        self.assertTrue(seen['validate_tool_arguments'])

    def test_agent_loop_adapter_forwards_the_flag(self):
        adapter = AgentLoopAdapter.__new__(AgentLoopAdapter)
        adapter._task_config = TaskConfig(model='dummy', agent_config=NativeAgentConfig(validate_tool_arguments=True))
        adapter.max_steps = 30
        loop_result = AgentLoopResult(
            messages=[],
            final_output=_output(content='answer'),
            trace=AgentTrace(strategy='function_calling', max_steps=30),
        )
        with patch('evalscope.api.agent.run_agent_loop', return_value=loop_result) as run_loop:
            adapter._on_inference(MagicMock(), Sample(input='hi'))
        self.assertTrue(run_loop.call_args.kwargs['validate_tool_arguments'])

    def test_agent_loop_adapter_defaults_to_off_without_native_config(self):
        adapter = AgentLoopAdapter.__new__(AgentLoopAdapter)
        adapter._task_config = TaskConfig(model='dummy')
        adapter.max_steps = 30
        loop_result = AgentLoopResult(
            messages=[],
            final_output=_output(content='answer'),
            trace=AgentTrace(strategy='function_calling', max_steps=30),
        )
        with patch('evalscope.api.agent.run_agent_loop', return_value=loop_result) as run_loop:
            adapter._on_inference(MagicMock(), Sample(input='hi'))
        self.assertFalse(run_loop.call_args.kwargs['validate_tool_arguments'])
