"""Guard against a model that keeps issuing the same tool call.

The loop budgets two ways of failing to make progress: a turn that produces no
tool call at all, and a turn that breaks the output protocol. A model that
repeats one valid call forever is neither -- every turn is a well-formed ACT
turn, which resets both budgets -- so it runs to ``max_steps``, executing the
same tool each time. ``NativeAgentConfig.max_repeated_tool_calls`` closes that
gap by turning an over-long streak into a malformed turn, which reuses the
existing nudge budget and terminal reason.
"""

import asyncio
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

import evalscope  # noqa: F401 - trigger strategy registration
from evalscope.agent.runner import run_native_agent
from evalscope.api.agent import AgentContext, AgentLoop, AgentLoopResult, AgentTrace, EventType, ToolExecutor
from evalscope.api.agent.constants import SubmissionSources, TraceSources
from evalscope.api.agent.runner import run_agent_loop
from evalscope.api.agent.types import NativeAgentConfig
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.registry import get_strategy
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.config import TaskConfig


def _call(name: str = 'lookup', args: Optional[Dict[str, Any]] = None, call_id: str = 'c') -> ToolCall:
    return ToolCall(id=call_id, function=ToolFunction(name=name, arguments=args if args is not None else {'q': 'x'}))


def _submit(answer: str = 'done') -> ToolCall:
    return ToolCall(id='s', function=ToolFunction(name='submit', arguments={'answer': answer}))


def _output(tool_calls: Optional[List[ToolCall]] = None, content: str = '') -> ModelOutput:
    msg = ChatMessageAssistant(content=content, tool_calls=tool_calls)
    return ModelOutput(model='mock', choices=[ChatCompletionChoice(message=msg, stop_reason='stop')])


class _Runner:
    """Drive a loop over a scripted sequence of model outputs.

    Once the script runs out the last output repeats, so a test only has to
    state what the model does rather than count how many turns the loop will
    take to give up on it.
    """

    def __init__(self, outputs: List[ModelOutput], *, max_repeated_tool_calls=None, max_steps=12):
        self.executed: List[Dict[str, Any]] = []
        self.generated = 0

        async def handler(call: ToolCall, env: Any) -> str:
            self.executed.append(call.function.arguments)
            return 'observation'

        remaining = list(outputs)

        def next_output(*_: Any, **__: Any) -> ModelOutput:
            self.generated += 1
            return remaining.pop(0) if len(remaining) > 1 else remaining[0]

        model = MagicMock()
        model.generate_async = AsyncMock(side_effect=next_output)
        self.loop = AgentLoop(
            model=model,
            strategy=get_strategy('function_calling')(),
            tool_executor=ToolExecutor(handlers={'lookup': handler, 'other': handler}, environment=None),
            max_steps=max_steps,
            max_repeated_tool_calls=max_repeated_tool_calls,
        )

    def run(self) -> AgentLoopResult:
        ctx = AgentContext(sample_id='s', messages=[ChatMessageUser(content='go')])
        return asyncio.run(self.loop.run(ctx))


def _errors(result: AgentLoopResult, source: str) -> List[Any]:
    return [ev for ev in result.trace.events if ev.type == EventType.ERROR and ev.payload.get('source') == source]


class TestStreakSemantics(unittest.TestCase):
    """What counts as a repeat, and what resets the streak."""

    def test_identical_calls_trip_the_guard_at_the_threshold(self):
        runner = _Runner([_output([_call()])], max_repeated_tool_calls=3)
        result = runner.run()

        # Two calls executed, then the third identical turn is stalled instead.
        self.assertEqual(runner.executed, [{'q': 'x'}, {'q': 'x'}])
        repeats = _errors(result, TraceSources.REPEATED_TOOL_CALLS)
        self.assertTrue(repeats)
        self.assertIn('identical call to lookup', repeats[0].payload['message'])

    def test_changing_arguments_resets_the_streak(self):
        """A poll that advances an offset is progress, not a repeat."""
        outputs = [_output([_call(args={'offset': i})]) for i in range(5)] + [_output([_submit()])]
        runner = _Runner(outputs, max_repeated_tool_calls=2)
        result = runner.run()

        self.assertEqual(runner.executed, [{'offset': i} for i in range(5)])
        self.assertEqual(_errors(result, TraceSources.REPEATED_TOOL_CALLS), [])

    def test_changing_tool_name_resets_the_streak(self):
        outputs = [
            _output([_call('lookup')]),
            _output([_call('other')]),
            _output([_call('lookup')]),
            _output([_submit()]),
        ]
        runner = _Runner(outputs, max_repeated_tool_calls=2)
        result = runner.run()

        self.assertEqual(len(runner.executed), 3)
        self.assertEqual(_errors(result, TraceSources.REPEATED_TOOL_CALLS), [])

    def test_a_turn_without_tool_calls_resets_the_streak(self):
        outputs = [
            _output([_call()]),
            _output(content='thinking out loud'),  # idle turn -> nudged
            _output([_call()]),
            _output([_submit()]),
        ]
        runner = _Runner(outputs, max_repeated_tool_calls=2)
        result = runner.run()

        self.assertEqual(len(runner.executed), 2)
        self.assertEqual(_errors(result, TraceSources.REPEATED_TOOL_CALLS), [])

    def test_call_order_within_a_turn_does_not_matter(self):
        a, b = _call('lookup', {'q': 'a'}, 'c1'), _call('other', {'q': 'b'}, 'c2')
        runner = _Runner([_output([a, b]), _output([b, a])], max_repeated_tool_calls=2)
        result = runner.run()

        self.assertTrue(_errors(result, TraceSources.REPEATED_TOOL_CALLS))


class TestLoopBehaviour(unittest.TestCase):

    def test_guarded_turn_does_not_execute_its_tools(self):
        runner = _Runner([_output([_call()])], max_repeated_tool_calls=2)
        runner.run()
        # Only the first turn of the streak reached the handler.
        self.assertEqual(runner.executed, [{'q': 'x'}])
        self.assertGreater(runner.generated, 1)

    def test_the_model_is_told_and_can_recover(self):
        outputs = [
            _output([_call(args={'q': 'x'})]),
            _output([_call(args={'q': 'x'})]),  # stalled by the guard
            _output([_call(args={'q': 'y'})]),  # corrected
            _output([_submit('recovered')]),
        ]
        runner = _Runner(outputs, max_repeated_tool_calls=2)
        result = runner.run()

        reminder = next(m for m in result.messages if isinstance(m, ChatMessageUser) and 'identical call' in m.text)
        self.assertIn('Change the arguments', reminder.text)
        self.assertEqual(runner.executed, [{'q': 'x'}, {'q': 'y'}])
        submit = [ev for ev in result.trace.events if ev.type == EventType.SUBMIT]
        self.assertEqual(submit[-1].payload.get('final_answer'), 'recovered')

    def test_a_model_that_never_changes_ends_on_the_nudge_budget_not_max_steps(self):
        runner = _Runner([_output([_call()])], max_repeated_tool_calls=2, max_steps=30)
        result = runner.run()

        submit = [ev for ev in result.trace.events if ev.type == EventType.SUBMIT]
        self.assertEqual(submit[-1].payload['source'], SubmissionSources.PARSE_ERROR_EXHAUSTED)
        self.assertEqual(_errors(result, TraceSources.LOOP), [])  # never reached max_steps
        self.assertLess(len(result.messages), 30)

    def test_disabled_by_default(self):
        runner = _Runner([_output([_call()]) for _ in range(4)] + [_output([_submit()])])
        result = runner.run()

        # All four identical calls ran; only the submit ended the episode.
        self.assertEqual(runner.executed, [{'q': 'x'}] * 4)
        self.assertEqual(_errors(result, TraceSources.REPEATED_TOOL_CALLS), [])

    def test_genuine_parse_errors_keep_their_own_source(self):
        strategy = get_strategy('function_calling')(max_tool_calls_per_turn=1)
        model = MagicMock()
        model.generate_async = AsyncMock(
            side_effect=[_output([_call('lookup', call_id='c1'), _call('other', call_id='c2')]), _output([_submit()])]
        )
        loop = AgentLoop(
            model=model,
            strategy=strategy,
            tool_executor=ToolExecutor(handlers={}, environment=None),
            max_steps=5,
            max_repeated_tool_calls=2,
        )
        result = asyncio.run(loop.run(AgentContext(sample_id='s', messages=[ChatMessageUser(content='go')])))

        self.assertTrue(_errors(result, TraceSources.PARSE))
        self.assertEqual(_errors(result, TraceSources.REPEATED_TOOL_CALLS), [])


class TestConfigPlumbing(unittest.TestCase):

    def test_defaults_to_disabled(self):
        self.assertIsNone(NativeAgentConfig().max_repeated_tool_calls)

    def test_threshold_below_two_is_rejected(self):
        """A threshold of 1 would stall the first call of every kind."""
        for value in (1, 0, -1):
            with self.assertRaises(ValidationError):
                NativeAgentConfig(max_repeated_tool_calls=value)

    def test_run_agent_loop_forwards_the_threshold(self):
        built: Dict[str, Any] = {}
        real_loop = AgentLoop

        def recording_loop(*args: Any, **kwargs: Any) -> AgentLoop:
            built.update(kwargs)
            return real_loop(*args, **kwargs)

        model = MagicMock()
        model.generate_async = AsyncMock(return_value=_output([_submit()]))
        with patch('evalscope.api.agent.runner.AgentLoop', side_effect=recording_loop):
            run_agent_loop(
                model=model,
                strategy=get_strategy('function_calling')(),
                handlers={},
                environment=None,
                initial_messages=[ChatMessageUser(content='go')],
                all_tools=[],
                max_steps=2,
                sample_id='s',
                trace_strategy_name='function_calling',
                trace_env_name=None,
                max_repeated_tool_calls=4,
            )
        self.assertEqual(built['max_repeated_tool_calls'], 4)

    def test_run_native_agent_forwards_the_threshold(self):
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
                    agent_config=NativeAgentConfig(strategy='fake', max_steps=1, max_repeated_tool_calls=5),
                ),
                model=object(),
                sample=Sample(id=1, input='do work', target='', metadata={}),
                build_sandbox_config=lambda _: None,
                extract_final_answer=lambda loop_result, strategy: 'final',
            )
        self.assertEqual(seen['max_repeated_tool_calls'], 5)

    def test_agent_loop_adapter_forwards_the_threshold(self):
        adapter = AgentLoopAdapter.__new__(AgentLoopAdapter)
        adapter._task_config = TaskConfig(model='dummy', agent_config=NativeAgentConfig(max_repeated_tool_calls=3))
        adapter.max_steps = 30
        loop_result = AgentLoopResult(
            messages=[],
            final_output=_output(content='answer'),
            trace=AgentTrace(strategy='function_calling', max_steps=30),
        )
        with patch('evalscope.api.agent.run_agent_loop', return_value=loop_result) as run_loop:
            adapter._on_inference(MagicMock(), Sample(input='hi'))
        self.assertEqual(run_loop.call_args.kwargs['max_repeated_tool_calls'], 3)

    def test_agent_loop_adapter_defaults_to_none(self):
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
        self.assertIsNone(run_loop.call_args.kwargs['max_repeated_tool_calls'])
