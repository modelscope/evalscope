# Copyright (c) Alibaba, Inc. and its affiliates.
"""T1 骨架 - TaskConfig / TaskState / ReviewResult / Adapter 集成验证.

Plan 覆盖点:
- ``TaskConfig.agent_config`` 开关 (None / dict / NativeAgentConfig)
- ``TaskState.agent_trace`` 替换旧 ``_trajectory``
- ``ReviewResult.agent_trace`` 持久化 + 旧 ``trajectory`` 向后兼容丢弃
- ``DefaultDataAdapter._on_inference`` 根据 ``agent_config`` 自动分支
- ``AgentLoopAdapter`` 使用 benchmark 默认值，并接受显式 Native 配置覆盖
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import evalscope  # noqa: F401 - trigger strategy registration
from evalscope.agent.tools.bash import BASH_TOOL_INFO
from evalscope.api.agent import AgentLoopResult, AgentTrace, EventType, NativeAgentConfig
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.evaluator.cache import ModelResult, ReviewResult
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.config import TaskConfig
from evalscope.utils.doc_utils.readme_generator import _format_usage_section


def _mock_model_generate_final(content: str = 'ok'):
    """Return a MagicMock model whose .generate() yields an assistant with no tool_calls."""
    model = MagicMock()
    output = ModelOutput(
        model='mock',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=content))],
    )
    model.generate.return_value = output
    # AgentLoop awaits ``generate_async``; mock both so sync/agent paths share
    # the same return value.
    model.generate_async = AsyncMock(return_value=output)
    return model


def _mock_model_generate_submit(answer: str = 'ok'):
    """Return a MagicMock model whose .generate() yields a submit tool call."""
    submit_call = ToolCall(id='sc1', function=ToolFunction(name='submit', arguments={'answer': answer}))
    model = MagicMock()
    output = ModelOutput(
        model='mock',
        choices=[ChatCompletionChoice(message=ChatMessageAssistant(content='', tool_calls=[submit_call]))],
    )
    model.generate.return_value = output
    model.generate_async = AsyncMock(return_value=output)
    return model


class TestTaskConfigNativeAgentConfig(unittest.TestCase):
    """TaskConfig 增加的 agent_config 字段与 validator."""

    def test_default_is_none(self):
        cfg = TaskConfig(model='dummy')
        self.assertIsNone(cfg.agent_config)

    def test_dict_auto_coerced_to_agent_config(self):
        cfg = TaskConfig(
            model='dummy',
            agent_config={'strategy': 'function_calling', 'max_steps': 5, 'tools': []},
        )
        self.assertIsInstance(cfg.agent_config, NativeAgentConfig)
        self.assertEqual(cfg.agent_config.max_steps, 5)

    def test_accepts_agent_config_instance(self):
        inst = NativeAgentConfig(strategy='function_calling', max_steps=3)
        cfg = TaskConfig(model='dummy', agent_config=inst)
        self.assertIs(cfg.agent_config, inst)

    def test_invalid_type_raises(self):
        with self.assertRaises(Exception):
            TaskConfig(model='dummy', agent_config=123)  # type: ignore[arg-type]


class TestTaskStateAgentTrace(unittest.TestCase):
    """TaskState 的 agent_trace property/setter (取代旧 _trajectory)."""

    def _make_state(self):
        sample = Sample(id=1, input='hi', target='ok')
        return TaskState(
            model='mock',
            sample=sample,
            messages=[ChatMessageUser(content='hi')],
            output=ModelOutput(
                model='mock',
                choices=[ChatCompletionChoice(message=ChatMessageAssistant(content='ok'))],
            ),
        )

    def test_default_agent_trace_is_none(self):
        state = self._make_state()
        self.assertIsNone(state.agent_trace)

    def test_assign_and_read_back(self):
        state = self._make_state()
        trace = AgentTrace(strategy='function_calling', max_steps=3)
        trace.add_event(step=0, type=EventType.MODEL_GENERATE)
        state.agent_trace = trace
        self.assertIs(state.agent_trace, trace)
        self.assertEqual(state.agent_trace.events[0].type, EventType.MODEL_GENERATE)

    def test_legacy_trajectory_api_removed(self):
        # plan §3.2: 旧的 trajectory / add_trajectory_step 必须已移除
        state = self._make_state()
        self.assertFalse(hasattr(state, 'trajectory'))
        self.assertFalse(hasattr(state, 'add_trajectory_step'))

    def test_prediction_cache_roundtrip_preserves_agent_trace(self):
        state = self._make_state()
        trace = AgentTrace(strategy='function_calling', environment='local', max_steps=3)
        trace.add_event(step=0, type=EventType.MODEL_GENERATE)
        state.agent_trace = trace

        cached = ModelResult.from_task_state(state)
        restored = ModelResult.model_validate_json(cached.model_dump_json()).to_task_state(
            dataset=MemoryDataset([
                Sample(id=0, input='unused'),
                Sample(id=1, input='hi', target='ok'),
            ])
        )

        self.assertIsNotNone(restored.agent_trace)
        self.assertEqual(restored.agent_trace.strategy, 'function_calling')
        self.assertEqual(restored.agent_trace.events[0].type, EventType.MODEL_GENERATE)


class TestReviewResultAgentTrace(unittest.TestCase):
    """ReviewResult 持久化复用: agent_trace 字段替换了 trajectory."""

    def _sample_score(self):
        return SampleScore(score=Score(value={'acc': 1.0}), sample_id='s1')

    def test_default_agent_trace_none(self):
        rr = ReviewResult(index=0, sample_score=self._sample_score())
        self.assertIsNone(rr.agent_trace)

    def test_agent_trace_roundtrip(self):
        trace = AgentTrace(strategy='function_calling', max_steps=1)
        trace.add_event(step=0, type=EventType.SUBMIT, payload={'final_answer': 'x'})
        rr = ReviewResult(index=0, sample_score=self._sample_score(), agent_trace=trace)
        restored = ReviewResult.model_validate_json(rr.model_dump_json())
        self.assertIsNotNone(restored.agent_trace)
        self.assertEqual(restored.agent_trace.events[0].type, EventType.SUBMIT)

    def test_legacy_trajectory_field_silently_dropped(self):
        # 老缓存带 trajectory 字段; _migrate_legacy_input 应丢弃
        raw = {
            'index': 0,
            'sample_score': self._sample_score().model_dump(),
            'trajectory': [{'role': 'user', 'content': 'legacy'}],  # 旧字段
            'messages': [],
        }
        rr = ReviewResult.model_validate(raw)
        self.assertIsNone(rr.agent_trace)
        # trajectory 字段不应复活 (禁止保留)
        self.assertNotIn('trajectory', ReviewResult.model_fields)


class TestDefaultDataAdapterAgentBranch(unittest.TestCase):
    """DefaultDataAdapter._on_inference 根据 agent_config 自动分支."""

    def _make_adapter(self, task_config):
        adapter = DefaultDataAdapter.__new__(DefaultDataAdapter)
        adapter._task_config = task_config
        return adapter

    def test_no_agent_config_uses_plain_generate(self):
        from evalscope.api.evaluator import InferenceResult
        from evalscope.api.model import ModelOutput

        adapter = self._make_adapter(TaskConfig(model='dummy'))
        model = _mock_model_generate_final('plain')
        sample = Sample(input='hi')

        out = adapter._on_inference(model, sample)

        model.generate.assert_called_once()
        # 普通分支返回纯 ModelOutput, 不应是 InferenceResult
        self.assertIsInstance(out, ModelOutput)
        self.assertNotIsInstance(out, InferenceResult)
        self.assertEqual(out.choices[0].message.content, 'plain')


class TestAgentLoopAdapterOverrides(unittest.TestCase):
    """AgentLoopAdapter merges explicit Native config over benchmark defaults."""

    @staticmethod
    def _make_adapter(task_config, *, strategy_name='function_calling', max_steps=30):
        adapter = AgentLoopAdapter.__new__(AgentLoopAdapter)
        adapter._task_config = task_config
        adapter.strategy_name = strategy_name
        adapter.max_steps = max_steps
        return adapter

    def test_explicit_native_config_overrides_benchmark_defaults(self):
        cfg = TaskConfig(
            model='dummy',
            agent_config={'strategy': 'react', 'max_steps': 99},
        )
        adapter = self._make_adapter(cfg)

        model = _mock_model_generate_submit('native_agent')
        sample = Sample(input='hi')
        out = adapter._on_inference(model, sample)

        self.assertEqual(out.output.choices[0].message.content, 'native_agent')
        trace = out.trace
        self.assertEqual(trace.strategy, 'react')
        self.assertEqual(trace.max_steps, 99)

    def test_mcp_only_native_config_preserves_benchmark_defaults(self):
        cfg = TaskConfig(model='dummy', agent_config=NativeAgentConfig(mcp_servers=[]))
        adapter = self._make_adapter(cfg, strategy_name='react', max_steps=50)

        out = adapter._on_inference(_mock_model_generate_submit('answer'), Sample(input='hi'))

        self.assertEqual(out.trace.strategy, 'react')
        self.assertEqual(out.trace.max_steps, 50)

    def test_native_tools_merge_and_benchmark_handler_wins_collision(self):

        async def benchmark_bash(call, env):
            return 'benchmark'

        class BenchmarkToolAdapter(AgentLoopAdapter):

            def build_tools(self, sample):
                return {'bash': benchmark_bash}

        cfg = TaskConfig(model='dummy', agent_config=NativeAgentConfig(tools=['bash']))
        adapter = BenchmarkToolAdapter.__new__(BenchmarkToolAdapter)
        adapter._task_config = cfg
        adapter.max_steps = 30
        trace = AgentTrace(strategy='function_calling', max_steps=30)
        loop_result = AgentLoopResult(
            messages=[],
            final_output=ModelOutput.from_content(model='mock', content='answer'),
            trace=trace,
        )

        with patch('evalscope.api.agent.run_agent_loop', return_value=loop_result) as run_loop:
            adapter._on_inference(_mock_model_generate_final(), Sample(input='hi', tools=[BASH_TOOL_INFO]))

        call_args = run_loop.call_args.kwargs
        self.assertIs(call_args['handlers']['bash'], benchmark_bash)
        self.assertEqual([tool.name for tool in call_args['all_tools']], ['bash'])

    def test_native_command_timeout_defaults_bash_calls_and_tool_schema(self):
        seen_args = []

        async def benchmark_bash(call, env):
            seen_args.append(call.function.arguments)
            return 'benchmark'

        class BenchmarkToolAdapter(AgentLoopAdapter):

            def build_tools(self, sample):
                return {'bash': benchmark_bash}

        cfg = TaskConfig(model='dummy', agent_config=NativeAgentConfig(command_timeout=180))
        adapter = BenchmarkToolAdapter.__new__(BenchmarkToolAdapter)
        adapter._task_config = cfg
        adapter.max_steps = 30
        loop_result = AgentLoopResult(
            messages=[],
            final_output=ModelOutput.from_content(model='mock', content='answer'),
            trace=AgentTrace(strategy='function_calling', max_steps=30),
        )

        with patch('evalscope.api.agent.run_agent_loop', return_value=loop_result) as run_loop:
            adapter._on_inference(_mock_model_generate_final(), Sample(input='hi', tools=[BASH_TOOL_INFO]))

        call_args = run_loop.call_args.kwargs
        bash_schema = next(tool for tool in call_args['all_tools'] if tool.name == 'bash')
        self.assertEqual(bash_schema.parameters.properties['timeout'].default, 180)
        self.assertEqual(BASH_TOOL_INFO.parameters.properties['timeout'].default, 60)

        wrapped_bash = call_args['handlers']['bash']
        asyncio.run(
            wrapped_bash(ToolCall(id='1', function=ToolFunction(name='bash', arguments={'command': 'pwd'})), None)
        )
        asyncio.run(
            wrapped_bash(
                ToolCall(id='2', function=ToolFunction(name='bash', arguments={
                    'command': 'pwd',
                    'timeout': 5,
                })),
                None,
            )
        )

        self.assertEqual(seen_args[0]['timeout'], 180)
        self.assertEqual(seen_args[1]['timeout'], 5)

    def test_explicit_native_config_selects_custom_benchmark_strategy(self):
        cfg = TaskConfig(model='dummy', agent_config=NativeAgentConfig(strategy='swe_bench_backticks'))
        adapter = self._make_adapter(cfg, strategy_name='swe_bench_toolcall', max_steps=250)
        loop_result = AgentLoopResult(
            messages=[],
            final_output=ModelOutput.from_content(model='mock', content='answer'),
            trace=AgentTrace(strategy='swe_bench_backticks', max_steps=250),
        )

        with patch('evalscope.api.agent.run_agent_loop', return_value=loop_result) as run_loop:
            adapter._on_inference(_mock_model_generate_final(), Sample(input='hi'))

        self.assertEqual(run_loop.call_args.kwargs['strategy'].name, 'swe_bench_backticks')

    def test_build_initial_messages_handles_str_and_list(self):
        adapter = AgentLoopAdapter.__new__(AgentLoopAdapter)
        s_str = Sample(input='hello')
        msgs = adapter.build_initial_messages(s_str)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, 'user')
        self.assertEqual(msgs[0].content, 'hello')

        s_list = Sample(input=[ChatMessageUser(content='a'), ChatMessageUser(content='b')])
        msgs = adapter.build_initial_messages(s_list)
        self.assertEqual(len(msgs), 2)

    def test_default_build_tools_and_environment(self):
        adapter = AgentLoopAdapter.__new__(AgentLoopAdapter)
        self.assertEqual(adapter.build_tools(Sample(input='x')), {})
        self.assertIsNone(adapter.build_environment(Sample(input='x')))

    def test_agent_loop_usage_example_contains_native_config(self):
        usage = _format_usage_section(
            'gaia',
            agent_config={
                'strategy': 'react',
                'max_steps': 50
            },
        )

        self.assertIn('from evalscope import TaskConfig, run_task', usage)
        self.assertIn('from evalscope.api.agent import NativeAgentConfig', usage)
        self.assertIn('agent_config=NativeAgentConfig(', usage)
        self.assertIn("strategy='react'", usage)
        self.assertIn('max_steps=50', usage)
        self.assertIn('--agent-config \'{"mode":"native","strategy":"react","max_steps":50}\'', usage)
        self.assertNotIn('# agent_config=NativeAgentConfig(', usage)

    def test_agent_loop_usage_example_skips_partial_agent_config(self):
        usage = _format_usage_section('gaia', agent_config={'strategy': 'react'})

        self.assertNotIn('--agent-config', usage)
        self.assertNotIn('NativeAgentConfig', usage)


if __name__ == '__main__':
    unittest.main()
