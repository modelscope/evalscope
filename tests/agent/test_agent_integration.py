# Copyright (c) Alibaba, Inc. and its affiliates.
"""T1 骨架 - TaskConfig / TaskState / ReviewResult / Adapter 集成验证.

Plan 覆盖点:
- ``TaskConfig.agent_config`` 开关 (None / dict / NativeAgentConfig)
- ``TaskState.agent_trace`` 替换旧 ``_trajectory``
- ``ReviewResult.agent_trace`` 持久化 + 旧 ``trajectory`` 向后兼容丢弃
- ``DefaultDataAdapter._on_inference`` 根据 ``agent_config`` 自动分支
- ``AgentLoopAdapter`` 忽略全局 ``agent_config`` 用自己的 ``strategy_name``
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import evalscope  # noqa: F401 - trigger strategy registration
from evalscope.api.agent import AgentTrace, EventType, NativeAgentConfig
from evalscope.api.benchmark.adapters.agent_loop_adapter import AgentLoopAdapter
from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.evaluator.cache import ReviewResult
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model.model_output import ChatCompletionChoice, ModelOutput
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.config import TaskConfig


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
    """AgentLoopAdapter._on_inference 忽略全局 agent_config; 用 strategy_name 类属性."""

    def test_ignores_global_agent_config(self):
        # 即使全局配置切了 react (此刻未注册), AgentLoopAdapter 仍该用自己的 strategy_name=function_calling
        cfg = TaskConfig(
            model='dummy',
            agent_config={'strategy': 'react', 'max_steps': 99},  # 全局
        )
        adapter = AgentLoopAdapter.__new__(AgentLoopAdapter)
        adapter._task_config = cfg
        # AgentLoopAdapter 类默认 strategy_name = 'function_calling' / max_steps_default = 30
        adapter.max_steps = AgentLoopAdapter.max_steps_default

        model = _mock_model_generate_final('native_agent')
        sample = Sample(input='hi')
        out = adapter._on_inference(model, sample)

        self.assertEqual(out.output.choices[0].message.content, 'native_agent')
        trace = out.trace
        # 用了 AgentLoopAdapter 自己的 strategy_name, 不是全局 'react'
        self.assertEqual(trace.strategy, 'function_calling')
        self.assertEqual(trace.max_steps, AgentLoopAdapter.max_steps_default)

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


if __name__ == '__main__':
    unittest.main()
