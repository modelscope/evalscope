# Copyright (c) Alibaba, Inc. and its affiliates.
"""T1 骨架 - 接口层 + Registry 统一验证.

Plan 覆盖点:
- ``api/agent/`` 接口公民一等 API 全部可导入
- Registry 合并至 ``api/registry.py``; ``evalscope.agent`` 不 re-export
- NativeAgentConfig / AgentTrace / ParsedAction / ExecResult 基础行为
"""

import importlib
import os
import unittest

# Trigger auto-registration (function_calling strategy).
import evalscope  # noqa: F401
from evalscope.api.agent import (
    AgentContext,
    AgentEnvironment,
    AgentLoop,
    AgentLoopResult,
    AgentStrategy,
    AgentTrace,
    AgentTraceEvent,
    EventType,
    ExecResult,
    NativeAgentConfig,
    ParsedAction,
    ToolExecutor,
    ToolHandler,
    ToolSchemaMode,
)
from evalscope.api.registry import (
    AGENT_TOOL_REGISTRY,
    ENVIRONMENT_REGISTRY,
    STRATEGY_REGISTRY,
    get_agent_tool,
    get_strategy,
    list_agent_tools,
    list_environments,
    list_strategies,
    register_agent_tool,
    register_environment,
    register_strategy,
    resolve_tools,
)


class TestAgentApiSurface(unittest.TestCase):
    """``api.agent.__init__`` 必须导出 T1 接口层全部类型."""

    def test_public_symbols_importable(self):
        # 仅触达一次即可确认 import 层完整;上方 import 成功即通过.
        self.assertTrue(issubclass(AgentEnvironment, object))
        self.assertTrue(callable(AgentLoop))
        self.assertTrue(callable(AgentTrace))
        self.assertIsInstance(EventType.MODEL_GENERATE, EventType)
        # ToolSchemaMode 是 Literal, 仅保证可引用.
        self.assertEqual(ToolSchemaMode.__args__, ('function_calling', 'textual_block', 'none'))


class TestRegistryUnification(unittest.TestCase):
    """Registry 统一到 api/registry.py, agent 包禁止 re-export."""

    def test_legacy_agent_registry_file_removed(self):
        # 对应 important_decision: 删除 evalscope/agent/registry.py
        path = os.path.join(os.path.dirname(importlib.import_module('evalscope.agent').__file__), 'registry.py')
        self.assertFalse(os.path.exists(path), f'legacy registry file should be removed: {path}')

    def test_legacy_agent_registry_module_not_importable(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module('evalscope.agent.registry')

    def test_agent_package_does_not_reexport(self):
        agent_pkg = importlib.import_module('evalscope.agent')
        for name in ('get_strategy', 'register_strategy', 'resolve_tools', 'STRATEGY_REGISTRY'):
            self.assertFalse(hasattr(agent_pkg, name), f'evalscope.agent must not re-export {name!r}')

    def test_builtin_strategy_registered(self):
        self.assertIn('function_calling', list_strategies())
        self.assertIn('function_calling', STRATEGY_REGISTRY)
        # 字段一致性.
        cls = get_strategy('function_calling')
        self.assertIs(STRATEGY_REGISTRY['function_calling'], cls)

    def test_register_strategy_rejects_duplicate(self):
        with self.assertRaises(ValueError):

            @register_strategy('function_calling')
            class _Dup:
                pass

    def test_register_environment_and_agent_tool(self):
        # 使用临时名字以免污染全局 registry.

        @register_environment('_unit_env')
        class _Env(AgentEnvironment):
            name = '_unit_env'

            async def exec(self, cmd, *, cwd=None, input=None, timeout=None):
                raise NotImplementedError

            async def close(self):
                pass

        try:
            self.assertIn('_unit_env', list_environments())
            self.assertIn('_unit_env', ENVIRONMENT_REGISTRY)
        finally:
            ENVIRONMENT_REGISTRY.pop('_unit_env', None)

        @register_agent_tool('_unit_tool')
        async def _tool_handler(call, env):
            return 'ok'

        try:
            self.assertIs(get_agent_tool('_unit_tool'), _tool_handler)
            self.assertIn('_unit_tool', list_agent_tools())
            self.assertIn('_unit_tool', AGENT_TOOL_REGISTRY)
        finally:
            AGENT_TOOL_REGISTRY.pop('_unit_tool', None)

    def test_resolve_tools_empty_and_missing(self):
        self.assertEqual(resolve_tools(None), {})
        self.assertEqual(resolve_tools([]), {})
        with self.assertRaises(ValueError):
            resolve_tools(['definitely_missing_tool'])


class TestAgentTypesBehavior(unittest.TestCase):
    """NativeAgentConfig / AgentTrace / ParsedAction / ExecResult 基础字段."""

    def test_agent_config_defaults(self):
        cfg = NativeAgentConfig()
        self.assertEqual(cfg.strategy, 'function_calling')
        self.assertEqual(cfg.tools, [])
        self.assertEqual(cfg.max_steps, 10)
        self.assertIsNone(cfg.environment)
        self.assertEqual(cfg.kwargs, {})

    def test_agent_config_dict_validate(self):
        cfg = NativeAgentConfig.model_validate({'strategy': 'function_calling', 'max_steps': 5})
        self.assertEqual(cfg.max_steps, 5)

    def test_parsed_action_dataclass_defaults(self):
        pa = ParsedAction()
        self.assertEqual(pa.tool_calls, [])
        self.assertIsNone(pa.final_answer)
        self.assertIsNone(pa.error)
        self.assertIsNone(pa.raw_text)

    def test_exec_result_defaults(self):
        r = ExecResult()
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout, '')
        self.assertEqual(r.stderr, '')
        self.assertFalse(r.timed_out)
        self.assertEqual(r.duration, 0.0)

    def test_agent_trace_add_event_and_step_count(self):
        trace = AgentTrace(strategy='function_calling', environment=None, max_steps=3)
        ev = trace.add_event(step=0, type=EventType.MODEL_GENERATE, latency_ms=12.5)
        self.assertIsInstance(ev, AgentTraceEvent)
        trace.add_event(step=0, type=EventType.SUBMIT, payload={'final_answer': 'x'})
        trace.add_event(step=1, type=EventType.MODEL_GENERATE)
        self.assertEqual(len(trace.events), 3)
        self.assertEqual(trace.step_count, 2)  # step 0 与 step 1 两个独立 step

    def test_agent_trace_json_roundtrip(self):
        # 为 ReviewResult 持久化设计, 必须 pydantic 可序列化.
        trace = AgentTrace(strategy='function_calling', environment=None, max_steps=1)
        trace.add_event(step=0, type=EventType.MODEL_GENERATE)
        loaded = AgentTrace.model_validate_json(trace.model_dump_json())
        self.assertEqual(loaded.events[0].type, EventType.MODEL_GENERATE)


if __name__ == '__main__':
    unittest.main()
