# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end evaluation tests for AgentLoop pipeline.

All tests run against the real qwen-plus API (DashScope).
No mocking – each test builds a TaskConfig and calls run_task().

Scenarios
---------
1. GSM8K baseline          – standard single-turn (no agent_config)
2. GSM8K agent, no tools   – function_calling strategy, pure multi-turn
3. GSM8K agent, python_exec + local env
4. GSM8K agent, bash + local env
5. GSM8K agent, python_exec + docker env
"""

import json
import unittest
from dotenv import dotenv_values, load_dotenv
from pathlib import Path

load_dotenv('.env')
env = dotenv_values('.env')

from evalscope.api.agent import AgentConfig
from evalscope.api.agent.trace import AgentTrace, EventType
from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

_API_KEY = env.get('DASHSCOPE_API_KEY')
_API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

requires_api = unittest.skipUnless(_API_KEY, 'Requires DASHSCOPE_API_KEY in .env')


def _base_cfg(**overrides) -> dict:
    """Common TaskConfig kwargs shared by all tests."""
    cfg = {
        'model': 'qwen-plus',
        'api_url': _API_URL,
        'api_key': _API_KEY,
        'eval_type': EvalType.OPENAI_API,
        'eval_batch_size': 2,
        'limit': 3,
        'generation_config': {
            'max_tokens': 2048,
            'temperature': 0.7,
            'parallel_tool_calls': True,
        },
        'debug': True,
    }
    cfg.update(overrides)
    return cfg


def _read_review_results(benchmark: str, model: str = 'qwen-plus', work_dir: str = './outputs'):
    """Read all ReviewResult dicts from the JSONL review cache."""
    reviews_dir = Path(work_dir) / 'reviews' / model
    results = []
    for jsonl_file in reviews_dir.glob(f'{benchmark}_*.jsonl'):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# 1. GSM8K baseline (no agent_config)
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KBaseline(unittest.TestCase):
    """Standard single-turn GSM8K – no agent loop, verifies baseline works."""

    def test_gsm8k_no_agent(self):
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

    def test_gsm8k_no_agent_trace_absent(self):
        """ReviewResult.agent_trace must be None for non-agent runs."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
            )
        )
        run_task(cfg)
        reviews = _read_review_results('gsm8k')
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            self.assertIsNone(r.get('agent_trace'), 'agent_trace should be absent for baseline')


# ---------------------------------------------------------------------------
# 2. GSM8K + agent, no tools (pure multi-turn chain-of-thought)
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentNoTools(unittest.TestCase):
    """function_calling strategy with no tools – model reasons step by step."""

    def test_run_succeeds_and_returns_score(self):
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=AgentConfig(
                    strategy='function_calling',
                    tools=[],
                    max_steps=3,
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

    def test_trace_populated_strategy_correct(self):
        """AgentTrace is saved; strategy field = 'function_calling', environment = None."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=AgentConfig(
                    strategy='function_calling',
                    tools=[],
                    max_steps=3,
                ),
            )
        )
        run_task(cfg)
        reviews = _read_review_results('gsm8k')
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict, f'agent_trace missing in: {r}')
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.strategy, 'function_calling')
            self.assertIsNone(trace.environment)
            # Every sample must have at least one MODEL_GENERATE event
            types = [e.type for e in trace.events]
            self.assertIn(EventType.MODEL_GENERATE, types)


# ---------------------------------------------------------------------------
# 3. GSM8K + agent, python_exec + local environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentPythonExecLocal(unittest.TestCase):
    """function_calling strategy with python_exec tool + local subprocess env."""

    def test_run_and_trace_with_python_exec(self):
        """Run succeeds; trace.environment == 'local'; TOOL_CALL paired with TOOL_RESULT."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=AgentConfig(
                    strategy='function_calling',
                    tools=['python_exec'],
                    environment='local',
                    max_steps=5,
                    extra={'system_prompt': (
                        'Always call the python_exec tool to compute the answer.'
                    )},
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k')
        self.assertGreater(len(reviews), 0)
        any_tool_used = False
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.environment, 'local')
            types = [e.type for e in trace.events]
            if EventType.TOOL_CALL in types:
                any_tool_used = True
                self.assertIn(EventType.TOOL_RESULT, types)
        logger.info(f'Any tool used: {any_tool_used}')
        # Note: we don't assert any_tool_used – model may choose not to call tool
        # but we verify the trace is structurally correct either way


# ---------------------------------------------------------------------------
# 4. GSM8K + agent, bash + local environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentBashLocal(unittest.TestCase):
    """function_calling strategy with bash tool + local subprocess env."""

    def test_run_and_trace_with_bash(self):
        """Run succeeds; trace.environment == 'local' and events are present."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=AgentConfig(
                    strategy='function_calling',
                    tools=['bash'],
                    environment='local',
                    max_steps=5,
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k')
        for r in reviews:
            trace = AgentTrace.model_validate(r['agent_trace'])
            self.assertEqual(trace.environment, 'local')
            self.assertGreater(len(trace.events), 0)


# ---------------------------------------------------------------------------
# 5. GSM8K + agent, python_exec + docker environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentPythonExecDocker(unittest.TestCase):
    """function_calling strategy with python_exec tool + Docker sandbox env."""

    def test_run_and_trace_environment_is_docker(self):
        """Run succeeds, returns score, and AgentTrace.environment == 'docker'."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=AgentConfig(
                    strategy='function_calling',
                    tools=['python_exec'],
                    environment='docker',
                    environment_extra={'image': 'python:3.11-slim', 'timeout': 60},
                    max_steps=5,
                    extra={'system_prompt': (
                        'You are a math solver. '
                        'Use the python_exec tool to verify your calculations.'
                    )},
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k')
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.environment, 'docker')
            self.assertGreater(len(trace.events), 0)
            types = [e.type for e in trace.events]
            self.assertIn(EventType.MODEL_GENERATE, types)


if __name__ == '__main__':
    unittest.main()
