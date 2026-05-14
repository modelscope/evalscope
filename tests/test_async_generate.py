# Copyright (c) Alibaba, Inc. and its affiliates.
"""
End-to-end concurrency test for ModelAPI.generate_async().

This test verifies that generate_async() truly enables concurrent execution
by comparing serial vs parallel invocation times against a real LLM endpoint.

Requirements:
  - DASHSCOPE_API_KEY (or EVALSCOPE_API_KEY) environment variable must be set
  - Network access to DashScope API
"""
import asyncio
import os
import time
import unittest
from dotenv import load_dotenv

load_dotenv('.env')

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.models.openai_compatible import OpenAICompatibleAPI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = 'qwen-plus'
BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
API_KEY = os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('EVALSCOPE_API_KEY')
N = 3  # Number of calls for serial/concurrent comparison
PROMPT = "Reply with only the word 'hello'"
CONCURRENCY_RATIO = 0.7  # T_concurrent should be < T_serial * this ratio


def _build_input():
    """Build a simple chat input list."""
    return [ChatMessageUser(content=PROMPT)]


def _default_config():
    """Build a default generation config for testing."""
    return GenerateConfig(temperature=0.0, max_tokens=32)


def _validate_output(test_case: unittest.TestCase, output):
    """Assert that a ModelOutput is valid and contains a message."""
    test_case.assertIsNotNone(output, 'generate returned None')
    test_case.assertIsInstance(output, ModelOutput, f'Expected ModelOutput, got {type(output)}')
    test_case.assertFalse(output.empty, 'ModelOutput has no choices')
    test_case.assertIsNotNone(output.message, 'ModelOutput.message is None')
    test_case.assertTrue(output.message.text, 'ModelOutput.message.text is empty')


class TestGenerateAsync(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for ModelAPI.generate_async()."""

    @classmethod
    def setUpClass(cls):
        if not API_KEY:
            raise unittest.SkipTest('DASHSCOPE_API_KEY or EVALSCOPE_API_KEY not set, skipping real API test.')
        cls.model = OpenAICompatibleAPI(
            model_name=MODEL_NAME,
            base_url=BASE_URL,
            api_key=API_KEY,
            config=GenerateConfig(temperature=0.0, max_tokens=32),
        )

    async def test_generate_async_concurrency(self):
        """Verify that generate_async enables true concurrency.

        Strategy:
          1. Call generate_async N times serially, record total time T_serial.
          2. Call generate_async N times concurrently via asyncio.gather, record T_concurrent.
          3. Assert T_concurrent < T_serial * CONCURRENCY_RATIO.
        """
        input_messages = _build_input()
        config = _default_config()

        # --- Serial execution ---
        t0 = time.monotonic()
        serial_outputs = []
        for _ in range(N):
            output = await self.model.generate_async(
                input=input_messages,
                tools=[],
                tool_choice='none',
                config=config,
            )
            serial_outputs.append(output)
        t_serial = time.monotonic() - t0

        # Validate all serial outputs
        for output in serial_outputs:
            _validate_output(self, output)

        # --- Concurrent execution ---
        t0 = time.monotonic()
        concurrent_outputs = await asyncio.gather(*[
            self.model.generate_async(
                input=input_messages,
                tools=[],
                tool_choice='none',
                config=config,
            )
            for _ in range(N)
        ])
        t_concurrent = time.monotonic() - t0

        # Validate all concurrent outputs
        for output in concurrent_outputs:
            _validate_output(self, output)

        # --- Timing comparison ---
        print(f'\n{"=" * 60}')
        print(f'  Serial   ({N} calls): {t_serial:.3f}s')
        print(f'  Concurrent ({N} calls): {t_concurrent:.3f}s')
        print(f'  Speedup ratio: {t_serial / t_concurrent:.2f}x')
        print(f'  Threshold: T_concurrent < T_serial * {CONCURRENCY_RATIO}')
        print(f'  Result: {t_concurrent:.3f}s < {t_serial * CONCURRENCY_RATIO:.3f}s = {"PASS" if t_concurrent < t_serial * CONCURRENCY_RATIO else "FAIL"}')
        print(f'{"=" * 60}')

        self.assertLess(
            t_concurrent,
            t_serial * CONCURRENCY_RATIO,
            f'Concurrent execution ({t_concurrent:.3f}s) was not significantly faster '
            f'than serial ({t_serial:.3f}s). Expected < {t_serial * CONCURRENCY_RATIO:.3f}s. '
            f'This suggests generate_async is not truly concurrent.',
        )

    async def test_generate_async_output_structure(self):
        """Verify that generate_async returns the same ModelOutput structure as generate."""
        input_messages = _build_input()
        config = _default_config()

        # Async call
        async_output = await self.model.generate_async(
            input=input_messages,
            tools=[],
            tool_choice='none',
            config=config,
        )

        # Sync call
        sync_output = self.model.generate(
            input=input_messages,
            tools=[],
            tool_choice='none',
            config=config,
        )

        # Both should be valid ModelOutput instances
        _validate_output(self, async_output)
        _validate_output(self, sync_output)

        # Structural equivalence checks
        self.assertIs(
            type(async_output),
            type(sync_output),
            f'Type mismatch: async={type(async_output)}, sync={type(sync_output)}',
        )
        self.assertEqual(
            async_output.model,
            sync_output.model,
            f'Model name mismatch: async={async_output.model}, sync={sync_output.model}',
        )
        # Both should have usage info
        self.assertIsNotNone(async_output.usage, 'Async output missing usage')
        self.assertIsNotNone(sync_output.usage, 'Sync output missing usage')
        # Both should have timing info
        self.assertIsNotNone(async_output.time, 'Async output missing time')
        self.assertGreater(async_output.time, 0, 'Async output time should be > 0')
        self.assertIsNotNone(sync_output.time, 'Sync output missing time')
        self.assertGreater(sync_output.time, 0, 'Sync output time should be > 0')


if __name__ == '__main__':
    unittest.main(verbosity=2)
