# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared base class and utilities for perf test modules.

All perf test files should import from this module rather than duplicating
setup / skip logic.  The module also re-exports the symbols that every test
file needs so that individual test files stay concise.
"""
import unittest
from dotenv import dotenv_values
from typing import Optional

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_env = dotenv_values('.env')
DASHSCOPE_API_KEY: Optional[str] = _env.get('DASHSCOPE_API_KEY')

# ---------------------------------------------------------------------------
# Common URL constants
# ---------------------------------------------------------------------------
DASHSCOPE_CHAT_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
DASHSCOPE_COMPLETIONS_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/completions'
DASHSCOPE_EMBEDDINGS_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'
DASHSCOPE_RERANKS_URL = 'https://dashscope.aliyuncs.com/compatible-api/v1/reranks'

LOCAL_CHAT_URL = 'http://127.0.0.1:8801/v1/chat/completions'
LOCAL_COMPLETIONS_URL = 'http://127.0.0.1:8801/v1/completions'


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------
class PerfTestBase(unittest.TestCase):
    """Base class for all perf test cases.

    Provides:
    - ``self.api_key``: the DASHSCOPE_API_KEY loaded from ``.env`` (or ``None``).
    - ``self.skip_without_api_key()``: skip the current test when the key is
      absent, which is the standard guard for remote DashScope tests.
    """

    def setUp(self) -> None:
        super().setUp()
        self.api_key: Optional[str] = DASHSCOPE_API_KEY

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def skip_without_api_key(self) -> None:
        """Skip the current test if DASHSCOPE_API_KEY is not set."""
        if not self.api_key:
            self.skipTest('DASHSCOPE_API_KEY is not set.')

    def get_api_key(self) -> Optional[str]:
        """Return the DASHSCOPE_API_KEY (may be ``None``)."""
        return self.api_key
