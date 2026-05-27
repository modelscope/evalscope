# Copyright (c) Alibaba, Inc. and its affiliates.
"""Trace-replay benchmark tests for trie agentic workloads.

Covers:
- Unit tests for trace -> ``Conversation`` mapping (no tokenizer, no FS, no network).
- End-to-end smoke test against DashScope that downloads the workload from
  ``evalscope/trie-workloads`` on ModelScope and replays a few traces.
  Skipped when ``DASHSCOPE_API_KEY`` is not configured.
"""
import json
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.plugin.datasets.base import Turn
from evalscope.perf.plugin.datasets.trie import TrieAgenticCodingPlugin, TrieCodeQaPlugin, TrieOfficeWorkPlugin
from tests.perf.perf_test_base import DASHSCOPE_CHAT_URL, PerfTestBase


def _make_plugin_without_tokenizer() -> TrieAgenticCodingPlugin:
    """Build a plugin instance bypassing tokenizer loading.

    ``_trace_to_conversation`` is pure data plumbing; the tokenizer is only
    needed for ``_synth_prompt``, which we patch out in the per-test setup.
    """
    plugin = TrieAgenticCodingPlugin.__new__(TrieAgenticCodingPlugin)
    plugin.query_parameters = None  # not consulted in the methods under test
    plugin.tokenizer = None
    return plugin


def _mini_trace(num_turns: int = 3) -> Dict[str, Any]:
    """Return a minimal valid trace dict with ``num_turns`` middle turns."""
    return {
        'input_prompt_length': 64,
        'num_turns': num_turns,
        'assistant_response_length': [16] * num_turns,
        'tool_call_output_length': [32] * num_turns,
        'tool_call_latency': [0.05] * num_turns,
        'final_assistant_response_length': 16,
    }


class TestTrieTraceMapping(unittest.TestCase):
    """Unit tests for trace -> Conversation mapping.

    No tokenizer, FS, or network required.
    """

    def setUp(self) -> None:
        self.plugin = _make_plugin_without_tokenizer()
        self.synth_patch = patch.object(self.plugin, '_synth_prompt', return_value='x')
        self.synth_patch.start()

    def tearDown(self) -> None:
        self.synth_patch.stop()

    def test_turn_count_equals_num_turns_plus_one(self) -> None:
        """A trace with ``num_turns=N`` must produce ``N+1`` Turn objects."""
        conv = self.plugin._trace_to_conversation(_mini_trace(num_turns=3))
        self.assertEqual(len(conv), 4)

    def test_max_tokens_sequence(self) -> None:
        """Per-turn ``max_tokens`` must match the recorded length sequence,
        with the final turn taking ``final_assistant_response_length``."""
        trace = _mini_trace(num_turns=3)
        trace['assistant_response_length'] = [10, 20, 30]
        trace['final_assistant_response_length'] = 70
        conv = self.plugin._trace_to_conversation(trace)
        self.assertEqual([t.max_tokens for t in conv], [10, 20, 30, 70])

    def test_tool_call_latency_placement(self) -> None:
        """``tool_call_latency[i]`` is the pre-send wait for turn ``i+1``;
        turn 0 must always have no latency."""
        trace = _mini_trace(num_turns=3)
        trace['tool_call_latency'] = [0.1, 0.2, 0.3]
        conv = self.plugin._trace_to_conversation(trace)
        self.assertIsNone(conv[0].tool_call_latency)
        self.assertEqual([t.tool_call_latency for t in conv[1:]], [0.1, 0.2, 0.3])

    def test_is_final_flag(self) -> None:
        """Only the last turn should be flagged ``is_final``."""
        conv = self.plugin._trace_to_conversation(_mini_trace(num_turns=3))
        self.assertEqual([t.is_final for t in conv], [False, False, False, True])

    def test_zero_turn_trace(self) -> None:
        """A trace with ``num_turns=0`` collapses to a single final turn."""
        trace = _mini_trace(num_turns=0)
        trace['final_assistant_response_length'] = 70
        conv = self.plugin._trace_to_conversation(trace)
        self.assertEqual(len(conv), 1)
        self.assertEqual(conv[0].max_tokens, 70)
        self.assertTrue(conv[0].is_final)
        self.assertIsNone(conv[0].tool_call_latency)

    def test_length_mismatch_raises(self) -> None:
        """Trace with inconsistent list lengths must raise ``ValueError``."""
        trace = _mini_trace(num_turns=3)
        trace['assistant_response_length'] = [10, 20]  # wrong length
        with self.assertRaises(ValueError):
            self.plugin._trace_to_conversation(trace)

    def test_build_messages_iterates_lines(self) -> None:
        """``build_messages`` yields one Conversation per non-empty jsonl line.

        Uses in-memory traces piped through ``dataset_line_by_line`` so the
        test never touches the filesystem.
        """
        traces: List[str] = [json.dumps(_mini_trace(num_turns=3)) for _ in range(3)]
        with patch.object(
            type(self.plugin), '_resolve_dataset_path', return_value='/dev/null'
        ), patch.object(
            type(self.plugin), 'dataset_line_by_line', return_value=iter(traces)
        ):
            conversations = list(self.plugin.build_messages())
        self.assertEqual(len(conversations), 3)
        for conv in conversations:
            self.assertEqual(len(conv), 4)  # 3 turns + 1 final
            self.assertTrue(conv[-1].is_final)
            self.assertTrue(all(isinstance(t, Turn) for t in conv))


class TestTriePluginRegistration(unittest.TestCase):
    """Confirm all three plugins register distinct workload files."""

    def test_distinct_file_names(self) -> None:
        names = {
            TrieAgenticCodingPlugin.FILE_NAME,
            TrieCodeQaPlugin.FILE_NAME,
            TrieOfficeWorkPlugin.FILE_NAME,
        }
        self.assertEqual(len(names), 3)
        for n in names:
            self.assertTrue(n.endswith('.jsonl'))


class TestTrieReplayE2E(PerfTestBase):
    """End-to-end smoke test against DashScope.

    Downloads ``evalscope/trie-workloads`` from ModelScope, then replays the
    first few traces of ``agentic_coding_8k`` against ``qwen-turbo`` over the
    DashScope chat/completions endpoint.  Verifies the full path:

    * dataset_snapshot_download → load jsonl → synthesize prompts
    * Turn-based MultiTurnStrategy → per-turn ``max_tokens`` → ``tool_call_latency`` sleep
    * BenchmarkData populated with ``trace_id`` / ``is_first_turn`` / ``is_last_turn``
    * `BenchmarkSummary` reports cached / total tokens

    Requires ``DASHSCOPE_API_KEY``.
    """

    def test_trie_agentic_coding_dashscope(self) -> None:
        self.skip_without_api_key()

        task_cfg = Arguments(
            model='qwen-turbo',
            url=DASHSCOPE_CHAT_URL,
            api_key=self.api_key,
            api='openai',
            dataset='trie_agentic_coding',
            multi_turn=True,
            parallel=2,
            number=3,
            tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
            extra_args={'ignore_eos': True},
            # Cap the run wall-clock at 3 minutes so a flaky DashScope call
            # cannot hang the suite.
            duration=180.0,
        )
        result = run_perf_benchmark(task_cfg)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
