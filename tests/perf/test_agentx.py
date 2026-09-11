# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the AgentX scenario wrapper (no AIPerf installation needed)."""

import json
import os
import unittest
from tempfile import TemporaryDirectory

from evalscope.perf.agentx.config import (
    AGENTX_DATASET_REPOS,
    AgentxArguments,
    validate_agentx_arguments,
)
from evalscope.perf.agentx.result import load_agentx_result
from evalscope.perf.agentx.runner import build_aiperf_command, parse_profile_export


class TestBuildCommand(unittest.TestCase):
    def test_basic_command_shape(self):
        cmd = build_aiperf_command(
            'aiperf',
            url='http://localhost:8000/v1/chat/completions',
            model='deepseek-ai/DeepSeek-V4-Pro',
            variant='256k',
            parallel=8,
            duration=3600,
        )
        self.assertEqual(cmd[0:2], ['aiperf', 'profile'])
        self.assertIn('--scenario', cmd)
        self.assertEqual(cmd[cmd.index('--scenario') + 1], 'inferencex-agentx-mvp')
        # Endpoint path must be stripped; AIPerf appends it from --endpoint-type.
        self.assertEqual(cmd[cmd.index('--url') + 1], 'http://localhost:8000')
        self.assertEqual(cmd[cmd.index('--public-dataset') + 1], 'semianalysis_cc_traces_weka_062126_256k')
        self.assertEqual(cmd[cmd.index('--concurrency') + 1], '8')
        self.assertEqual(cmd[cmd.index('--benchmark-duration') + 1], '3600')
        # Scenario-locked flags are written out explicitly.
        for locked in ('--streaming', '--use-server-token-count', '--cache-bust', '--system-idle-gap-cap-seconds'):
            self.assertIn(locked, cmd)

    def test_optional_flags_omitted_by_default(self):
        cmd = build_aiperf_command('aiperf', 'http://h:1', 'm')
        for flag in ('--api-key', '--tokenizer', '--max-context-length', '--num-dataset-entries', '--random-seed'):
            self.assertNotIn(flag, cmd)

    def test_all_optional_flags_present_when_set(self):
        cmd = build_aiperf_command(
            'aiperf', 'http://h:1', 'm',
            api_key='sk-secret', tokenizer='org/tok', max_context_length=256000,
            num_dataset_entries=4, random_seed=20260707, artifact_dir='/tmp/art',
        )
        self.assertEqual(cmd[cmd.index('--api-key') + 1], 'sk-secret')
        self.assertEqual(cmd[cmd.index('--max-context-length') + 1], '256000')
        self.assertEqual(cmd[cmd.index('--num-dataset-entries') + 1], '4')
        self.assertEqual(cmd[cmd.index('--random-seed') + 1], '20260707')
        self.assertEqual(cmd[cmd.index('--artifact-dir') + 1], '/tmp/art')


class TestValidation(unittest.TestCase):
    def test_rejects_unknown_variant(self):
        with self.assertRaises(ValueError):
            AgentxArguments(agentx_variant='512k')

    def test_rejects_scheduler_conflicts(self):
        with self.assertRaises(ValueError):
            AgentxArguments(agentx_extra_args=['--fixed-schedule', 'foo'])
        with self.assertRaises(ValueError):
            AgentxArguments(agentx_extra_args=['--request-rate=100'])

    def test_requires_model_and_http_url(self):
        with self.assertRaises(ValueError):
            validate_agentx_arguments('http://x:1', '', 8, 3600)
        with self.assertRaises(ValueError):
            validate_agentx_arguments('ftp://x', 'm', 8, 3600)

    def test_rejects_sweeps(self):
        with self.assertRaises(ValueError):
            validate_agentx_arguments('http://x:1', 'm', [1, 8], 3600)
        with self.assertRaises(ValueError):
            validate_agentx_arguments('http://x:1', 'm', 8, [60, 3600])

    def test_accepts_single_element_lists(self):
        validate_agentx_arguments('http://x:1', 'm', [8], [3600])


class TestResultParsing(unittest.TestCase):
    PER_RUN = {
        'metadata': {
            'scenario': 'inferencex-agentx-mvp',
            'submission_valid': False,
            'submission_invalid_reasons': ['context_overflow_rate_exceeded'],
        },
        'request_throughput': {'avg': 12.5},
        'time_to_first_token': {'avg': 0.9, 'unit': 's'},
        'total_requests': 900,
    }

    def _write(self, root, relpath, payload):
        path = os.path.join(root, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        return path

    def test_per_run_layout_metrics_top_level(self):
        with TemporaryDirectory() as root:
            self._write(root, 'profile_export_aiperf.json', self.PER_RUN)
            parsed = parse_profile_export(os.path.join(root, 'profile_export_aiperf.json'))
            self.assertIn('request_throughput', parsed['metrics'])
            self.assertEqual(parsed['metadata']['submission_valid'], False)

    def test_aggregate_layout_metrics_nested(self):
        aggregate = {'metadata': {'submission_valid': True}, 'metrics': {'request_throughput': {'avg': 10.0}}}
        with TemporaryDirectory() as root:
            self._write(root, 'aggregate/profile_export_aiperf_aggregate.json', aggregate)
            result = load_agentx_result(
                root,
                aiperf_version='0.13.0',
                aiperf_command=['aiperf', 'profile', '--api-key', 'sk-secret'],
                dataset_variant='256k',
                random_seed=7,
            )
            self.assertEqual(result.request_throughput, 10.0)
            self.assertTrue(result.submission_valid)
            self.assertTrue(result.canonical)
            # Provenance
            self.assertEqual(result.dataset_repo, AGENTX_DATASET_REPOS['256k'])
            self.assertEqual(result.aiperf_version, '0.13.0')
            # Secret redaction
            self.assertNotIn('sk-secret', ' '.join(result.aiperf_command))
            self.assertEqual(result.aiperf_command[result.aiperf_command.index('--api-key') + 1], '***')

    def test_per_run_result_and_smoke_flag(self):
        with TemporaryDirectory() as root:
            self._write(root, 'profile_export_aiperf.json', self.PER_RUN)
            result = load_agentx_result(root, dataset_variant='full',
                                        smoke_reasons=['num_dataset_entries capped (smoke run)'])
            # s → ms conversion
            self.assertAlmostEqual(result.time_to_first_token_ms, 900.0)
            # Missing metrics stay unavailable, not zero
            self.assertIsNone(result.output_throughput)
            self.assertIsNone(result.inter_token_latency_ms)
            # Counts read from raw values
            self.assertEqual(result.total_requests, 900)
            # AIPerf invalid + wrapper smoke reason => non-canonical, explained
            self.assertFalse(result.submission_valid)
            self.assertFalse(result.canonical)
            self.assertIn('context_overflow_rate_exceeded', result.invalid_reasons)
            self.assertIn('num_dataset_entries capped (smoke run)', result.invalid_reasons)

    def test_missing_export_raises(self):
        with TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                load_agentx_result(root)


if __name__ == '__main__':
    unittest.main()
