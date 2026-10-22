# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for perf-run identity metadata (alias / endpoint host) in the web archive.

Skipped automatically when Flask (service extra) is not installed.
"""
import json
import os
import tempfile
import unittest

import pytest

pytest.importorskip('flask')  # service extra not installed → skip

from evalscope.service.perf_archive import _build_identity_metadata, list_run_summaries  # noqa: E402


class TestIdentityMetadata(unittest.TestCase):

    def test_alias_and_host_port_extracted(self):
        meta = _build_identity_metadata({
            'alias': ' server-A ',
            'url': 'http://10.0.0.2:8000/v1/chat/completions',
        })
        self.assertEqual(meta['alias'], 'server-A')
        self.assertEqual(meta['api_host'], '10.0.0.2:8000')

    def test_host_without_port(self):
        meta = _build_identity_metadata({'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'})
        self.assertEqual(meta['api_host'], 'dashscope.aliyuncs.com')
        self.assertNotIn('alias', meta)

    def test_no_identity_fields(self):
        self.assertEqual(_build_identity_metadata({}), {})
        self.assertEqual(_build_identity_metadata({'url': ''}), {})

    def test_invalid_port_does_not_invent_an_endpoint(self):
        meta = _build_identity_metadata({'url': 'http://host.example:notaport/v1'})
        self.assertNotIn('api_host', meta)


class TestRunSummaryIdentity(unittest.TestCase):
    """The archive list item must carry alias/api_host so same-model runs stay distinct."""

    def _seed_run(self, root: str, args: dict) -> str:
        run_dir = os.path.join(root, '20260101_120000', 'Qwen3.8-27B', 'parallel_1_number_1')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'benchmark_args.json'), 'w', encoding='utf-8') as f:
            json.dump(args, f)
        with open(os.path.join(run_dir, 'benchmark_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'total_requests': 1,
                    'succeed_requests': 1,
                    'request_throughput': 1.0,
                    'avg_latency': 0.5,
                    'time_taken': 1.0,
                },
                f,
            )
        return root

    def test_summary_carries_alias_and_host(self):
        tmp = tempfile.mkdtemp()
        self._seed_run(
            tmp,
            {
                'model': 'Qwen3.8-27B',
                'api': 'openai',
                'dataset': 'openqa',
                'url': 'http://10.0.0.2:8000/v1/chat/completions',
                'alias': 'server-A',
            },
        )
        runs = list_run_summaries(tmp)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]['alias'], 'server-A')
        self.assertEqual(runs[0]['api_host'], '10.0.0.2:8000')

    def test_summary_without_identity_omits_keys(self):
        tmp = tempfile.mkdtemp()
        self._seed_run(tmp, {'model': 'm', 'api': 'openai', 'dataset': 'openqa'})
        runs = list_run_summaries(tmp)
        self.assertEqual(len(runs), 1)
        self.assertIsNone(runs[0].get('alias'))
        self.assertIsNone(runs[0].get('api_host'))


if __name__ == '__main__':
    unittest.main()
