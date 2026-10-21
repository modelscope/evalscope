# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the active-task registry, endpoint sanitization, task manifests and /tasks/active.

Skipped automatically when Flask (service extra) is not installed.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import pytest

flask = pytest.importorskip('flask')  # noqa: F841  (service extra not installed → skip)

from evalscope.service.utils.log import read_task_manifest, write_task_manifest  # noqa: E402
from evalscope.service.utils.process import (  # noqa: E402
    list_active_tasks,
    register_process,
    sanitize_endpoint,
    unregister_process,
)


class TestSanitizeEndpoint(unittest.TestCase):

    def test_full_url_reduces_to_host_port(self):
        self.assertEqual(sanitize_endpoint('http://10.0.0.2:8000/v1/chat/completions'), '10.0.0.2:8000')
        self.assertEqual(sanitize_endpoint('https://DashScope.aliyuncs.com/compatible-mode/v1'), 'dashscope.aliyuncs.com')

    def test_schemeless_and_credentials(self):
        self.assertEqual(sanitize_endpoint('server-a:8877/v1'), 'server-a:8877')
        # userinfo never leaks
        self.assertEqual(sanitize_endpoint('http://user:secret@host.example:9000/v1'), 'host.example:9000')

    def test_missing_or_unparseable_returns_none(self):
        self.assertIsNone(sanitize_endpoint(None))
        self.assertIsNone(sanitize_endpoint(''))
        self.assertIsNone(sanitize_endpoint('   '))
        self.assertIsNone(sanitize_endpoint('http://:bad-port/'))


class TestActiveRegistry(unittest.TestCase):

    def tearDown(self):
        for task in list_active_tasks():
            unregister_process(task['task_id'])

    def test_list_returns_metadata_without_process_handle(self):
        register_process(
            'eval_1',
            mock.Mock(),
            meta={
                'kind': 'eval',
                'model': 'Qwen3.8-27B',
                'endpoint': '10.0.0.2:8000',
                'alias': 'server-A',
            },
        )
        tasks = list_active_tasks()
        self.assertEqual(len(tasks), 1)
        record = tasks[0]
        self.assertEqual(record['task_id'], 'eval_1')
        self.assertEqual(record['kind'], 'eval')
        self.assertEqual(record['model'], 'Qwen3.8-27B')
        self.assertEqual(record['endpoint'], '10.0.0.2:8000')
        self.assertEqual(record['alias'], 'server-A')
        self.assertTrue(record['started_at'])
        self.assertNotIn('proc', record)

    def test_register_without_meta_defaults(self):
        register_process('legacy_1', mock.Mock())
        record = list_active_tasks()[0]
        self.assertEqual(record['kind'], '')
        self.assertIsNone(record['endpoint'])
        self.assertIsNone(record['alias'])


class TestTaskManifest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        patcher = mock.patch('evalscope.service.utils.log.OUTPUT_DIR', self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_roundtrip(self):
        write_task_manifest('eval_1', {'task_id': 'eval_1', 'kind': 'eval', 'alias': 'A', 'api_url': 'http://h:1/v1'})
        manifest = read_task_manifest(os.path.join(self.tmp, 'eval_1'))
        self.assertEqual(manifest['alias'], 'A')

    def test_missing_and_corrupt_return_none(self):
        self.assertIsNone(read_task_manifest(os.path.join(self.tmp, 'nope')))
        bad = os.path.join(self.tmp, 'bad')
        os.makedirs(bad)
        with open(os.path.join(bad, 'task_info.json'), 'w', encoding='utf-8') as f:
            f.write('{not json')
        self.assertIsNone(read_task_manifest(bad))

    def test_invalid_task_id_rejected(self):
        with self.assertRaises(ValueError):
            write_task_manifest('../escape', {})


class TestActiveTasksEndpoint(unittest.TestCase):

    def setUp(self):
        from evalscope.service.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.client = create_app(outputs=self.tmp).test_client()
        patcher = mock.patch('evalscope.service.blueprints.tasks.OUTPUT_DIR', self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for task in list_active_tasks():
            unregister_process(task['task_id'])

    def test_empty(self):
        body = self.client.get('/api/v1/tasks/active').get_json()
        self.assertEqual(body, {'tasks': [], 'total': 0})

    def test_lists_registered_task_with_progress(self):
        task_dir = os.path.join(self.tmp, 'perf_1')
        os.makedirs(os.path.join(task_dir, 'perf'))
        with open(os.path.join(task_dir, 'perf', 'progress.json'), 'w', encoding='utf-8') as f:
            json.dump({'percent': 42.0}, f)
        register_process('perf_1', mock.Mock(), meta={'kind': 'perf', 'model': 'm', 'endpoint': 'h:1', 'alias': 'A'})

        body = self.client.get('/api/v1/tasks/active').get_json()
        self.assertEqual(body['total'], 1)
        task = body['tasks'][0]
        self.assertEqual(task['task_id'], 'perf_1')
        self.assertEqual(task['percent'], 42.0)
        self.assertEqual(task['alias'], 'A')

    def test_missing_progress_file_yields_zero(self):
        register_process('eval_2', mock.Mock(), meta={'kind': 'eval', 'model': 'm'})
        body = self.client.get('/api/v1/tasks/active').get_json()
        self.assertEqual(body['tasks'][0]['percent'], 0.0)


if __name__ == '__main__':
    unittest.main()
