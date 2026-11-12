# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fail-fast checks for API plugins that cannot send HTTP requests (issue #2069).

``DashScopeApiPlugin`` defines ``build_request`` and ``parse_responses`` but not
``process_request``.  Every request used to fail with an empty error message;
the perf entry point now rejects such plugins before any request is sent.
"""
import unittest
from unittest.mock import MagicMock

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.perf.main import check_api_plugin, run_perf_benchmark
from evalscope.perf.plugin import ApiRegistry
from evalscope.perf.plugin.api.dashscope_api import DashScopeApiPlugin
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin


class TestCheckApiPlugin(unittest.TestCase):

    def test_dashscope_is_rejected_with_plugin_name(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            check_api_plugin(Arguments(model='m', api='dashscope'))
        message = str(ctx.exception)
        self.assertIn('dashscope', message)
        self.assertIn('DashScopeApiPlugin', message)
        self.assertIn('process_request', message)

    def test_openai_passes(self) -> None:
        check_api_plugin(Arguments(model='m', api='openai'))

    def test_run_perf_benchmark_rejects_before_creating_output(self) -> None:
        with self.assertRaises(ValueError):
            run_perf_benchmark(Arguments(model='m', api='dashscope'))

    def test_implements_process_request(self) -> None:
        self.assertFalse(DashScopeApiPlugin.implements_process_request())
        self.assertTrue(OpenaiPlugin.implements_process_request())

    def test_registered_plugins_other_than_dashscope_implement_process_request(self) -> None:
        missing = [
            name for name in ApiRegistry.all_classes()
            if not ApiRegistry.get_class(name).implements_process_request()
        ]
        self.assertEqual(missing, ['dashscope'])


class TestBaseProcessRequestMessage(unittest.IsolatedAsyncioTestCase):

    async def test_not_implemented_error_names_the_class(self) -> None:
        plugin = DashScopeApiPlugin(Arguments(model='m', api='dashscope'))
        with self.assertRaises(NotImplementedError) as ctx:
            await plugin.process_request(MagicMock(), 'http://localhost', {}, {})
        self.assertIn('DashScopeApiPlugin', str(ctx.exception))
        self.assertIn('process_request', str(ctx.exception))


class TestEmptyExceptionMessage(unittest.IsolatedAsyncioTestCase):

    async def test_failure_record_has_nonempty_error(self) -> None:
        args = Arguments(model='m', api='dashscope', parallel=1, number=1)
        args.parallel = 1
        plugin = DashScopeApiPlugin(args)
        plugin.process_request = MagicMock(side_effect=RuntimeError())
        async with AioHttpClient(args, plugin) as client:
            with self.assertLogs('evalscope', level='ERROR') as logs:
                record = await client.post({'prompt': 'hi'})
        self.assertFalse(record.success)
        self.assertIn('RuntimeError', record.error)
        self.assertTrue(all(m.strip() for m in logs.output))
        self.assertTrue(any('RuntimeError' in m for m in logs.output))


if __name__ == '__main__':
    unittest.main()
