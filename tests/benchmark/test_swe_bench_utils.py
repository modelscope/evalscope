"""Unit tests for SWE-bench utility functions (custom dockerhub_username)."""

import unittest
from unittest.mock import patch


class TestGetRemoteDockerImageFromId(unittest.TestCase):
    """Test get_remote_docker_image_from_id with various dockerhub_username values."""

    @patch('platform.machine', return_value='x86_64')
    def test_default_namespace(self, _mock_machine):
        from evalscope.benchmarks.swe_bench.utils import get_remote_docker_image_from_id

        result = get_remote_docker_image_from_id('django__django-12345')
        assert result == 'swebench/sweb.eval.x86_64.django_1776_django-12345:latest'

    @patch('platform.machine', return_value='x86_64')
    def test_custom_namespace(self, _mock_machine):
        from evalscope.benchmarks.swe_bench.utils import get_remote_docker_image_from_id

        result = get_remote_docker_image_from_id('django__django-12345', dockerhub_username='my-mirror')
        assert result == 'my-mirror/sweb.eval.x86_64.django_1776_django-12345:latest'

    @patch('platform.machine', return_value='x86_64')
    def test_empty_namespace_falls_back_to_default(self, _mock_machine):
        from evalscope.benchmarks.swe_bench.utils import get_remote_docker_image_from_id

        result = get_remote_docker_image_from_id('django__django-12345', dockerhub_username='')
        assert result == 'swebench/sweb.eval.x86_64.django_1776_django-12345:latest'

    @patch('platform.machine', return_value='x86_64')
    def test_none_namespace_falls_back_to_default(self, _mock_machine):
        from evalscope.benchmarks.swe_bench.utils import get_remote_docker_image_from_id

        result = get_remote_docker_image_from_id('django__django-12345', dockerhub_username=None)
        assert result == 'swebench/sweb.eval.x86_64.django_1776_django-12345:latest'

    @patch('platform.machine', return_value='x86_64')
    def test_full_registry_path(self, _mock_machine):
        from evalscope.benchmarks.swe_bench.utils import get_remote_docker_image_from_id

        result = get_remote_docker_image_from_id(
            'django__django-12345',
            dockerhub_username='registry.cn-hangzhou.aliyuncs.com/swebench',
        )
        expected = 'registry.cn-hangzhou.aliyuncs.com/swebench/sweb.eval.x86_64.django_1776_django-12345:latest'
        assert result == expected


if __name__ == '__main__':
    unittest.main()
