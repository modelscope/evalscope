import os
import unittest

import evalscope.benchmarks  # noqa: F401
from evalscope.api.messages import ContentVideo
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.models.utils.openai import openai_chat_messages


class TestMVBenchVideoBenchmark(unittest.TestCase):

    def test_loads_public_mvbench_video_sample(self):
        task_cfg = TaskConfig(
            model='mockllm',
            eval_type=EvalType.MOCK_LLM,
            datasets=['mvbench'],
            dataset_args={'mvbench': {'subset_list': ['action_antonym']}},
            limit=1,
        )

        adapter = get_benchmark('mvbench', task_cfg)
        dataset = adapter.load_dataset()
        sample = dataset['action_antonym'][0]

        video_parts = [
            part for message in sample.input for part in message.content if isinstance(part, ContentVideo)
        ]
        self.assertEqual(len(video_parts), 1)
        self.assertTrue(os.path.exists(video_parts[0].video))
        self.assertEqual(video_parts[0].format, 'mp4')
        self.assertIn(sample.target, ['A', 'B', 'C'])

        request_messages = openai_chat_messages(sample.input)
        video_request_parts = [part for part in request_messages[0]['content'] if part['type'] == 'video_url']
        self.assertEqual(len(video_request_parts), 1)
        self.assertTrue(video_request_parts[0]['video_url']['url'].startswith('data:video/mp4;base64,'))
