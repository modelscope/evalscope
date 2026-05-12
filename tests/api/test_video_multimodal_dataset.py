import os
import tempfile
import unittest

import evalscope.benchmarks  # noqa: F401
from evalscope.api.messages import ChatMessageUser, ContentVideo
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.models.utils.openai import openai_chat_messages


class TestVideoMultimodalDataset(unittest.TestCase):

    def test_general_vqa_loads_video_url_messages(self):
        task_cfg = TaskConfig(
            model='mockllm',
            eval_type=EvalType.MOCK_LLM,
            datasets=['general_vqa'],
            dataset_args={
                'general_vqa': {
                    'local_path': 'custom_eval/multimodal/vqa',
                    'subset_list': ['example_video'],
                }
            },
        )

        adapter = get_benchmark('general_vqa', task_cfg)
        dataset = adapter.load_dataset()
        sample = dataset['example_video'][0]

        video_parts = [
            part for message in sample.input for part in message.content if isinstance(part, ContentVideo)
        ]
        self.assertEqual(len(video_parts), 1)
        self.assertEqual(video_parts[0].video, 'custom_eval/multimodal/videos/sample.mp4')
        self.assertEqual(video_parts[0].format, 'mp4')

        request_messages = openai_chat_messages(sample.input)
        content = request_messages[0]['content']
        video_request_parts = [part for part in content if part['type'] == 'video_url']
        self.assertEqual(len(video_request_parts), 1)
        self.assertTrue(video_request_parts[0]['video_url']['url'].startswith('data:video/mp4;base64,'))

    def test_general_vmcq_loads_video_placeholder(self):
        task_cfg = TaskConfig(
            model='mockllm',
            eval_type=EvalType.MOCK_LLM,
            datasets=['general_vmcq'],
            dataset_args={
                'general_vmcq': {
                    'local_path': 'custom_eval/multimodal/mcq',
                    'subset_list': ['example_video'],
                }
            },
        )

        adapter = get_benchmark('general_vmcq', task_cfg)
        dataset = adapter.load_dataset()
        sample = dataset['example_video'][0]

        video_parts = [
            part for message in sample.input for part in message.content if isinstance(part, ContentVideo)
        ]
        self.assertEqual(len(video_parts), 1)
        self.assertEqual(video_parts[0].video, 'custom_eval/multimodal/videos/sample.mp4')
        self.assertEqual(video_parts[0].format, 'mp4')
        self.assertEqual(sample.choices, ['Image', 'Audio', 'Video', 'Text'])
        self.assertEqual(sample.target, 'C')

    def test_video_format_hint_is_used_for_extensionless_local_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = os.path.join(tmp_dir, 'sample')
            with open('custom_eval/multimodal/videos/sample.mp4', 'rb') as source, open(video_path, 'wb') as target:
                target.write(source.read())

            messages = [ChatMessageUser(content=[ContentVideo(video=video_path, format='mp4')])]
            request_messages = openai_chat_messages(messages)

        video_request_parts = [part for part in request_messages[0]['content'] if part['type'] == 'video_url']
        self.assertEqual(len(video_request_parts), 1)
        self.assertTrue(video_request_parts[0]['video_url']['url'].startswith('data:video/mp4;base64,'))
