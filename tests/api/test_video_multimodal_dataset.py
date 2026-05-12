import os
import tempfile
import unittest

import evalscope.benchmarks  # noqa: F401
from evalscope.api.dataset import DatasetHub, download_dataset_file
from evalscope.api.messages import ChatMessageUser, ContentVideo
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, HubType
from evalscope.models.utils.openai import content_from_openai, openai_chat_messages


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

    def test_openai_video_content_preserves_time_boundary_metadata(self):
        content = content_from_openai({
            'type': 'video_url',
            'video_url': {
                'url': 'https://example.com/sample.mp4',
                'format': 'mp4',
                'start': 3,
                'end': 9.5,
                'fps': 1,
            },
        })

        self.assertEqual(len(content), 1)
        self.assertIsInstance(content[0], ContentVideo)
        self.assertEqual(content[0].start, 3.0)
        self.assertEqual(content[0].end, 9.5)
        self.assertEqual(content[0].fps, 1.0)

    def test_local_dataset_file_resolution_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_dir = os.path.join(tmp_dir, 'video')
            os.makedirs(video_dir)
            local_file = os.path.join(video_dir, 'sample.mp4')
            with open(local_file, 'wb') as target:
                target.write(b'data')

            resolved_path = DatasetHub(tmp_dir, data_source=HubType.LOCAL).download_file('video/sample.mp4')
            self.assertEqual(resolved_path, local_file)

            with self.assertRaisesRegex(ValueError, 'Invalid dataset file path'):
                download_dataset_file(tmp_dir, '../escape.mp4', data_source=HubType.LOCAL)
