import os
import unittest
from unittest import mock

import evalscope.benchmarks  # noqa: F401
from evalscope.api.messages import ContentVideo
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, HubType
from evalscope.models.utils.openai import openai_chat_messages


class TestVideoMMEv2VideoBenchmark(unittest.TestCase):

    def _task_config(self, extra_params=None):
        dataset_args = {'videomme_v2': {'subset_list': ['all']}}
        if extra_params:
            dataset_args['videomme_v2']['extra_params'] = extra_params
        return TaskConfig(
            model='mockllm',
            eval_type=EvalType.MOCK_LLM,
            datasets=['videomme_v2'],
            dataset_args=dataset_args,
            limit=1,
        )

    @staticmethod
    def _record():
        return {
            'video_id': '001',
            'url': 'https://www.youtube.com/watch?v=AYSYelOQtQI',
            'group_type': 'logic',
            'group_structure': '[1,2,3,4]',
            'question_id': '001-1',
            'question': "What is the ethnicity of the protagonist's mother?",
            'options': '\n'.join([
                'A. Malaysian.',
                'B. British.',
                'C. Singaporean.',
                'D. German.',
                'E. Canadian.',
                'F. Chinese.',
                'G. American.',
                'H. Cannot be determined.',
            ]),
            'answer': 'F',
            'level': None,
            'second_head': None,
            'third_head': None,
        }

    def test_loads_videomme_v2_video_url_sample_without_network(self):
        adapter = get_benchmark('videomme_v2', self._task_config())
        with mock.patch.object(adapter, '_load_records', return_value=[self._record()]):
            dataset = adapter.load_dataset()
        sample = dataset['all'][0]

        video_parts = [
            part for message in sample.input for part in message.content if isinstance(part, ContentVideo)
        ]
        self.assertEqual(len(video_parts), 1)
        self.assertEqual(video_parts[0].video, 'https://www.youtube.com/watch?v=AYSYelOQtQI')
        self.assertEqual(video_parts[0].format, 'mp4')
        self.assertEqual(sample.target, 'F')
        self.assertEqual(sample.choices[5], 'Chinese.')
        self.assertEqual(sample.metadata['dataset_hub'], HubType.MODELSCOPE)

        request_messages = openai_chat_messages(sample.input)
        video_request_parts = [part for part in request_messages[0]['content'] if part['type'] == 'video_url']
        self.assertEqual(len(video_request_parts), 1)
        self.assertEqual(video_request_parts[0]['video_url']['url'], 'https://www.youtube.com/watch?v=AYSYelOQtQI')

    def test_includes_subtitles_when_enabled(self):
        adapter = get_benchmark('videomme_v2', self._task_config(extra_params={'use_subtitles': True}))
        with mock.patch.object(adapter, '_load_records', return_value=[self._record()]), \
                mock.patch.object(adapter, '_subtitle_for_record', return_value='Hi, I saw your ad.'):
            dataset = adapter.load_dataset()

        sample = dataset['all'][0]
        self.assertIn('Subtitles:', sample.input[0].content[0].text)
        self.assertIn('Hi, I saw your ad.', sample.input[0].content[0].text)

    def test_rejects_video_cache_path_traversal(self):
        adapter = get_benchmark('videomme_v2', self._task_config())
        with self.assertRaisesRegex(ValueError, 'Invalid Video-MME-v2 video id'):
            adapter._cache_output_path('../escape')

    def test_filters_level_subset(self):
        task_cfg = TaskConfig(
            model='mockllm',
            eval_type=EvalType.MOCK_LLM,
            datasets=['videomme_v2'],
            dataset_args={'videomme_v2': {'subset_list': ['level_1']}},
        )
        adapter = get_benchmark('videomme_v2', task_cfg)
        level_1_record = dict(self._record(), video_id='002', question_id='002-1', level='1')
        level_2_record = dict(self._record(), video_id='003', question_id='003-1', level='2')
        with mock.patch.object(adapter, '_load_records', return_value=[level_1_record, level_2_record]):
            dataset = adapter.load_dataset()

        self.assertEqual(len(dataset['level_1']), 1)
        self.assertEqual(dataset['level_1'][0].metadata['video_id'], '002')

    @unittest.skipUnless(
        os.getenv('EVALSCOPE_RUN_REMOTE_DATA_TESTS') == '1',
        'Set EVALSCOPE_RUN_REMOTE_DATA_TESTS=1 to run public dataset tests.',
    )
    def test_loads_public_videomme_v2_annotation_sample(self):
        adapter = get_benchmark(
            'videomme_v2',
            self._task_config(extra_params={
                'use_subtitles': True,
                'subtitle_word_limit': 24,
            })
        )
        dataset = adapter.load_dataset()
        sample = dataset['all'][0]

        video_parts = [
            part for message in sample.input for part in message.content if isinstance(part, ContentVideo)
        ]
        self.assertEqual(len(video_parts), 1)
        self.assertTrue(video_parts[0].video.startswith('https://'))
        self.assertIn(sample.target, list('ABCDEFGH'))
        self.assertIn('Subtitles:', sample.input[0].content[0].text)
