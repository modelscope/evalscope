# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import tempfile
import unittest

import evalscope.benchmarks  # noqa: F401
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


class TestAIRBenchAdapters(unittest.TestCase):

    def _write_json(self, path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def test_foundation_loads_local_data_with_ids_repeats_and_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = os.path.join(tmp, 'Foundation')
            os.makedirs(os.path.join(track, 'Music_AQA_music_avqa'), exist_ok=True)
            os.makedirs(os.path.join(track, 'Sound_AQA_avqa'), exist_ok=True)
            with open(os.path.join(track, 'Music_AQA_music_avqa', 'a.wav'), 'wb'):
                pass
            with open(os.path.join(track, 'Sound_AQA_avqa', 'b.wav'), 'wb'):
                pass
            self._write_json(
                os.path.join(track, 'Foundation_meta.json'),
                [
                    {
                        'uniq_id': 1,
                        'task_name': 'Music_AQA',
                        'dataset_name': 'music_avqa',
                        'path': 'a.wav',
                        'question': 'Which instrument is heard?',
                        'choice_a': 'piano',
                        'choice_b': 'guitar',
                        'choice_c': 'drums',
                        'choice_d': 'violin',
                        'answer_gt': 'guitar',
                    },
                    {
                        'uniq_id': 2,
                        'task_name': 'Sound_AQA',
                        'dataset_name': 'avqa',
                        'path': 'b.wav',
                        'question': 'What sound is present?',
                        'choice_a': 'rain',
                        'choice_b': 'wind',
                        'choice_c': 'car',
                        'choice_d': 'dog',
                        'answer_gt': 'A',
                    },
                ],
            )

            cfg = TaskConfig(
                dataset_args={
                    'air_bench_foundation': {
                        'local_path': tmp,
                        'subset_list': ['Music_AQA_music_avqa'],
                    }
                },
                limit=1,
                repeats=2,
            )
            adapter = get_benchmark('air_bench_foundation', config=cfg)
            dataset_dict = adapter.load_dataset()

            self.assertEqual(list(dataset_dict.keys()), ['Music_AQA_music_avqa'])
            samples = list(dataset_dict['Music_AQA_music_avqa'])
            self.assertEqual(len(samples), 2)
            self.assertEqual([s.id for s in samples], [0, 1])
            self.assertEqual([s.group_id for s in samples], [0, 0])
            self.assertEqual(samples[0].target, 'B')
            self.assertEqual(samples[0].metadata['category'], 'music')
            self.assertEqual(adapter.category_map['Music_AQA_music_avqa'], 'music')

    def test_chat_loads_local_data_and_parses_labeled_judge_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = os.path.join(tmp, 'Chat')
            os.makedirs(os.path.join(track, 'speech_QA_common_voice_en'), exist_ok=True)
            with open(os.path.join(track, 'speech_QA_common_voice_en', 'a.wav'), 'wb'):
                pass
            self._write_json(
                os.path.join(track, 'Chat_meta.json'),
                [
                    {
                        'uniq_id': 1,
                        'task_name': 'speech_QA',
                        'dataset_name': 'common_voice_en',
                        'path': 'a.wav',
                        'question': 'What did the speaker say?',
                        'answer_gt': 'hello',
                        'meta_info': 'A speaker says hello.',
                    }
                ],
            )

            cfg = TaskConfig(
                dataset_args={
                    'air_bench_chat': {
                        'local_path': tmp,
                        'extra_params': {'tasks': ['speech_QA'], 'do_swap': False},
                    }
                },
                limit=1,
            )
            adapter = get_benchmark('air_bench_chat', config=cfg)
            dataset_dict = adapter.load_dataset()

            self.assertEqual(list(dataset_dict.keys()), ['speech_QA'])
            sample = dataset_dict['speech_QA'][0]
            self.assertEqual(sample.id, 0)
            self.assertEqual(sample.target, 'hello')
            self.assertEqual(sample.metadata['category'], 'speech')
            self.assertEqual(adapter.category_map['speech_QA'], 'speech')
            self.assertEqual(adapter._extract_judge_scores('Assistant 1: 8\nAssistant 2: 7'), ['8', '7'])
            self.assertEqual(adapter._extract_judge_scores('Scores: 8.5 7'), ['8.5', '7'])
            self.assertEqual(adapter._extract_judge_scores('Assistant 1: 8/10 Assistant 2: 7/10'), ['8', '7'])
            self.assertEqual(adapter._extract_judge_scores('Assistant 1: 0/10 Assistant 2: 8/10'), ['0', '8'])

            class DummyJudge:
                model_id = 'dummy'

                def judge(self, prompt, system_prompt=None):
                    return '0 8'

            adapter.llm_judge = DummyJudge()
            self.assertEqual(adapter._judge_pair('prompt')[:2], (None, None))


if __name__ == '__main__':
    unittest.main()
