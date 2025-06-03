import json
import os
import unittest

from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.utils.io_utils import dump_jsonl_data
from evalscope.utils.utils import test_level_list


class TestCollection(unittest.TestCase):
    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_create_collection(self):
        schema = CollectionSchema(name='math&reasoning', datasets=[
                    CollectionSchema(name='math', datasets=[
                        CollectionSchema(name='generation', datasets=[
                            DatasetInfo(name='gsm8k', weight=1, task_type='math', tags=['en', 'math']),
                            DatasetInfo(name='competition_math', weight=1, task_type='math', tags=['en', 'math']),
                        ]),
                        CollectionSchema(name='multiple_choice', datasets=[
                            DatasetInfo(name='cmmlu', weight=2, task_type='math', tags=['zh', 'math'], args={'subset_list': ['college_mathematics', 'high_school_mathematics']}),
                            DatasetInfo(name='ceval', weight=3, task_type='math', tags=['zh', 'math'], args={'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}),
                        ]),
                    ]),
                    CollectionSchema(name='reasoning', datasets=[
                        DatasetInfo(name='arc', weight=1, task_type='reasoning', tags=['en', 'reasoning']),
                        DatasetInfo(name='ceval', weight=1, task_type='reasoning', tags=['zh', 'reasoning'], args={'subset_list': ['logic']}),
                        DatasetInfo(name='race', weight=1, task_type='reasoning', tags=['en', 'reasoning']),
                    ]),
                ])
        print(schema.to_dict())
        print(schema.flatten())
        schema.dump_json('outputs/schema_test.json')


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_generate_data(self):
        schema = CollectionSchema.from_dict(json.load(open('outputs/schema_test.json', 'r')))
        print(schema.to_dict())
        mixed_data = WeightedSampler(schema).sample(100)
        dump_jsonl_data(mixed_data, 'outputs/mixed_data_test.jsonl')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_evaluate_collection(self):
        from evalscope import TaskConfig, run_task

        task_cfg = TaskConfig(
            model='Qwen2.5-0.5B-Instruct',
            api_url='http://127.0.0.1:8801/v1/chat/completions',
            api_key='EMPTY',
            eval_type=EvalType.SERVICE,
            datasets=['data_collection'],
            dataset_args={'data_collection': {
                'local_path': 'outputs/mixed_data_test.jsonl'
                # 'local_path': 'outputs/weighted_mixed_data.jsonl'
            }},
        )
        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_evaluate_collection_with_judge(self):
        from evalscope import TaskConfig, run_task

        task_cfg = TaskConfig(
            model='qwen2.5-7b-instruct',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= os.getenv('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=['data_collection'],
            dataset_args={'data_collection': {
                'local_path': 'outputs/mixed_data_test.jsonl'
                # 'local_path': 'outputs/weighted_mixed_data.jsonl'
            }},
            limit=5,
            judge_strategy=JudgeStrategy.AUTO,
            judge_model_args={
                # 'model_id': 'qwen2.5-72b-instruct',
                # 'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                # 'api_key': os.getenv('DASHSCOPE_API_KEY'),
            },
            analysis_report=True,
            # use_cache='outputs/20250522_204520'
        )
        res = run_task(task_cfg=task_cfg)
        print(res)
