from dotenv import dotenv_values

env = dotenv_values('.env')
import json
import os
import unittest

from evalscope.collections import CollectionSchema, DatasetInfo, StratifiedSampler, UniformSampler, WeightedSampler
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.utils.io_utils import dump_jsonl_data
from tests.common import TestBenchmark
from tests.utils import test_level_list


class TestCollection(TestBenchmark):
    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen-vl-plus',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.SERVICE,
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'max_tokens': 2048,
                'temperature': 0.0,
                'seed': 42,
                'parallel_tool_calls': True
            },
            'judge_strategy': JudgeStrategy.AUTO,
            'judge_worker_num': 5,
            'judge_model_args': {
                'model_id': 'qwen-plus',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096,
                }
            },
            'debug': True,
        }

    def test_create_simple_collection(self):
        schema = CollectionSchema(name='reasoning_index', datasets=[
            DatasetInfo(name='arc', weight=2.0, task_type='reasoning', tags=['en']),
            DatasetInfo(name='ceval', weight=3.0, task_type='reasoning', tags=['zh'], args={'subset_list': ['logic']}),
        ])
        print(schema.to_dict())
        print(schema.flatten())
        schema.dump_json('outputs/schema_simple_test.json')

    def test_generate_simple_data(self):
        schema = CollectionSchema.from_dict(json.load(open('outputs/schema_simple_test.json', 'r')))
        print(schema.to_dict())
        mixed_data = WeightedSampler(schema).sample(10)
        dump_jsonl_data(mixed_data, 'outputs/mixed_data_simple_test.jsonl')

    def test_generate_simple_data_stratified(self):
        schema = CollectionSchema.from_dict(json.load(open('outputs/schema_simple_test.json', 'r')))
        print(schema.to_dict())
        mixed_data = StratifiedSampler(schema).sample(10)
        dump_jsonl_data(mixed_data, 'outputs/mixed_data_simple_stratified_test.jsonl')

    def test_generate_simple_data_uniform(self):
        schema = CollectionSchema.from_dict(json.load(open('outputs/schema_simple_test.json', 'r')))
        print(schema.to_dict())
        mixed_data = UniformSampler(schema).sample(10)
        dump_jsonl_data(mixed_data, 'outputs/mixed_data_simple_uniform_test.jsonl')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_create_collection(self):
        schema = CollectionSchema(name='math&reasoning', datasets=[
                    CollectionSchema(name='math', datasets=[
                        CollectionSchema(name='generation', datasets=[
                            DatasetInfo(name='gsm8k', weight=1, task_type='math', tags=['en', 'math']),
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

    def test_evaluate_simple_collection(self):
        self._run_dataset_test(
            dataset_name='data_collection',
            dataset_args={
                'local_path': 'outputs/mixed_data_simple_stratified_test.jsonl',
                'shuffle': True,
            },
            model='qwen2.5-7b-instruct',
            limit=10,
        )

    def test_evaluate_collection_with_judge(self):
        self._run_dataset_test(
            dataset_name='data_collection',
            dataset_args={
                'local_path': 'outputs/mixed_data_test.jsonl',
                'shuffle': True,
            },
            model='qwen2.5-7b-instruct',
            limit=10,
        )

    def test_evaluate_rag_index(self):
        self._run_dataset_test(
            dataset_name='data_collection',
            dataset_args={
                'local_path': 'examples/collection/index/rag_index_sampled.jsonl',
            },
            model='qwen-plus',
            limit=10,
            use_cache='outputs/20251124_170860',
            rerun_review=True,
        )
