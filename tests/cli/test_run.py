# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

import os
import subprocess
import unittest

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy, OutputType
from evalscope.run import run_task
from evalscope.utils import is_module_installed, test_level_list
from evalscope.utils.logger import get_logger

os.environ['EVALSCOPE_LOG_LEVEL'] = 'DEBUG'

logger = get_logger()


class TestRun(unittest.TestCase):

    def setUp(self) -> None:
        logger.info('Init env for evalscope native run UTs ...\n')
        self._check_env('evalscope')

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'{module_name} is installed.')
        else:
            raise ModuleNotFoundError(f'run: pip install {module_name}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_simple_eval(self):
        model = 'qwen/Qwen2-0.5B-Instruct'
        datasets = 'arc'  # arc ceval
        limit = 10

        cmd_simple = f'evalscope eval ' \
                     f'--model {model} ' \
                     f'--datasets {datasets} ' \
                     f'--limit {limit}'

        logger.info(f'Start to run command: {cmd_simple}')
        run_res = subprocess.run(cmd_simple, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        assert run_res.returncode == 0, f'Failed to run command: {cmd_simple}'
        logger.info(f'>>test_run_simple_eval stdout: {run_res.stdout}')
        logger.error(f'>>test_run_simple_eval stderr: {run_res.stderr}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_eval_with_args(self):
        model = 'qwen/Qwen2-0.5B-Instruct'
        datasets = 'arc'  # arc ceval
        limit = 5
        dataset_args = '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}'

        cmd_with_args = f'evalscope eval ' \
                        f'--model {model} ' \
                        f'--datasets {datasets} ' \
                        f'--limit {limit} ' \
                        f'--generation-config do_sample=false,temperature=0.0 ' \
                        f"""--dataset-args \'{dataset_args}\' """

        logger.info(f'Start to run command: {cmd_with_args}')
        run_res = subprocess.run(cmd_with_args, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        assert run_res.returncode == 0, f'Failed to run command: {cmd_with_args}'
        logger.info(f'>>test_run_eval_with_args stdout: {run_res.stdout}')
        logger.error(f'>>test_run_eval_with_args stderr: {run_res.stderr}')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_yaml_config(self):
        from evalscope import run_task

        run_task(task_cfg='examples/tasks/eval_native.yaml')

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_task(self):
        task_cfg = TaskConfig(
            model='qwen/Qwen2.5-0.5B-Instruct',
            datasets=[
                'iquiz',
                # 'ifeval',
                # 'mmlu',
                # 'mmlu_pro',
                # 'musr',
                # 'process_bench',
                # 'race',
                # 'trivia_qa',
                # 'cmmlu',
                # 'humaneval',
                # 'super_gpqa',
                # 'gsm8k',
                # 'bbh',
                # 'competition_math',
                # 'math_500',
                'aime24',
                'gpqa',
                # 'arc',
                # 'ceval',
                # 'hellaswag',
                # 'general_mcq',
                # 'general_qa'
            ],
            dataset_args={
                'mmlu': {
                    'subset_list': ['elementary_mathematics'],
                    'few_shot_num': 0
                },
                'mmlu_pro': {
                    'subset_list': ['math', 'health'],
                    'few_shot_num': 4
                },
                'ceval': {
                    'subset_list': [
                        'computer_network', 'operating_system', 'computer_architecture'
                    ],
                    'few_shot_num': 0
                },
                'cmmlu': {
                    'subset_list': ['elementary_chinese'],
                    'few_shot_num': 0
                },
                'bbh': {
                    'subset_list': ['word_sorting', 'movie_recommendation'],
                },
                'gpqa': {
                    'subset_list': ['gpqa_diamond'],
                    'few_shot_num': 0
                },
                'humaneval': {
                    'metric_list': ['Pass@1', 'Pass@2', 'Pass@5'],
                },
                'competition_math': {
                    'subset_list': ['Level 1']
                },
                'process_bench': {
                    'subset_list': ['gsm8k'],
                },
                'musr': {
                    'subset_list': ['murder_mysteries'],
                },
                'general_mcq': {
                    'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ],
                    'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}'  # 问题模板
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example',  # 评测数据集名称，上述 *_dev.csv 中的 *
                        # 'test'
                    ],
                    'metric_list': ['AverageBLEU']
                },
                'super_gpqa': {
                    'subset_list': ['Philosophy', 'Education'],
                    'few_shot_num': 0
                },
                'ifeval': {
                    'filters': {
                        'remove_until': '</think>'
                    }
                }
            },
            limit=2,
            eval_batch_size=2,
            generation_config={
                'max_new_tokens': 2048,
                'temperature': 0.7,
                'num_return_sequences': 1,
            },
            # debug=True
        )
        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_custom_task(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen/Qwen2-0.5B-Instruct',
            datasets=['general_mcq', 'general_qa'],  # 数据格式，选择题格式固定为 'ceval'
            dataset_args={
                'general_mcq': {
                    'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ],
                    'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}'  # 问题模板
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ]
                }
            },
        )
        res = run_task(task_cfg=task_cfg)
        print(res)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_one_task(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='Qwen/Qwen3-1.7B',
            datasets=[
                'iquiz',
                # 'math_500',
                # 'aime24',
                # 'competition_math',
                # 'mmlu',
                # 'simple_qa',
            ],
            model_args={
                'device_map': 'auto',
            },
            dataset_args={
                'competition_math': {
                    'subset_list': ['Level 4', 'Level 5']
                },
                'mmlu': {
                    'subset_list': ['elementary_mathematics', 'high_school_european_history', 'nutrition'],
                    'few_shot_num': 0
                },
            },
            limit=5,
            eval_batch_size=5,
            generation_config={
                'max_new_tokens': 1000,  # 最大生成token数，建议设置为较大值避免输出截断
                'temperature': 0.7,  # 采样温度 (qwen 报告推荐值)
                'top_p': 0.8,  # top-p采样 (qwen 报告推荐值)
                'top_k': 20,  # top-k采样 (qwen 报告推荐值)
                'chat_template_kwargs': {'enable_thinking': False}  # 关闭思考模式
            },
            judge_strategy=JudgeStrategy.AUTO,
        )

        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_task_loop(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        from evalscope.config import TaskConfig

        task_cfg1 = TaskConfig(
            model='Qwen/Qwen2.5-0.5B-Instruct',
            model_id='model1',
            datasets=['iquiz'],
            limit=10
        )
        task_cfg2 = TaskConfig(
            model='Qwen/Qwen2.5-0.5B-Instruct',
            model_id='model2',
            datasets=['iquiz'],
            limit=10
        )
        task_cfg3 = TaskConfig(
            model='Qwen/Qwen2.5-0.5B-Instruct',
            model_id='model3',
            datasets=['iquiz'],
            limit=10
        )

        run_task(task_cfg=[task_cfg1, task_cfg2, task_cfg3])

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_server_model(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen-plus',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= env.get('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=[
                # 'iquiz',
                # 'ifeval',
                # 'mmlu',
                # 'mmlu_pro',
                # 'musr',
                # 'process_bench',
                # 'race',
                # 'trivia_qa',
                # 'cmmlu',
                # 'humaneval',
                # 'gsm8k',
                # 'bbh',
                # 'competition_math',
                # 'math_500',
                # 'aime24',
                # 'gpqa',
                # 'arc',
                # 'ceval',
                # 'hellaswag',
                # 'general_mcq',
                # 'general_qa',
                # 'super_gpqa',
                # 'mmlu_redux',
                # 'maritime_bench',
                # 'drop',
                # 'winogrande',
                # 'tool_bench',
                'frames',
            ],
            dataset_args={
                'mmlu': {
                    'subset_list': ['elementary_mathematics', 'high_school_european_history', 'nutrition'],
                    'few_shot_num': 0
                },
                'mmlu_pro': {
                    'subset_list': ['math', 'health'],
                    'few_shot_num': 4
                },
                'ceval': {
                    'subset_list': [
                        'computer_network', 'operating_system', 'computer_architecture'
                    ],
                    'few_shot_num': 0
                },
                'cmmlu': {
                    'subset_list': ['elementary_chinese'],
                    'few_shot_num': 0
                },
                'bbh': {
                    'subset_list': ['word_sorting', 'movie_recommendation'],
                },
                'gpqa': {
                    # 'subset_list': ['gpqa_diamond'],
                    'few_shot_num': 0,
                    'local_path': './data/data/gpqa',
                },
                'humaneval': {
                    'metric_list': ['Pass@1', 'Pass@2', 'Pass@5'],
                },
                'competition_math': {
                    'subset_list': ['Level 1']
                },
                'process_bench': {
                    'subset_list': ['gsm8k'],
                },
                'musr': {
                    'subset_list': ['murder_mysteries'],
                    'local_path': '/root/.cache/modelscope/hub/datasets/AI-ModelScope/MuSR'
                },
                'general_mcq': {
                    'local_path': 'custom_eval/text/mcq',  # 自定义数据集路径
                    'subset_list': [
                        'example'  # 评测数据集名称，上述 *_dev.csv 中的 *
                    ],
                    'query_template': 'Question: {question}\n{choices}\nAnswer: {answer}'  # 问题模板
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example',  # 评测数据集名称，上述 *_dev.csv 中的 *
                        # 'test'
                    ],
                    'metric_list': ['AverageRouge']
                },
                'super_gpqa': {
                    # 'subset_list': ['Philosophy', 'Education'],
                    'few_shot_num': 0
                },
                'mmlu_redux':{
                    'subset_list': ['abstract_algebra']
                },
            },
            eval_batch_size=32,
            limit=10,
            debug=True,
            stream=False,
            generation_config={
                'temperature': 0,
                'n': 1,
                'max_tokens': 4096,
                # 'extra_headers':{'key': 'value'},
            },
            # ignore_errors=True,
            # use_cache='outputs/20250519_142106'
        )

        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_batch_eval(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='LLM-Research/Llama-3.2-1B-Instruct',
            datasets=[
                # 'math_500',
                # 'aime24',
                # 'competition_math'
                # 'arc',
                'gsm8k'
                # 'truthful_qa'
            ],
            dataset_args={
                'competition_math': {
                    'subset_list': ['Level 4', 'Level 5']
                }
            },
            eval_batch_size=2,
            limit=5,
            generation_config={
                'max_new_tokens': 2048,
                'temperature': 0.7,
                'num_return_sequences': 2,
            }
        )

        run_task(task_cfg=task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_judge_model(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            model='qwen-plus',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key= env.get('DASHSCOPE_API_KEY'),
            eval_type=EvalType.SERVICE,
            datasets=[
                # 'math_500',
                'aime24',
                # 'competition_math',
                # 'arc',
                # 'gsm8k',
                # 'truthful_qa',
                # 'simple_qa',
                # 'chinese_simpleqa',
                # 'live_code_bench',
                # 'humaneval',
                # 'general_qa',
                # 'alpaca_eval',
                # 'arena_hard',
                # 'frames',
                # 'docmath',
                # 'needle_haystack',
            ],
            dataset_args={
                'competition_math': {
                    'subset_list': ['Level 4']
                },
                'live_code_bench': {
                    'extra_params': {
                        'start_date': '2024-08-01',
                        'end_date': '2025-02-28'
                    },
                    'local_path': '/root/.cache/modelscope/hub/datasets/AI-ModelScope/code_generation_lite'
                },
                'general_qa': {
                    'local_path': 'custom_eval/text/qa',  # 自定义数据集路径
                    'subset_list': [
                        'example',  # 评测数据集名称，上述 *_dev.csv 中的 *
                        # 'test'
                    ]
                },
                'chinese_simpleqa': {
                    'subset_list': [
                        '中华文化'
                    ]
                },
                'frames': {
                    'local_path': '/root/.cache/modelscope/hub/datasets/iic/frames'
                }
            },
            eval_batch_size=10,
            limit=1,
            judge_strategy=JudgeStrategy.AUTO,
            judge_worker_num=5,
            judge_model_args={
                'model_id': 'qwen2.5-72b-instruct',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096
                }
            },
            generation_config={
                'max_new_tokens': 20000,
                'temperature': 0.0,
                'seed': 42,
                'n': 1
            },
            timeout=60000,
            stream=True,
            analysis_report=True,
            # debug=True,
            # use_cache='outputs/20250602_135859'
        )

        run_task(task_cfg=task_cfg)

if __name__ == '__main__':
    unittest.main()
