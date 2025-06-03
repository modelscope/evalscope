from dotenv import dotenv_values

env = dotenv_values('.env')

import os
import unittest

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy, ModelTask, OutputType
from evalscope.run import run_task
from evalscope.utils import test_level_list
from evalscope.utils.logger import get_logger

os.environ['EVALSCOPE_LOG_LEVEL'] = 'DEBUG'

logger = get_logger()


class TestRun(unittest.TestCase):
    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_general(self):
        from evalscope.config import TaskConfig

        task_cfg = TaskConfig(
            datasets=[
                'general_t2i'
            ],
            dataset_args={
                'general_t2i': {
                    'metric_list': [
                        'PickScore',
                        'CLIPScore',
                        'HPSv2Score',
                        'HPSv2.1Score',
                        'BLIPv2Score',
                        'ImageRewardScore',
                        'VQAScore',
                        'FGA_BLIP2Score',
                        'MPS'
                    ],
                    'dataset_id': 'custom_eval/multimodal/t2i/example.jsonl',
                }
            }
        )

        run_task(task_cfg=task_cfg)


    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_benchmark(self):

        task_cfg = TaskConfig(
            model='stabilityai/stable-diffusion-xl-base-1.0',  # model on modelscope
            model_task=ModelTask.IMAGE_GENERATION,  # must be IMAGE_GENERATION
            model_args={
                'use_safetensors': True,
                'variant': 'fp16',
                'torch_dtype': 'torch.float16',
            },
            datasets=[
                # 'tifa160',
                # 'genai_bench',
                'evalmuse',
                # 'hpdv2',
            ],
            dataset_args={
                'tifa160': {
                    'metric_list': [
                        'PickScore',
                        # 'CLIPScore',
                        # 'HPSv2Score',
                        # 'BLIPv2Score',
                        # 'ImageRewardScore',
                        # 'VQAScore',
                        # 'FGA_BLIP2Score',
                    ]
                }
            },
            limit=5,
            generation_config={
                'num_inference_steps': 50,
                'guidance_scale': 7.5
            },
            # use_cache='outputs/20250427_134122',
        )

        run_task(task_cfg=task_cfg)

    @unittest.skipUnless(0 in test_level_list(), 'skip test in current test level')
    def test_run_benchmark_flux(self):

        task_cfg = TaskConfig(
            model='black-forest-labs/FLUX.1-dev',  # model on modelscope
            model_task=ModelTask.IMAGE_GENERATION,  # must be IMAGE_GENERATION
            model_args={
                'torch_dtype': 'torch.float16',
            },
            datasets=[
                # 'tifa160',
                # 'genai_bench',
                'evalmuse',
                # 'hpdv2',
            ],
            dataset_args={
                'tifa160': {
                    'metric_list': [
                        'PickScore',
                        # 'CLIPScore',
                        # 'HPSv2Score',
                        # 'BLIPv2Score',
                        # 'ImageRewardScore',
                        # 'VQAScore',
                        # 'FGA_BLIP2Score',
                    ]
                }
            },
            generation_config={
                'num_inference_steps': 50,
                'guidance_scale': 3.5
            },
            use_cache='outputs/20250520_112314'
        )

        run_task(task_cfg=task_cfg)
