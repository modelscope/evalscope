# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import os
from evals.constants import PredictorMode, PredictorKeys, PredictorEnvs
from evals.predictors.moss_predictor import MossPredictor
from evals.utils.utils import test_level_list

DEFAULT_TEST_LEVEL = 0
ENABLE_LOCAL_PREDICTOR = False


def condition(test_level=DEFAULT_TEST_LEVEL):
    return test_level in test_level_list()


class TestMossPredictor(unittest.TestCase):

    # TODO: to be adapted with new predictor

    def setUp(self) -> None:
        api_key = os.environ.get(PredictorEnvs.DASHSCOPE_API_KEY, None)
        self.remote_predictor = MossPredictor(api_key=api_key, mode=PredictorMode.REMOTE)
        self.local_predictor = None
        if ENABLE_LOCAL_PREDICTOR:
            self.local_predictor = TestMossPredictor._init_local_predictor()

    @staticmethod
    def _init_local_predictor():
        model_cfg = dict(
            local_model={'model_path': ''},
        )
        predictor = MossPredictor(api_key='', mode=PredictorMode.LOCAL, **model_cfg)

        return predictor

    @unittest.skipUnless(condition(test_level=0), 'skip test in current test level')
    def test_remote_predict(self):
        from dashscope import Models

        input_args = dict(
            model='moss_dev3',
            prompt='推荐一个附近的公园',
            history=[
                {
                    "user": "今天天气好吗？",
                    "bot": "今天天气不错，要出去玩玩嘛？"
                },
                {
                    "user": "那你有什么地方推荐？",
                    "bot": "我建议你去公园，春天来了，花朵开了，很美丽。"
                }
            ],
            max_length=500,
            top_k=15,
        )

        result_dict = self.remote_predictor(**input_args)
        print(result_dict)
        self.assertTrue(result_dict)
        self.assertTrue(result_dict['output'])
        self.assertTrue(result_dict['output']['text'])

    @unittest.skipUnless(condition(test_level=1), 'skip test in current test level')
    def test_local_predict(self):

        input_args = dict(
            prompt='推荐一个附近的公园',
            history=[
                {
                    "user": "今天天气好吗？",
                    "bot": "今天天气不错，要出去玩玩嘛？"
                },
                {
                    "user": "那你有什么地方推荐？",
                    "bot": "我建议你去公园，春天来了，花朵开了，很美丽。"
                }
            ],
            max_length=500,
            top_k=15,
        )

        result_dict = self.local_predictor(**input_args)
        print(result_dict)

