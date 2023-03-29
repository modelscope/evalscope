# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from evals.constants import PredictorMode
from evals.predictors.moss_predictor import MossPredictor
from evals.utils.utils import test_level_list

TEST_LEVEL = 0


def get_condition():
    return TEST_LEVEL in test_level_list()


class TestMossPredictor(unittest.TestCase):

    def setUp(self) -> None:
        self.predictor = MossPredictor(mode=PredictorMode.REMOTE)

    @unittest.skipUnless(get_condition(), 'skip test in current test level')
    def test_predict(self):
        from dashscope import Models

        input_args = dict(
            model=Models.chatm6_v1,
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

        result_dict = self.predictor.predict(**input_args)
        self.assertTrue(result_dict['output'])
        self.assertTrue(result_dict['output']['text'])
        print(result_dict)
