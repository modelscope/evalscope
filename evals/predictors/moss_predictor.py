# Copyright (c) Alibaba, Inc. and its affiliates.

from evals.constants import PredictorMode
from evals.predictors.base import Predictor

import dashscope
from http import HTTPStatus
from dashscope import Models
from dashscope import Generation


class MossPredictor(Predictor):
    # TODO:
    #   1. class name to be confirmed
    #   2. tdb

    def __init__(self, mode=PredictorMode.REMOTE, **kwargs):
        super(MossPredictor, self).__init__(mode=mode, **kwargs)

    def predict(self, **kwargs) -> dict:
        if self.mode == PredictorMode.LOCAL:
            result = self._run_local_inference(**kwargs)
        elif self.mode == PredictorMode.REMOTE:
            result = self._run_remote_inference(**kwargs)
        else:
            raise ValueError(f"Invalid predictor mode: {self.mode}")

        return result

    def _run_local_inference(self, **kwargs):
        pass

    def _run_remote_inference(self, **kwargs) -> dict:
        try:
            responses = Generation.call(**kwargs)
        except Exception as e:
            raise e

        # TODO: output format to be confirmed
        return responses.output



# if __name__ == '__main__':
#
#     import dashscope
#     from http import HTTPStatus
#     from dashscope import Models
#     from dashscope import Generation
#
#     dashscope.base_http_api_url = 'https://int-dashscope.aliyun-inc.com/api/v1/services'
#     dashscope.api_key = 'ztPODdVunydIpfBKFASpHMTAHCzvlcu5124BE76AB9A511ED830AFA4166304A26'
#
#     responses = Generation.call(
#         model='moss_dev3',
#         prompt='推荐一个附近的公园',
#         history=[
#             {
#                 "user": "今天天气好吗？",
#                 "bot": "今天天气不错，要出去玩玩嘛？"
#             },
#             {
#                 "user": "那你有什么地方推荐？",
#                 "bot": "我建议你去公园，春天来了，花朵开了，很美丽。"
#             }
#         ],
#         max_length=500,
#         top_k=15)
#
#     if responses.code == HTTPStatus.OK:
#         print('Result is: %s' % responses.output)
#     else:
#         print('Code: %s, status: %s, message: %s' % (responses.code,
#                                                      responses.status,
#                                                      responses.message))
