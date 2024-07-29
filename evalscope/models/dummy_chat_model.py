# Copyright (c) Alibaba, Inc. and its affiliates.

import random
import time
from evalscope.models import ChatBaseModel
from evalscope.utils.logger import get_logger

logger = get_logger()


class DummyChatModel(ChatBaseModel):

    MODEL_ID = 'dummy_chat_model_0801'
    REVISION = 'v1.0.0'

    def __init__(self, model_cfg: dict, **kwargs):
        model_cfg['model_id'] = self.MODEL_ID
        model_cfg['revision'] = self.REVISION
        super(DummyChatModel, self).__init__(model_cfg=model_cfg)

    def predict(self, inputs: dict, **kwargs) -> dict:

        debug: bool = False
        if debug:
            messages = inputs['messages']
            history = inputs['history']

            logger.info(f'** messages: {messages}')
            logger.info(f'** history: {history}')

        choice = random.choice(['A', 'B', 'C', 'D'])

        # Build response
        res = {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'content': choice,
                        'role': 'assistant'
                    }
                }
            ],
            'created': time.time(),
            'model': self.MODEL_ID + '-' + self.REVISION,
            'object': 'chat.completion',
            'usage': {}
        }

        return res
