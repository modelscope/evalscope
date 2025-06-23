# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from typing import List

from evalscope.models import CustomModel
from evalscope.utils.logger import get_logger

logger = get_logger()


class DummyCustomModel(CustomModel):

    def __init__(self, config: dict = {}, **kwargs):
        super(DummyCustomModel, self).__init__(config=config, **kwargs)

    def make_request_messages(self, input_item: dict) -> list:
        """
        Make request messages for OpenAI API.
        """
        if input_item.get('messages', None):
            return input_item['messages']

        data: list = input_item['data']
        if isinstance(data[0], tuple):  # for truthful_qa and hellaswag
            query = '\n'.join(''.join(item) for item in data)
            system_prompt = input_item.get('system_prompt', None)
        else:
            query = data[0]
            system_prompt = input_item.get('system_prompt', None)

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': query})

        return messages

    def predict(self, prompts: List[dict], **kwargs):
        original_inputs = kwargs.get('origin_inputs', None)
        infer_cfg = kwargs.get('infer_cfg', None)

        logger.debug(f'** Prompts: {prompts}')
        if original_inputs is not None:
            logger.debug(f'** Original inputs: {original_inputs}')
        if infer_cfg is not None:
            logger.debug(f'** Inference config: {infer_cfg}')

        # Simulate a response based on the prompts
        # Must return a list of dicts with the same format as the OpenAI API.
        responses = []
        for input_item in original_inputs:
            message = self.make_request_messages(input_item)
            response = f'Dummy response for prompt: {message}'

            res_d = {
                'choices': [{
                    'index': 0,
                    'message': {
                        'content': response,
                        'role': 'assistant'
                    }
                }],
                'created': time.time(),
                'model': self.config.get('model_id'),
                'object': 'chat.completion',
                'usage': {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0
                }
            }

            responses.append(res_d)

        return responses


if __name__ == '__main__':
    from evalscope import TaskConfig, run_task

    dummy_model = DummyCustomModel()
    task_config = TaskConfig(
        model=dummy_model,
        model_id='evalscope-model-dummy',
        datasets=['gsm8k'],
        eval_type='custom',  # must be custom for custom model evaluation
        generation_config={
            'max_new_tokens': 100,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 50,
            'repetition_penalty': 1.0
        },
        debug=True,
        limit=5,
    )

    eval_results = run_task(task_cfg=task_config)
