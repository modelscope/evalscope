# Copyright (c) Alibaba, Inc. and its affiliates.

import time

from llmuses.models.custom import CustomModel
from llmuses.run import run_task


class SwiftModel(CustomModel):

    def __init__(self, config: dict, **kwargs):
        if len(config) == 0:
            config: dict = {'model_id': 'swift_dummy_model_0308'}

        # TODO:  swift model implementation
        ####  swift model implementation  ####
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        from swift.llm import (
            get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
        )
        from swift.utils import seed_everything

        model_type = ModelType.qwen_7b_chat
        template_type = get_default_template_type(model_type)
        print(f'template_type: {template_type}')  # template_type: qwen

        kwargs = {}
        # kwargs['use_flash_attn'] = True  # 使用flash_attn
        self.model, self.tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)
        # 修改max_new_tokens
        self.model.generation_config.max_new_tokens = 128

        self.template = get_template(template_type, self.tokenizer)
        seed_everything(42)

        self.inference = inference

        ####  swift model implementation  ####

        super(SwiftModel, self).__init__(config=config, **kwargs)

    def predict(self, prompt: str, **kwargs):
        # TODO

        # query = '浙江的省会在哪里？'
        response, history = self.inference(self.model, self.template, prompt)
        response: str = str(response)

        res_d: dict = {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        # 'content': f'The answer is B. Raw prompt: {prompt}',
                        'content': response,
                        'role': 'assistant'
                    }
                }
            ],
            'created': time.time(),
            'model': self.config.get('model_id'),           # should be model_id
            'object': 'chat.completion',
            'usage': {
                'completion_tokens': 0,
                'prompt_tokens': 0,
                'total_tokens': 0
            }
        }

        return res_d


def get_task_cfg(model_instance: CustomModel):
    from llmuses.constants import DEFAULT_ROOT_CACHE_DIR

    # config 示例
    swift_task_cfg = {
        'model_args': {},
        'generation_config': {},
        'dataset_args': {},
        'dry_run': False,
        'model': model_instance,    # NOTE: model_id or # model_dir or model_instance(CustomModel)
        'eval_type': 'custom',      # NOTE: `checkpoint` or `custom` or `service`
        'datasets': ['arc'],
        'work_dir': DEFAULT_ROOT_CACHE_DIR,
        'outputs': './outputs/eval_swift_dummy',
        'mem_cache': False,
        'dataset_hub': 'ModelScope',
        'dataset_dir': DEFAULT_ROOT_CACHE_DIR,
        'stage': 'all',
        'limit': 10,
        'debug': False
    }

    return swift_task_cfg


if __name__ == '__main__':

    swift_model = SwiftModel(config={})
    task_cfg = get_task_cfg(model_instance=swift_model)
    run_task(task_cfg=task_cfg)
