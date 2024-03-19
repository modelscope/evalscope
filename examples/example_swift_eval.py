# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time

from llmuses.models.custom import CustomModel
from llmuses.run import run_task
from llmuses.constants import DEFAULT_ROOT_CACHE_DIR
from llmuses.utils import yaml_to_dict
from llmuses.summarizer import Summarizer
from llmuses.utils.logger import get_logger

logger = get_logger()


class SwiftModel(CustomModel):

    def __init__(self, config: dict, **kwargs):

        # TODO:  swift model implementation
        ####  swift model implementation  ####
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

        # query = '浙江的省会在哪里？'
        response, history = self.inference(self.model, self.template, prompt)
        response: str = str(response)

        # ONLY FOR TEST
        # response = 'The answer is C. NOTE: ONLY FOR TEST'

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


def get_task_cfg(cfg_file: str, model_instance: CustomModel):

    if cfg_file:
        cfg_file: str = os.path.abspath(cfg_file)
        logger.info(f'Loading task config from {cfg_file}')
        task_cfg_d: dict = yaml_to_dict(yaml_file=cfg_file)
        task_cfg_d.update({'model': model_instance})
        logger.info(f'**Task config: {task_cfg_d}')
    else:
        # 默认config 示例
        task_cfg_d = {
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

    return task_cfg_d


if __name__ == '__main__':

    task_cfg_file: str = '../tasks/eval_qwen-7b-chat_v100.yaml'

    # `model_id` is required in config for CustomModel, e.g. swift_qwen-7b-chat_v100
    swift_model = SwiftModel(config={'model_id': 'swift_qwen-7b-chat_v100'})
    task_cfg = get_task_cfg(cfg_file=task_cfg_file, model_instance=swift_model)
    run_task(task_cfg=task_cfg)

    report_list: list = Summarizer.get_report_from_cfg(cfg_file=task_cfg_file)
    print(f'*** Final report ***\n {report_list}\n')
