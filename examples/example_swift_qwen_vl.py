# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation
eval-scope: pip install llmuses>=0.4.0
ms-vlmeval: pip install ms-vlmeval

2. Deploy judge model

3. Run eval task
"""
from llmuses.backend.vlm_eval_kit import VLMEvalKitBackendManager
from llmuses.run import run_task
from llmuses.summarizer import Summarizer


def run_swift_eval():

    # List all datasets
    print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_VLMs().keys()}')
    print(f'** All datasets from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_datasets()}')

    # Prepare the config

    # Option 1: Use dict format
    # task_cfg = {'eval_backend': 'VLMEvalKit',
    #             'eval_config': {'LOCAL_LLM': 'models/Qwen2-7B-Instruct',
    #                             'OPENAI_API_BASE': 'http://localhost:8866/v1/chat/completions',
    #                             'OPENAI_API_KEY': 'EMPTY',
    #                             'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
    #                             'limit': 100,
    #                             'mode': 'all',
    #                             'model': [{'name': 'qwen_chat',
    #                                         'path': '../models/Qwen-VL-Chat',
    #                                         'type': 'qwen-vl-chat'},
    #                                     {'name': 'cogvlm-chat',
    #                                         'path': '../models/cogvlm-chat',
    #                                         'type': 'cogvlm-chat'}],
    #                             'rerun': True,
    #                             'work_dir': 'output'}}

    # Option 2: Use yaml file
    task_cfg = "examples/tasks/eval_swift_qwen_vl.yaml"

    # Option 3: Use json file
    # task_cfg = 'examples/tasks/default_eval_swift_openai_api.json'

    # Run task
    run_task(task_cfg=task_cfg)

    # [Optional] Get the final report with summarizer
    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>>The report list: {report_list}')


if __name__ == '__main__':
    run_swift_eval()
