# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation TODO - to be updated
eval-scope分支： https://github.com/modelscope/eval-scope/tree/dev/add_oc
opencompass代码： https://github.com/wangxingjun778/opencompass/tree/dev
pip3 install -e .

Note: pip3 uninstall ms-opencompass

2. Download dataset to data/ folder     TODO
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip

3. Deploy model serving

4. Run eval task
"""
from llmuses.backend.opencompass import OpenCompassBackendManager
from llmuses.run import run_task


def run_swift_eval():

    # List all datasets
    print(f'** all datasets: {OpenCompassBackendManager.list_datasets()}')

    # Prepare the config
    """
    Attributes:
        `eval_backend`: Default to 'OpenCompass'
        `datasets`: list, refer to `OpenCompassBackendManager.list_datasets()`
        `models`: list of dict, each dict must contain `path` and `openai_api_base` 
                `path`: reuse the value of '--model_type' in the command line `swift deploy`
                `openai_api_base`: the base URL of swift model serving
                
                Refer to `opencompass.cli.arguments.ModelConfig` for other optional attributes.
    """
    # Option 1: Use dict format
    task_cfg = dict(
        eval_backend='OpenCompass',
        eval_config={'datasets': ['mmlu', 'ceval', 'ARC_c', 'gsm8k'],
                     'models': [{'path': 'qwen-7b-chat', 'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions'}, ]
                     },
    )

    # Option 2: Use yaml file
    # task_cfg = 'llmuses/examples/tasks/default_eval_swift_openai_api.yaml'

    # Option 3: Use json file
    # task_cfg = 'llmuses/examples/tasks/default_eval_swift_openai_api.json'

    # Run task
    run_task(task_cfg=task_cfg)


if __name__ == '__main__':
    run_swift_eval()
