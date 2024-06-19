# Copyright (c) Alibaba, Inc. and its affiliates.

"""
1. Installation TODO - to be updated
eval-scope分支： https://github.com/modelscope/eval-scope/tree/dev/add_oc
opencompass代码： https://github.com/wangxingjun778/opencompass/tree/dev
pip3 install -e .

2. Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip

3. Deploy model serving

4. Run eval task
"""
from llmuses.backend.opencompass import OpenCompassBackendManager


def run_swift_eval():

    # List all datasets
    print(OpenCompassBackendManager.list_datasets())

    # Run task
    oc_backend_manager = OpenCompassBackendManager(
        config={'datasets': ['mmlu', 'ceval', 'ARC_c', 'gsm8k'],
                'models': [
                    {'path': 'qwen-7b-chat', 'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions'},
                ]}
    )

    oc_backend_manager.run()


if __name__ == '__main__':
    run_swift_eval()
