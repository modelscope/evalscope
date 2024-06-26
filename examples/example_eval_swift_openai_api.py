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
from llmuses.summarizer import Summarizer


def run_swift_eval():

    # List all datasets
    # e.g.  ['mmlu', 'WSC', 'DRCD', 'chid', 'gsm8k', 'AX_g', 'BoolQ', 'cmnli', 'ARC_e', 'ocnli_fc', 'summedits', 'MultiRC', 'GaokaoBench', 'obqa', 'math', 'agieval', 'hellaswag', 'RTE', 'race', 'flores', 'ocnli', 'strategyqa', 'triviaqa', 'WiC', 'COPA', 'commonsenseqa', 'piqa', 'nq', 'mbpp', 'csl', 'Xsum', 'CB', 'tnews', 'ARC_c', 'afqmc', 'eprstmt', 'ReCoRD', 'bbh', 'TheoremQA', 'CMRC', 'AX_b', 'siqa', 'storycloze', 'humaneval', 'cluewsc', 'winogrande', 'lambada', 'ceval', 'bustm', 'C3', 'lcsts']
    print(f'** All datasets from OpenCompass backend: {OpenCompassBackendManager.list_datasets()}')

    # Prepare the config
    """
    Attributes:
        `eval_backend`: Default to 'OpenCompass'
        `datasets`: list, refer to `OpenCompassBackendManager.list_datasets()`
        `models`: list of dict, each dict must contain `path` and `openai_api_base` 
                `path`: reuse the value of '--model_type' in the command line `swift deploy`
                `openai_api_base`: the base URL of swift model serving
        `work_dir`: str, the directory to save the evaluation results、logs and summaries. Default to 'outputs/default'
                
        Refer to `opencompass.cli.arguments.ModelConfig` for other optional attributes.
    """
    # Option 1: Use dict format
    task_cfg = dict(
        eval_backend='OpenCompass',
        eval_config={'datasets': ['gsm8k'],       # ['mmlu', 'ceval', 'ARC_c', 'gsm8k']
                     'models': [{'path': 'llama3-8b-instruct', 'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions'}, ],
                     'work_dir': 'outputs/llama3_eval_result',
                     },
    )

    # Option 2: Use yaml file
    # task_cfg = 'examples/tasks/default_eval_swift_openai_api.yaml'

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
