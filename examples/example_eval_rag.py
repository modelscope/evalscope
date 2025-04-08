# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

gpt_4_o = {
    'model_name': 'gpt-4o',
    'api_base': 'http://localhost:8088/v1',
    'api_key': 'EMPTY',
}

qwen2 = {
    'model_name': 'qwen2.5',
    'api_base': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
}


def run_eval():
    generate_testset_task_cfg = {
        'eval_backend': 'RAGEval',
        'eval_config': {
            'tool': 'RAGAS',
            'testset_generation': {
                'docs': ['README.md'],
                'test_size': 10,
                'output_file': 'outputs/testset_chinese.json',  # json file
                'generator_llm': {
                    'model_name_or_path': 'Qwen/Qwen2.5-72B-Instruct',
                },
                'embeddings': {
                    'model_name_or_path': 'AI-ModelScope/m3e-base',
                },
            },
        },
    }

    generate_zh_cfg = {
        'eval_backend': 'RAGEval',
        'eval_config': {
            'tool': 'RAGAS',
            'testset_generation': {
                'docs': ['test_zh.md'],
                'test_size': 10,
                'output_file': 'outputs/testset_chinese.json',  # json file
                'generator_llm': gpt_4_o,
                'embeddings': {
                    'model_name_or_path': 'AI-ModelScope/bge-large-zh',
                },
                'language': 'chinese',
            },
        },
    }

    eval_task_cfg = {
        'eval_backend': 'RAGEval',
        'eval_config': {
            'tool': 'RAGAS',
            'eval': {
                'testset_file': 'outputs/testset_chinese_with_answer.json',
                'critic_llm': qwen2,
                'embeddings': {
                    'model_name_or_path': 'AI-ModelScope/bge-large-zh',
                },
                'metrics': [
                    'Faithfulness',
                    'AnswerRelevancy',
                    'ContextPrecision',
                    'AnswerCorrectness',
                ],
                'language': 'chinese',
            },
        },
    }

    multi_modal_eval_task_cfg = {
        'eval_backend': 'RAGEval',
        'eval_config': {
            'tool': 'RAGAS',
            'eval': {
                'testset_file': 'outputs/testset_multi_modal.json',
                'critic_llm': gpt_4_o,
                'embeddings': {
                    'model_name_or_path': 'AI-ModelScope/bge-large-zh',
                },
                'metrics': [
                    'MultiModalFaithfulness',
                    'MultiModalRelevance',
                ],
                'language': 'chinese',
            },
        },
    }

    # Run task
    # run_task(task_cfg=eval_task_cfg)
    # or
    run_task(task_cfg=generate_testset_task_cfg)


if __name__ == '__main__':
    run_eval()
