from evalscope import TaskConfig
from evalscope.run import run_task

two_stage_task_cfg = TaskConfig(
    eval_backend='RAGEval',
    eval_config={
        'tool': 'MTEB',
        'model': [
            {
                'model_name_or_path': 'Qwen/Qwen3-Embedding-0.6B',
                'is_cross_encoder': False,
                'max_seq_length': 512,
                'model_kwargs': {'torch_dtype': 'auto'},
                'encode_kwargs': {'batch_size': 256},
            },
            {
                'model_name_or_path': 'Qwen/Qwen3-Reranker-0.6B',
                'is_cross_encoder': True,
                'max_seq_length': 2042,
                'prompt': '',
                'model_kwargs': {'torch_dtype': 'auto'},
                'encode_kwargs': {'batch_size': 256},
            },
        ],
        'eval': {
            'tasks': ['T2Retrieval'],
            'verbosity': 2,
            'overwrite_results': True,
            'top_k': 100,
            'limits': 500,
        },
    },
    work_dir='outputs'
)

if __name__ == '__main__':
    run_task(task_cfg=two_stage_task_cfg)
