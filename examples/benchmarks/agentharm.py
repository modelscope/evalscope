"""Run the public harmful/benign pair with native mock tools and one configured judge."""

import os

from evalscope import TaskConfig, run_task

if __name__ == '__main__':
    key = os.environ['DASHSCOPE_API_KEY']
    run_task(
        TaskConfig(
            model='qwen3.8-flash',
            eval_type='openai_api',
            api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key=key,
            datasets=['agentharm', 'agentharm_benign'],
            # The default uses the pinned ModelScope mirror; select huggingface explicitly if needed.
            dataset_args={
                'agentharm': {'subset_list': ['Drugs'], 'extra_params': {'behavior_ids': ['9-1']}},
                'agentharm_benign': {'subset_list': ['Drugs'], 'extra_params': {'behavior_ids': ['9-1']}},
            },
            agent_config={'max_steps': 20},
            generation_config={'temperature': 0, 'max_tokens': 4096, 'seed': 0},
            judge={
                'models': [
                    {
                        'model_id': 'qwen3.8-flash',
                        'eval_type': 'openai_api',
                        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                        'api_key': key,
                        'generation_config': {'temperature': 0},
                    }
                ],
                'position_swap': 'off',
            },
            limit=1,
            eval_batch_size=2,
        )
    )
