# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) ZhipuAI, Inc. and its affiliates.

import os
import json
from typing import List

import torch
import numpy as np
import random
from modelscope import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from evalscope.third_party.longbench_write.utils import count_words
from evalscope.models.api import OpenaiApi
from evalscope.utils import get_logger

logger = get_logger()

DEFAULT_PROC_NUM = 8

"""
This script is used to generate predictions for the LongWriter model.
Refer to https://github.com/THUDM/LongWriter for more details.
"""


def get_pred(rank, world_size, data, path, max_new_tokens, temperature, tokenizer, fout):
    device = torch.device(f'cuda:{rank}')
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()

    for dt in tqdm(data, total=len(data), desc=f'Infer on rank-{rank}: '):
        prompt = dt['prompt']
        if "llama" in path.lower():
            prompt = f"[INST]{prompt}[/INST]"
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            output = model.generate(
                **input,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=True,
                temperature=temperature,
            )[0]
            response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            response, history = model.chat(tokenizer, prompt, history=[], max_new_tokens=max_new_tokens,
                                           temperature=temperature)
        dt["response_length"], _ = count_words(response)
        dt["response"] = response

        logger.info(dt)

        fout.write(json.dumps(dt, ensure_ascii=False) + '\n')
        fout.flush()

    logger.info(f'Successfully generated predictions for {len(data)} samples.')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# def run_infer(model: str,
#               data_path: str,
#               output_dir: str,
#               generation_kwargs: dict = None,
#               enable: bool = True, ):
#     """
#     Process inference for LongWriter model.
#
#     Args:
#         model: The model id of the LongWriter model on ModelScope, or local model path.
#         data_path: The path to the data file.
#         output_dir: The output directory for the predictions.
#         generation_kwargs: The generation arguments for the model.
#             Attributes: `max_new_tokens`: The maximum number of tokens to generate. `temperature`: The temperature
#         enable: Whether to run infer process.
#     """
#     model_id_path: str = os.path.join(output_dir, model.strip(os.sep).replace(os.sep, '__'))
#
#     if not enable:
#         logger.warning('*** Skip `infer` stage ***')
#         return f'{model_id_path}/pred.jsonl'
#
#     seed_everything(42)
#
#     os.makedirs(model_id_path, exist_ok=True)
#     fout = open(f'{model_id_path}/pred.jsonl', 'w', encoding='utf-8')
#
#     if generation_kwargs is None:
#         generation_kwargs = dict({
#             'max_new_tokens': 32768,
#             'temperature': 0.5
#         })
#
#     tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
#     world_size = torch.cuda.device_count()
#
#     logger.info(f'>>Input data path: {data_path}')
#     with open(data_path, encoding='utf-8') as f:
#         data = [json.loads(line) for line in f]
#
#     data_subsets = [data[i::world_size] for i in range(world_size)]
#     processes = []
#     for rank in range(world_size):
#         p = mp.Process(target=get_pred,
#                        args=(rank, world_size, data_subsets[rank], model, generation_kwargs.get('max_new_tokens'), generation_kwargs.get('temperature'), tokenizer, fout))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     logger.info(f'Finish generating predictions for {model}.')
#     logger.info(f'Predictions are saved in {model_id_path}/pred.jsonl.')
#
#     return f'{model_id_path}/pred.jsonl'


def run_infer(model: str,
              data_path: str,
              output_dir: str,
              api_config: dict,
              generation_kwargs: dict = None,
              enable: bool = True, ):
    """
    Process inference for LongWriter model.

    Args:
        model: The model id of the LongWriter model on ModelScope, or local model path.
        data_path: The path to the data file.
        output_dir: The output directory for the predictions.
        api_config: The configuration for the OpenAI API inference.
            Attributes:
                `openai_api_key`: The OpenAI API key. Default is None for custom model serving.
                `openai_api_base`: The OpenAI API base URL.
                `is_chat`: Whether to chat. Default is True.
                `verbose`: Whether to print verbose information. Default is False.
        generation_kwargs: The generation arguments for the model.
            Attributes: `max_new_tokens`: The maximum number of tokens to generate. `temperature`: The temperature
        enable: Whether to run infer process.
    """
    model_id_path: str = os.path.join(output_dir, model.strip(os.sep).replace(os.sep, '__'))

    if not enable:
        logger.warning('*** Skip `infer` stage ***')
        return f'{model_id_path}/pred.jsonl'

    seed_everything(42)

    if generation_kwargs is None:
        generation_kwargs = dict({
            'max_new_tokens': 32768,
            'temperature': 0.5,
            'repetition_penalty': 1.0,
        })

    # Prepare inputs
    logger.info(f'>>Input data path: {data_path}')
    # TODO: add load data from ms
    with open(data_path, encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]

    logger.info(f'Input example: {data_list[0]}')

    api_client = OpenaiApi(model=model,
                           openai_api_key=None,
                           openai_api_base=api_config.get('openai_api_base', 'http://127.0.0.1:8000/v1/chat/completions'),
                           max_new_tokens=generation_kwargs.get('max_new_tokens', 4096),
                           temperature=generation_kwargs.get('temperature', 0.0),
                           repetition_penalty=generation_kwargs.get('repetition_penalty', 1.0),
                           is_chat=api_config.get('is_chat', True),
                           verbose=api_config.get('verbose', False),
                           )

    # TODO: ONLY FOR TEST  generate_simple
    results: List[str] = api_client.generate_simple(inputs=[example['prompt'] for example in data_list])
    assert len(results) == len(data_list), f'Error: The number of predictions {len(results)} is not equal to the number of inputs {len(data_list)}.'
    logger.info(f'Finish generating predictions with {len(data_list)} samples for {model}')

    # Outputs
    os.makedirs(model_id_path, exist_ok=True)
    output_pred_file: str = f'{model_id_path}/pred.jsonl'
    with open(output_pred_file, 'w', encoding='utf-8') as f:
        for dt, res in zip(data_list, results):
            dt["response_length"], _ = count_words(res)
            dt["response"] = res
            f.write(json.dumps(dt, ensure_ascii=False) + '\n')

    logger.info(f'Predictions are saved in {output_pred_file}')

    return output_pred_file


if __name__ == '__main__':
    # ZhipuAI/LongWriter-glm4-9b, ZhipuAI/LongWriter-llama3.1-8b
    api_config = dict(openai_api_key=None,
                      openai_api_base='http://127.0.0.1:8000/v1/chat/completions',
                      is_chat=True,
                      verbose=True,)

    run_infer(model='ZhipuAI/LongWriter-glm4-9b',
              data_path='resources/longbench_write.jsonl',
              output_dir='outputs',
              api_config=api_config,
              generation_kwargs=dict({
                      'max_new_tokens': 32768,
                      'temperature': 0.5})
              )
