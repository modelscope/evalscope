# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) ZhipuAI, Inc. and its affiliates.

import os
import json
import time

import torch
import numpy as np
import random
import re
import torch.multiprocessing as mp
from modelscope import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from evalscope.utils import get_logger

logger = get_logger()

"""
This script is used to generate predictions for the LongWriter model.
Refer to https://github.com/THUDM/LongWriter for more details.
"""


def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)

    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)

    total_count = chinese_char_count + english_word_count

    return total_count


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
        dt["response_length"] = count_words(response)
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


def run_infer(model_id_or_path: str,
              data_path: str,
              output_dir: str,
              generation_kwargs: dict = None,
              enable: bool = True, ):
    """
    Process inference for LongWriter model.

    Args:
        model_id_or_path: The model id of the LongWriter model on ModelScope.
        data_path: The path to the data file.
        output_dir: The output directory for the predictions.
        generation_kwargs: The generation arguments for the model.
            Attributes: `max_new_tokens`: The maximum number of tokens to generate. `temperature`: The temperature
        enable: Whether to run infer process.
    """
    model_id_path: str = os.path.join(output_dir, model_id_or_path.replace('/', '__'))

    if not enable:
        logger.warning('*** Skip `infer` stage ***')
        return f'{model_id_path}/pred.jsonl'

    seed_everything(42)

    os.makedirs(model_id_path, exist_ok=True)
    fout = open(f'{model_id_path}/pred.jsonl', 'w', encoding='utf-8')

    if generation_kwargs is None:
        generation_kwargs = dict({
            'max_new_tokens': 32768,
            'temperature': 0.5
        })

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    world_size = torch.cuda.device_count()

    logger.info(f'>>Input data path: {data_path}')
    with open(data_path, encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    data_subsets = [data[i::world_size] for i in range(world_size)]
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=get_pred,
                       args=(rank, world_size, data_subsets[rank], model_id_or_path, generation_kwargs.get('max_new_tokens'), generation_kwargs.get('temperature'), tokenizer, fout))
        p.start()
        processes.append(p)

    # count = mp.Value('i', 0)    # 一个共享的整数变量
    # with tqdm(total=len(data)) as pbar:
    #     while any(p.is_alive() for p in processes):
    #         with count.get_lock():
    #             pbar.n = count.value
    #         pbar.refresh()
    #         time.sleep(0.1)

    for p in processes:
        p.join()

    logger.info(f'Finish generating predictions for {model_id_or_path}.')
    logger.info(f'Predictions are saved in {model_id_path}/pred.jsonl.')

    return f'{model_id_path}/pred.jsonl'


if __name__ == '__main__':
    # ZhipuAI/LongWriter-glm4-9b, ZhipuAI/LongWriter-llama3.1-8b
    run_infer(model_id_or_path='ZhipuAI/LongWriter-glm4-9b',
              data_path='resources/longbench_write.jsonl',
              output_dir='outputs',
              generation_kwargs=dict({
                      'max_new_tokens': 32768,
                      'temperature': 0.5})
              )
