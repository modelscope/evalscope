# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) ZhipuAI, Inc. and its affiliates.
import multiprocessing
import os
import json
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm

from evalscope.utils import jsonl_to_list
from evalscope.utils import get_logger

logger = get_logger()

"""
This script is used to evaluate results of predictions for the LongWriter model.
Refer to https://github.com/THUDM/LongWriter for more details.

EvalLength:
    Evaluate the length of the generated responses.
    Metrics:
        score_l: The average score of the length evaluation.

EvalQuality:
    Evaluate the quality of the generated responses by using Judge Model.
    Metrics:
        score_q: The average score of the quality evaluation.
"""


class EvalLength:

    EVAL_L = 'eval_length'

    def __init__(self, model_id_or_path: str, pred_path: str, output_dir: str):
        self.model_id_or_path = model_id_or_path
        self.pred_path = pred_path
        self.output_dir = output_dir

        self.model_id_path = self.model_id_or_path.replace('/', '__')

    def score(self, x, y):
        if y > x:
            return 100 * max(0, 1. - (y / x - 1) / 3)
        else:
            return 100 * max(0, 1. - (x / y - 1) / 2)

    def eval(self):
        predictions = [json.loads(line) for line in open(self.pred_path, encoding='utf-8')]
        x, y, scores = [], [], []

        for pred in tqdm(predictions, total=len(predictions), desc=f'Process of eval_l: '):
            x.append(pred["length"])
            y.append(pred["response_length"])
            scores.append(self.score(pred["length"], pred["response_length"]))

        avg_score_l = np.mean(scores)
        logger.info(f'Average score of length evaluation: {avg_score_l:.2f}')

        # Dump to output file
        output_res_path = f'{self.output_dir}/{self.model_id_path}/{self.EVAL_L}.jsonl'
        with open(output_res_path, 'w') as f:
            f.write(json.dumps({'score_l': avg_score_l, 'scores': scores}, ensure_ascii=False) + '\n')
            logger.info(f'Successfully dumped evaluation results to {output_res_path}')

        return x, y, scores

    def plot(self, x: list, y: list):

        # set plt size 6x6
        plt.figure(figsize=(6, 6))
        lmt = 25000
        # plot x, y
        plt.scatter(x, y, s=100, c='r', alpha=0.3)
        # plot x=y
        plt.plot([0, lmt], [0, lmt], 'k--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(50, lmt)
        plt.ylim(50, lmt)
        plt.xlabel('Required Length', fontsize=20, fontweight='bold')
        plt.ylabel('Output Length', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()

        output_pic_path = f'{self.output_dir}/{self.model_id_path}/eval_length_scatter.png'
        plt.savefig(output_pic_path)
        logger.info(f'Successfully saved scatter plot to {output_pic_path}')


class EvalQuality:

        EVAL_Q = 'eval_quality'
        OPENAI_BASE_URL = 'https://api.openai.com/v1/chat/completions'
        DIMS = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience"]

        def __init__(self,
                     model_id_or_path: str,
                     pred_path: str,
                     output_dir: str,
                     prompt_template_path: str,
                     openai_api_key: str = None,
                     openai_gpt_model: str = 'gpt-4o-2024-05-13',
                     generation_kwargs: dict = None,
                     proc_num: int = 8):

            self.model_id_or_path = model_id_or_path
            self.pred_path = pred_path
            self.output_dir = output_dir
            self.proc_num = proc_num
            self.eval_scores = []

            assert os.path.exists(self.pred_path), f'Prediction file not found: {self.pred_path}'

            # Default: temperature=0.5, max_new_tokens=1024, stop=None
            if generation_kwargs is None:
                self.generation_kwargs = dict({
                    'max_new_tokens': 1024,
                    'temperature': 0.5,
                    'stop': None,
                })
            else:
                self.generation_kwargs = generation_kwargs

            self.prompt_template: str = open(prompt_template_path, 'r', encoding='utf-8').read()

            self.model_id_path = self.model_id_or_path.replace('/', '__')
            self.output_res_path = f'{self.output_dir}/{self.model_id_path}/{self.EVAL_Q}.jsonl'

            self.openai_api_key: str = openai_api_key
            self.openai_gpt_model = openai_gpt_model
            assert self.openai_api_key, 'Please set `OPENAI_API_KEY` in environment variables.'

        def get_response_gpt4(self, prompt, temperature=0.5, max_new_tokens=1024, stop=None):
            tries = 0
            while tries < 10:
                tries += 1
                try:
                    headers = {
                        'Authorization': "Bearer {}".format(self.openai_api_key),
                    }
                    messages = [
                        {'role': 'user', 'content': prompt},
                    ]
                    resp = requests.post(self.OPENAI_BASE_URL, json={
                        "model": self.openai_gpt_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_new_tokens,
                        "stop": stop,
                    }, headers=headers, timeout=600)
                    if resp.status_code != 200:
                        raise Exception(resp.text)
                    resp = resp.json()
                    break
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    if "maximum context length" in str(e):
                        raise e
                    elif "triggering" in str(e):
                        return 'Trigger OpenAI\'s content management policy'
                    logger.error("Error Occurs: \"%s\"        Retry ..." % (str(e)))
            else:
                logger.error("Max tries. Failed.")
                return "Max tries. Failed."
            try:
                return resp["choices"][0]["message"]["content"]
            except:
                return ''

        @staticmethod
        def extract_info(pattern, text):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
            else:
                return None

        def process_data(self, items):
            for item in tqdm(items, total=len(items), desc=f'Process of eval_q: '):
                prompt = self.prompt_template.replace('$INST$', item['prompt']).replace('$RESPONSE$', item["response"])
                scores = None
                trys = 0
                while scores is None and trys < 5:
                    output = self.get_response_gpt4(prompt, **self.generation_kwargs)
                    try:
                        if '```json' in output:
                            output = self.extract_info(r'```json\n(.*?)\n```', output)
                        output = output.replace('\n', '')
                        scores = json.loads(output)
                        for dim in self.DIMS:
                            if dim not in scores:
                                scores = None
                                trys += 1
                    except Exception as e:
                        trys += 1
                        logger.error(f'Error occurs: {str(e)} Retry ...')
                if scores is None:
                    logger.error(f'Failed to extract scores for item: {item}')
                else:
                    item['scores'] = scores
                    self.eval_scores.append(item)

        def eval(self):

            data = jsonl_to_list(self.pred_path)
            total = len(data)
            assert total > 0, f'No data found in prediction file: {self.pred_path}'

            random.shuffle(data)
            pool = multiprocessing.Pool(processes=self.proc_num)

            for i in range(self.proc_num):
                start = (i * total) // self.proc_num
                end = None if i == self.proc_num - 1 else ((i + 1) * total) // self.proc_num
                pool.apply_async(self.process_data, args=(data[start:end],))

            pool.close()
            pool.join()

            total_score = dict()
            for dim in self.DIMS:
                scores = [float(score[dim]) if dim in score else 3 for score in self.eval_scores]
                total_score[dim] = ((sum(scores) / len(scores)) - 1) * 25
            total_score['total'] = sum(total_score.values()) / len(total_score)
            logger.info(f'Total score of quality evaluation: {total_score["total"]:.2f}')

            output_res_path: str = f'{self.output_dir}/{self.model_id_path}/{self.EVAL_Q}.jsonl'
            with open(output_res_path, 'w', encoding='utf-8') as fout:
                fout.write(json.dumps(total_score, ensure_ascii=False) + '\n')


def run_eval(model_id_or_path: str,
             pred_path: str,
             output_dir: str,
             prompt_template_path: str,
             openai_api_key: str,
             openai_gpt_model: str,
             generation_kwargs: dict,
             proc_num: int,
             stage: list,
             ):
    logger.info(f'Got eval stages: {stage}')

    if 'eval_l' in stage:
        try:
            logger.info(f'Processing evaluation of length for model: {model_id_or_path}')
            eval_length = EvalLength(model_id_or_path=model_id_or_path,
                                     pred_path=pred_path,
                                     output_dir=output_dir)
            x, y, _ = eval_length.eval()
            eval_length.plot(x, y)
        except Exception as e:
            logger.error(f'Error occurs during evaluating length: {str(e)}')
    else:
        logger.warning(f'*** Skip `eval_l` stage ***')

    if 'eval_q' in stage:
        try:
            logger.info(f'Processing evaluation of quality for model: {model_id_or_path}')
            eval_quality = EvalQuality(model_id_or_path=model_id_or_path,
                                       pred_path=pred_path,
                                       output_dir=output_dir,
                                       prompt_template_path=prompt_template_path,
                                       openai_api_key=openai_api_key,
                                       openai_gpt_model=openai_gpt_model,
                                       generation_kwargs=generation_kwargs,
                                       proc_num=proc_num)
            eval_quality.eval()
        except Exception as e:
            logger.error(f'Error occurs during evaluating quality: {str(e)}')
    else:
        logger.warning('*** Skip `eval_q` stage ***')
