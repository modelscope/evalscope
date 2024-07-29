# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os.path

from evalscope.benchmarks.data_adapter import DataAdapter
from evalscope.metrics.metrics import bleu_ngram_one_sample, weighted_mean
from evalscope.metrics.rouge_metric import compute_rouge_score_one_sample_zh
from evalscope.utils import jsonl_to_list
from evalscope.utils.logger import get_logger
from typing import Any, Optional
from collections import defaultdict
import json

logger = get_logger()

DATASET_ID = 'general_qa'
SUBSET_LIST = ['default']


class GeneralQAAdapter(DataAdapter):
    # TODO: set few_shot_num

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 train_split: str = None,
                 eval_split: str = 'test',
                 **kwargs):
        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageBLEU', 'object': weighted_mean}]
        
        super().__init__(subset_list=subset_list,
                         metric_list=metric_list,
                         train_split=train_split,
                         eval_split=eval_split,
                         **kwargs)
    
    def load(self,
             dataset_name_or_path: str,
             subset_list: list = None,
             **kwargs) -> dict:

        data_file_list = glob.glob(os.path.join(dataset_name_or_path, '*.jsonl'))
        data_list = []

        try:
            for file_path in data_file_list:
                data_list.extend(jsonl_to_list(file_path))
        except Exception as e:
            raise ValueError(f"Failed to load data from {dataset_name_or_path}, got error: {e}")

        data_dict = {'default': {'test': data_list}}

        return data_dict
    
    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Args:
            input_d:
                format1: {'history': [['q1', 'a1'], ['q2', 'a2']], 'question': '', 'answer': ''}
                format2: {'history': [['q1', 'a1'], ['q2', 'a2']], 'query': '', 'response': ''}

        Returns:
            {'data': [prompt]}

        """
        # prompt = f"'<|im_start|>user\n{input_d['input']}<|im_end|>\n<|im_start|>assistant\n'"
        history = input_d.get('history', [])    # history: [['q1', 'a1'], ['q2', 'a2'], ...]
        if len(history) > 0:
            logger.warning(f"The history is not included in the prompt for GeneralQA. To be supported in the future.")

        prompt = input_d.get('question', '') or input_d.get('query', '')

        # if len(history) > 0:
        #     prompt = '\n'.join(history) + '\n' + prompt
        return {'data': [prompt]}
    
    def get_gold_answer(self, input_d: dict) -> str:
        """
        Args:
            input_d: {'history': [], 'question': '', 'answer': ''}

        Returns:
            gold_answer: str

        """
        return input_d.get('answer', '') or input_d.get('response', '')
    
    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Args:
            result: str

        Returns:
            pred_result: str

        """
        return result
    
    def match(self, gold: str, pred: str) -> float:
        """
        Args:
            gold: str
            pred: str

        Returns:
            bleu_score: float

        """
        item = [(gold, pred)]
        res = dict()
        rouge_dict = compute_rouge_score_one_sample_zh([pred], [gold])
        bleu_dict = bleu_ngram_one_sample(pred, gold)
        res.update(rouge_dict)
        res.update(bleu_dict)
        # return bleu(item)
        return res
    
    def compute_metric(self, review_res_list: list) -> float:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [score1, score2, ...]

        Returns:
            avg_res: float

        """
        items = defaultdict(list)
        for scores in review_res_list:
            for k,v in scores.items():
                items[k].append((v, 1.0))
        # items = [(score, 1.0) for score in review_res_list]
        res = {k: weighted_mean(v) for k,v in items.items()}
        # return weighted_mean(items)
        return res
    
    def gen_report(self, subset_score_map: dict, report_name: str = None) -> dict:
        """
        Args:
            subset_score_map: {subset_name: (score_dict, num), ...}
            report_name: str, the user-defined report name.

        Returns:
        {
            "name":"GeneralQA",
            "metric":"WeightedAverageBLEU",
            "score":0.399,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.399,
                    "subset":[
                        {
                            "name":"default",
                            "score":0.399
                        },
                    ]
                }
            ],
            "total_num":10
        }
        """
        total_num: int = sum([num for _, num in subset_score_map.values()])
        # weighted_avg_bleu: float = sum([score * num for score, num in subset_score_map.values()]) / total_num
        cate_avg_list = [{'name': subset_name, 'score': score_dict} for subset_name, (score_dict, _) in subset_score_map.items()]
        total_avg_list = defaultdict(float)
        for score_dict, num in subset_score_map.values():
            for metric, score in score_dict.items():
                total_avg_list[metric] += score * num / total_num

        category_d = dict(name="DEFAULT",
                          score=total_avg_list,
                          subset=cate_avg_list)
        
        res_map = dict(name=report_name or "general_qa",
                       metric=self.metric_list[0]['name'],
                       score=total_avg_list,
                       category=[category_d],
                       total_num=total_num)
        
        return res_map