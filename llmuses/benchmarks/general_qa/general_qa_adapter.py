# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.benchmarks.data_adapter import DataAdapter
from llmuses.metrics.metrics import bleu_ngram_one_sample, weighted_mean
from llmuses.metrics.rouge_metric import compute_rouge_score_one_sample_zh
from llmuses.utils.logger import get_logger
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
        data_dict = {}

        split_list = [split for split in [self.train_split, self.eval_split] if split is not None]
        for sub_name in subset_list:
            data_dict[sub_name] = {}

            try:
                with open(dataset_name_or_path, 'r', encoding='utf-8') as f:
                    # data = json.load(f)
                    data = [json.loads(line) for line in f.readlines()]
            except Exception as e:
                raise e
            
            for split in split_list:
                dataset = data
                data_dict[sub_name].update({split: dataset})

        return data_dict
    
    def gen_prompt(self, input_d: list, subset_name: str, few_shot_list: list, user_prompt: dict, **kwargs) -> dict:
        """
        Args:
            input_d: [{'question': '', 'answer': ''},{'question': '', 'answer': ''},...]

        Returns:
            {'data': [prompt]}

        """
        # prompt = f"'<|im_start|>user\n{input_d['input']}<|im_end|>\n<|im_start|>assistant\n'"
        system_prompt = user_prompt.get("system_prompt", "")
        human_prefix = user_prompt.get("human_prefix", "Human: ")
        assistant_prefix = user_prompt.get("assistant_prefix", "Assistant: ")
        seperator = user_prompt.get("seperator", "\n")

        prompt = f"{system_prompt}{seperator}"
        for qa in input_d[:-1]:
            prompt += f"{human_prefix}{qa['question']}{seperator}{assistant_prefix}{qa['answer']}{seperator}"
        prompt += f"{human_prefix}{input_d[-1]['question']}{seperator}{assistant_prefix}"

        return {'data': [prompt]}
    
    def get_gold_answer(self, input_d: list) -> str:
        """
        Args:
            input_d: [{'question': '', 'answer': ''},{'question': '', 'answer': ''},...]

        Returns:
            gold_answer: str

        """
        return input_d[-1].get('answer', '')
    
    def parse_pred_result(self, result: str, raw_input_d: dict = None) -> str:
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
    
    def gen_report(self, subset_score_map: dict) -> dict:
        """
        Args:
            subset_score_map: {subset_name: (score_dict, num), ...}

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
        
        res_map = dict(name="GeneralQA",
                       metric=self.metric_list[0]['name'],
                       score=total_avg_list,
                       category=[category_d],
                       total_num=total_num)
        
        return res_map