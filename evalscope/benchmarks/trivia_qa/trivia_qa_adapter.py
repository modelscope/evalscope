# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI Inc, and its affiliates.
import csv
import os
from typing import List
import numpy as np

from evalscope.benchmarks.data_adapter import DataAdapter
from evalscope.metrics.metrics import exact_match, weighted_mean
from evalscope.utils.logger import get_logger
# flake8: noqa

logger = get_logger()


DATASET_ID = 'modelscope/trivia_qa'
SUBSET_LIST = ['default']


class TriviaQaAdapter(DataAdapter):

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 few_shot_num: int = None,
                 train_split: str = 'dev',
                 eval_split: str = 'test',
                 **kwargs):

        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageAccuracy', 'object': weighted_mean}]

        if few_shot_num is None:
            logger.info(f'few_shot_num is not specified for TriviaQA, use default value: 5')
            few_shot_num = 5

        super().__init__(subset_list=subset_list,
                         metric_list=metric_list,
                         few_shot_num=few_shot_num,
                         train_split=train_split,
                         eval_split=eval_split,
                         **kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}
            for split in [self.train_split, self.eval_split]:
                if os.path.exists(dataset_name_or_path):
                    file_path = os.path.join(dataset_name_or_path, f'trivia-{split}.qa.csv')
                else:
                    file_path = os.path.join(work_dir, dataset_name_or_path, f'trivia-{split}.qa.csv')
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='\t')
                        split_data = []
                        for row in reader:
                            assert len(row) == 2
                            question = row[0]
                            answers = eval(row[1])
                            split_data.append({
                                'input': [
                                    {"role": "system", "content": "Follow the given examples and answer the question."},
                                    {"role": "user", "content": question}
                                ],
                                'ideal': answers
                            })
                        data_dict[subset_name][split] = split_data

        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw input, unify the prompt format for TriviaQA benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the TriviaQA:

            {
                "input": [
                    {"role": "system", "content": "Follow the given examples and answer the question."},
                    {"role": "user", "content": "Which Lloyd Webber musical premiered in the US on 10th December 1993?"}
                ],
                "ideal": [
                    "Sunset Blvd",
                    "West Sunset Boulevard",
                    "Sunset Boulevard",
                    "Sunset Bulevard",
                    "Sunset Blvd.",
                    "sunset boulevard",
                    "sunset bulevard",
                    "west sunset boulevard",
                    "sunset blvd"
                ]
            }

        Returns:
            {'data': [(context, continuation), ...]}
        """
        def get_sys_prompt(inp: dict) -> str:
            return inp['input'][0]['content']

        prompt = get_sys_prompt(input_d)
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]
        context: str = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, include_answer=False)
        full_prompt = prompt + context

        return {'data': [full_prompt]}

    def get_gold_answer(self, input_d: dict) -> list:
        # Get the gold choice
        ans: list = input_d.get("ideal", [])
        return ans

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer.

        Args:
            result: Predicted answer from the model. A list of loglikelihood values for inputs pairs.
            raw_input_d: The raw input. A single data format of the TriviaQA:
            eval_type: The type of evaluation, e.g. 'checkpoint' or 'service' or 'custom'.

        Returns:
            The predicted answer.
        """
        if eval_type == 'checkpoint':
            return result
        elif eval_type == 'service':  # TODO: to be implemented
            return result
        elif eval_type == 'custom':  # TODO: to be implemented
            return result
        else:
            raise ValueError(f'Unknown eval_type: {eval_type}')

    def match(self, gold: list, pred: str) -> float:
        return max([exact_match(gold=ref, pred=pred) for ref in gold])

    def compute_metric(self, review_res_list: list) -> float:
        """
        Compute evaluation result by specific metric.

        Args:
            review_res_list: review score list, e.g. [0, 1, 1, 0, ...]

        Returns:
            The metric score.
        """
        items = [(score, 1.0) for score in review_res_list]
        return weighted_mean(items)

    def gen_report(self, subset_score_map: dict, report_name: str = None) -> dict:
        """
        Generate the report for the model output.

        Args:
            subset_score_map: {subset_name: (score, num), ...}
            report_name: The user-defined report name.

        Returns:
        {
            "name":"TriviaQA",
            "metric":"WeightedAverageAccuracy",
            "score":0.3389,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.3389,
                    "subset":[
                        {
                            "name":"default",
                            "score":0.3389
                        }
                    ]
                }
            ],
            "total_num":100
        }
        """
        total_num: int = sum([num for _, num in subset_score_map.values()])
        weighted_avg_acc: float = sum([score * num for score, num in subset_score_map.values()]) / total_num
        cate_avg_list = [{'name': subset_name, 'score': score} for subset_name, (score, _) in subset_score_map.items()]

        category_d = dict(name='DEFAULT',
                          score=weighted_avg_acc,
                          subset=cate_avg_list)

        res_map = dict(name=report_name or 'trivia_qa',
                       metric=self.metric_list[0]['name'],
                       score=weighted_avg_acc,
                       category=[category_d],
                       total_num=total_num)

        return res_map

    @classmethod
    def _generate_prompt(cls, input_d: dict, include_answer=True) -> str:

        example: str = f"Question: {input_d['input'][1]['content']}\nAnswer:"
        if include_answer:
            example += f" {input_d['ideal'][0]}\n\n"

        return example
