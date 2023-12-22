# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random

from llmuses.benchmarks.data_adapter import DataAdapter
from llmuses.constants import AnswerKeys
from llmuses.metrics.metrics import exact_match, weighted_mean
from llmuses.utils.logger import get_logger
# flake8: noqa

logger = get_logger()

DATASET_ID = 'modelscope/bbh'


# The free form subset list of BBH dataset
SUBSET_LIST = [
    'multistep_arithmetic_two',
    'navigate',
    'dyck_languages',
    'word_sorting',
    'sports_understanding',
    'boolean_expressions',
    'object_counting',
    'formal_fallacies',
    'causal_judgement',
    'web_of_lies',
]


class BBHAdapter(DataAdapter):
    """
    Adapter for BBH free-form and multiple-choices sub-tasks.  # TODO: add multiple choice support
    """

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 few_shot_num: int = 3,     # 3-shot with CoT by system
                 train_split: str = None,
                 eval_split: str = 'test',
                 **kwargs):

        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageAccuracy', 'object': weighted_mean}]

        if few_shot_num != 3:
            logger.warning(f'BBHAdapter: few_shot_num is set to {few_shot_num}, but the BBH dataset uses 3-shot with CoT by system.')

        super().__init__(subset_list=subset_list,
                         metric_list=metric_list,
                         few_shot_num=few_shot_num,
                         train_split=train_split,
                         eval_split=eval_split,
                         **kwargs)

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw data, unify the prompt format for bbh(multiple choice) benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the BBH:

            {
                'input': '((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =',
                'target': '24',
            }

        Returns:
            {'data': ['xxx']}
        """
        # few_shot_list: should be ['xxxx']
        cot_prompts: str = few_shot_list[0] if len(few_shot_list) > 0 else ''
        full_prompt: str = f"Follow the given examples and answer the question.\n{cot_prompts}\n\nQ: {input_d['input']}\nA: Let's think step by step."

        return {'data': [full_prompt]}

    def gen_prompts(self, data_dict: dict) -> dict:
        """
        Generate dataset prompts from raw input, unify the prompt format for different datasets.

        Args:
            data_dict:  Refer to the output of load method: llmuses.benchmarks.benchmark.Benchmark.load

        Returns:
            {'subset_name': [prompt_d_1, prompt_d_2, ...]}
            prompt_d_i (dict): refer to the output of gen_prompt method.

        e.g. train -- few-shot data, test -- target dataset to evaluate.
        """
        res_dict: dict = {}

        if self.few_shot_num < 0:
            raise ValueError(f'Invalid shot_num: {self.few_shot_num} for few-shot evaluation.')

        logger.info(f'\n** Use default settings: \n'
                    f'>few_shot_num: {self.few_shot_num}, '
                    f'>few_shot_split: {self.train_split}, '
                    f'>target_eval_split: {self.eval_split}')

        for sub_name, sub_data_dict in data_dict.items():
            few_shot_data = []
            if self.few_shot_num > 0:
                with open(os.path.join(os.path.dirname(__file__), 'cot_prompts', f'{sub_name}.txt'), 'r') as f:
                    cot_prompt_str = f.read()
                few_shot_data = [cot_prompt_str]

            res_dict[sub_name] = []
            for sample_d in sub_data_dict[self.eval_split]:
                prompt_d = self.gen_prompt(input_d=sample_d, few_shot_list=few_shot_data)
                prompt_d[AnswerKeys.RAW_INPUT] = sample_d
                res_dict[sub_name].append(prompt_d)

        rnd = random.Random()
        rnd.seed(42)
        for k, v in res_dict.items():
            rnd.shuffle(v)

        return res_dict

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        gold = input_d.get('target')
        if gold is None:
            logger.error(f'BBHAdapter: gold is None.')
        return gold

    def parse_pred_result(self, result: str, raw_input_d: dict = None) -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d (dict): The raw input. Depending on the dataset.

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        return self._extract_pred_answer(result)

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)

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

    def gen_report(self, subset_score_map: dict) -> dict:
        """
        Generate the report for the model output.

        Args:
            subset_score_map: The subset-score mapping. e.g. {subset_name: (score, num), ...}

        Returns: A dict of metric calculation results. The format is like:
        {
            "name":"BBH",
            "metric":"WeightedAverageAccuracy",
            "score":0.3389,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.3389,
                    "subset":[
                        {
                            "name":"BBH",
                            "score":0.3389
                        },
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

        res_map = dict(name='bbh',
                       metric=self.metric_list[0]['name'],
                       score=weighted_avg_acc,
                       category=[category_d],
                       total_num=total_num)

        return res_map

    @classmethod
    def _extract_pred_answer(cls, ans: str):
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        ans = ans.split('\n')[0]
        if ans.endswith('.'):
            ans = ans[:-1]
        return ans
