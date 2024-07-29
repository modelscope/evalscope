# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import numpy as np

from evalscope.benchmarks.data_adapter import DataAdapter
from evalscope.metrics.metrics import exact_match, weighted_mean
from evalscope.utils import normalize_score, jsonl_to_list
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


DATASET_ID = 'modelscope/hellaswag'
SUBSET_LIST = ['default']


class HellaSwagAdapter(DataAdapter):

    choices = ['0', '1', '2', '3']

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 few_shot_num: int = None,
                 train_split: str = 'train',
                 eval_split: str = 'validation',
                 **kwargs):

        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageAccuracy', 'object': weighted_mean}]

        if few_shot_num is None:
            # Use 0-shot by default
            logger.info(f'Set 0-shot examples by system for HellaSwag.')
            few_shot_num = 0

        if few_shot_num != 0:
            logger.warning(f'few_shot_num should be 0 for HellaSwag, but got {few_shot_num}. Use 0-shot by default.')
            few_shot_num = 0

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
                    file_path = os.path.join(dataset_name_or_path, f'hellaswag_{split}.jsonl')
                else:
                    file_path = os.path.join(work_dir, dataset_name_or_path, f'hellaswag_{split}.jsonl')
                if os.path.exists(file_path):
                    data_dict[subset_name][split] = jsonl_to_list(file_path)

        return data_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw data, unify the prompt format for HellaSwag benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the HellaSwag:

            {
                'ind': 4,
                'activity_label': 'Removing ice from car',
                'ctx_a': 'Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.',
                'ctx_b': 'then',
                'ctx': 'Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then',
                'endings': [', the man adds wax to the windshield and cuts it.', ', a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.', ', the man puts on a christmas coat, knitted with netting.', ', the man continues removing the snow on his car.'],
                'source_id': 'activitynet~v_-1IBHYS3L-Y',
                'split': 'train',
                'split_type': 'indomain',
                'label': '3'
            }

        Returns:
            Refer to function: evalscope.benchmarks.data_adapter.DataAdapter.gen_prompt for details.
        """

        endings: list = [self._preprocess(ending) for ending in input_d['endings']]

        few_shot_prompts = [self._generate_prompt(input_d=sample, endings=endings, include_answer=True) for sample in few_shot_list]
        context: str = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, endings=endings, include_answer=False)

        ctx_continuation_pair_list = [(context.strip(), ' ' + cont.strip()) for cont in endings]

        return {'data': ctx_continuation_pair_list, 'multi_choices': self.choices}

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d['label']

    def parse_pred_result(self, result: list, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input dict.
            eval_type: The evaluation type. e.g. checkpoint, service, custom.

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        if eval_type == 'checkpoint':
            # answer: in the form of [-2.3, -4.5, ...], len of self.choices
            result = np.array(result)
            endings: list = [self._preprocess(ending) for ending in raw_input_d['endings']]
            completion_len = np.array([float(len(i)) for i in endings])
            best_choice_idx = np.argmax(result / completion_len)

            return str(best_choice_idx)
        elif eval_type == 'service':
            return result           # TODO: to be supported !
        elif eval_type == 'custom':
            return result           # TODO: to be supported !
        else:
            raise ValueError(f'Invalid eval_type: {eval_type}')

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=str(gold), pred=str(pred))

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
            subset_score_map: The subset-score mapping. e.g. {subset_name: (score, num), ...}
            report_name: The user-defined report name.

        Returns: A dict of metric calculation results. The format is like:
        {
            "name":"HellaSwag",
            "metric":"WeightedAverageAccuracy",
            "score":0.3389,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.4128,
                    "subset":[
                        {
                            "name":"default",
                            "score":0.5632
                        },
                    ]
                }
            ],
            "total_num":7800
        }
        """
        total_num: int = sum([num for _, num in subset_score_map.values()])
        weighted_avg_acc: float = sum([score * num for score, num in subset_score_map.values()]) / total_num
        weighted_avg_acc = normalize_score(score=weighted_avg_acc)
        cate_avg_list = [{'name': subset_name, 'score': normalize_score(score=score)} for subset_name, (score, _) in subset_score_map.items()]

        category_d = dict(name='DEFAULT',
                          score=weighted_avg_acc,
                          subset=cate_avg_list)

        res_map = dict(name=report_name or 'hellaswag',
                       metric=self.metric_list[0]['name'],
                       score=weighted_avg_acc,
                       category=[category_d],
                       total_num=total_num)

        return res_map

    @classmethod
    def _preprocess(cls, text):
        text = text.strip()
        text = text.replace(' [title]', '. ')
        text = re.sub('\\[.*?\\]', '', text)
        text = text.replace('  ', ' ')
        return text

    @classmethod
    def _generate_prompt(cls, input_d: dict, endings: list, include_answer=True) -> str:
        """
        Generate prompt for HellaSwag dataset.

        Args:
            input_d: a single data of the hellaswag.
            endings:  preprocessed endings
            include_answer: bool

        Returns:

        """

        ctx = input_d['ctx_a'] + ' ' + input_d['ctx_b'].capitalize()
        example: str = cls._preprocess(input_d['activity_label'] + ': ' + ctx)

        if include_answer:
            example += '{}\n\n'.format(endings[int(input_d['label'])])

        return example
