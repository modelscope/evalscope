# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import json
from evalscope.benchmarks.data_adapter import DataAdapter
from evalscope.metrics.metrics import exact_match, weighted_mean
from evalscope.utils import normalize_score, ResponseParser
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

DATASET_ID = 'modelscope/ai2_arc'

# task_list = ['ARC-Easy', 'ARC-Challenge']
SUBSET_LIST = ['ARC-Challenge']


class ARCAdapter(DataAdapter):

    choices = ['A', 'B', 'C', 'D']

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 few_shot_num: int = None,
                 train_split: str = 'train',
                 eval_split: str = 'test',
                 prompt_template: str = '',
                 **kwargs):

        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageAccuracy', 'object': weighted_mean}]

        if few_shot_num is None:
            # Use 0-shot by default
            logger.info(f'Set 0-shot examples by system for ARC.')
            few_shot_num = 0

        if few_shot_num != 0:
            logger.warning(f'few_shot_num is recommended to set 0 for ARC, got {few_shot_num}.')

        super().__init__(subset_list=subset_list,
                         metric_list=metric_list,
                         few_shot_num=few_shot_num,
                         train_split=train_split,
                         eval_split=eval_split,
                         prompt_template=prompt_template,
                         **kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        """
        Load the dataset from local disk.

        dataset_name_or_path: str, the dataset id or path. e.g. 'arc'
        subset_list: list, the subset list to load. e.g. ['ARC-Easy', 'ARC-Challenge']
        work_dir: str, the local root data directory. e.g. '/path/to/data'
        kwargs: dict, other arguments.
        """
        data_dict = {}
        for subset_name in subset_list:
            if os.path.exists(dataset_name_or_path):
                subset_path = os.path.join(dataset_name_or_path, subset_name)
            else:
                subset_path = os.path.join(work_dir, dataset_name_or_path, subset_name)
            for split_name in ['Train', 'Test']:
                split_path = os.path.join(subset_path, f'{subset_name}-{split_name}.jsonl')
                if os.path.exists(split_path):
                    with open(split_path, 'r', errors='ignore') as in_f:
                        rows = []
                        for line in in_f:
                            item = json.loads(line.strip())
                            raw_choices = item['question']['choices']
                            rows.append({
                                'id': item['id'],
                                'question': item['question']['stem'],
                                'choices': {'text': [d['text'] for d in raw_choices],
                                            'label': [d['label'] for d in raw_choices]},
                                'answerKey': item['answerKey'],
                            })

                        if subset_name in data_dict:
                            data_dict[subset_name].update({split_name.lower(): rows})
                        else:
                            data_dict[subset_name] = {split_name.lower(): rows}

        return data_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw data, unify the prompt format for ARC benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the ARC:

            {
                'id': 'Mercury_7220990',
                'question': 'Which factor will most likely cause a person to develop a fever?',
                'choices':
                    {
                        'text':['a leg muscle relaxing after exercise',
                                'a bacterial population in the bloodstream',
                                'several viral particles on the skin',
                                'carbohydrates being digested in the stomach'],
                        'label': ['A', 'B', 'C', 'D']
                    },
                'answerKey': 'B'
            }

        Returns:
            {'data': ['xxx'], 'multi_choices': ['A', 'B', 'C', 'D']}
        """
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]
        context: str = '\n'.join(few_shot_prompts)

        context = f'{self.prompt_template}\n{context}' if self.prompt_template else context

        # context = f'The following are multiple choice questions, please output correct answer in the form of A or B or C or D, do not output explanation:\n {context}'
        full_prompt: str = context + self._generate_prompt(input_d=input_d, include_answer=False)

        return {'data': [full_prompt], 'multi_choices': self.choices}

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d.get('answerKey', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d (dict): The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or `custom`, default: 'checkpoint'

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        if eval_type == 'checkpoint':
            return result
        elif eval_type == 'service':
            return ResponseParser.parse_first_option_with_choices(text=result, options=self.choices)  # TODO: to be checked !
        elif eval_type == 'custom':
            return ResponseParser.parse_first_option_with_choices(text=result, options=self.choices)  # TODO: to be checked !
        else:
            raise ValueError(f'Invalid eval_type: {eval_type}')

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

    def gen_report(self, subset_score_map: dict, report_name: str = None) -> dict:
        """
        Generate the report for the model output.

        Args:
            subset_score_map: The subset-score mapping. e.g. {subset_name: (score, num), ...}
            report_name: The user-defined report name.

        Returns: A dict of metric calculation results. The format is like:
        {
            "name":"ARC",
            "metric":"WeightedAverageAccuracy",
            "score":0.3389,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.4128,
                    "subset":[
                        {
                            "name":"ARC-Easy",
                            "score":0.5632
                        },
                        {
                            "name":"ARC-Challenge",
                            "score":0.3157
                        }
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

        res_map = dict(name=report_name or 'arc',
                       metric=self.metric_list[0]['name'],
                       score=weighted_avg_acc,
                       category=[category_d],
                       total_num=total_num)

        return res_map

    @classmethod
    def _generate_prompt(cls, input_d: dict, include_answer=True) -> str:

        example: str = input_d['question']

        choices_texts: list = input_d['choices']['text']
        choices_labels: list = input_d['choices']['label']
        choices_prompts: str = '\n'.join([label + '. ' + text for text, label in zip(choices_texts, choices_labels)])
        example += '\n' + choices_prompts

        example += '\nAnswer:'
        if include_answer:
            example += ' {}\n\n'.format(input_d['answerKey'])

        return example
