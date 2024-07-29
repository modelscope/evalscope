# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import json
from evalscope.benchmarks.data_adapter import DataAdapter
from evalscope.metrics.metrics import exact_match, weighted_mean
from evalscope.utils import normalize_score, jsonl_to_list
from evalscope.utils.logger import get_logger
# flake8: noqa

logger = get_logger()

DATASET_ID = 'modelscope/race'

SUBSET_LIST = [
    "high",
    "middle"
]


SUBJECT_MAPPING = {"high": "High",
                   "middle": "Middle"
}


class RACEAdapter(DataAdapter):

    choices = ['A', 'B', 'C', 'D']

    def __init__(self,
                 subset_list: list = None,
                 metric_list: list = None,
                 few_shot_num: int = None,
                 train_split: str = 'train',
                 eval_split: str = 'test',
                 **kwargs):

        if subset_list is None:
            subset_list = SUBSET_LIST

        if metric_list is None:
            metric_list = [{'name': 'WeightedAverageAccuracy', 'object': weighted_mean}]

        if few_shot_num is None:
            logger.info(f'Set 3-shot examples by system for RACE.')
            few_shot_num = 3

        if few_shot_num > 3:
            logger.warning(f'few_shot_num <= 3 for RACE, but got {few_shot_num}. Use 3-shot by default.')
            few_shot_num = 3

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
                    file_path = os.path.join(dataset_name_or_path, subset_name, f'{split}.jsonl')
                else:
                    file_path = os.path.join(work_dir, dataset_name_or_path, subset_name, f'{split}.jsonl')
                if os.path.exists(file_path):
                    data_dict[subset_name][split] = jsonl_to_list(file_path)

        return data_dict

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from raw input, unify the prompt format for RACE benchmark.

        Args:
            input_d (dict): The raw input. A single data format of the RACE:

            {'example_id': 'high3680.txt',
            'article': 'Astronauts on shorter shuttle missions often work very long days. Tasks are scheduled so tightly that break times are often used to finish the day's work. This type of schedule is far too demanding for long missions on the International Space Station(ISS). ISS crewmembers usually live in space for at least a quarter of a year. They work five days on and two days off to _ the normal way they do things on Earth as much as possible. Weekends give the crew valuable time to rest and do a few hours of housework. They can communicate with family and friends by email , internet phone and through private video conferences. While astronauts cannot go to a baseball game or a movie in orbit, there are many familiar activities that they can still enjoy . Before a mission, the family and friends of each ISS crewmember put together a collection of family photos, messages, videos and reading material for the astronauts to look at when they will be floating 370 kilometers above the Earth. During their mission, the crew also receives care packages with CDs, books, magazines, photos and letters . And as from early 2010, the internet became available on the ISS , giving astronauts the chance to do some "web surfing "in their personal time. Besides relaxing with these more common entertainments, astronauts can simply enjoy the experience of living in space. Many astronauts say that one of the most relaxing things to do in space is to look out the window and stare at the universe and the Earth's vast land mass and oceans.',
            'answer': 'C',
            'question': 'The passage mainly discusses how astronauts _ .',
            'options': [
                "work for longer missions in space",
                "connect with people on the Earth",
                "spend their free time in space",
                "observe the Earth from space"]}

        Returns:
            {'data': [(context, continuation), ...]}

        """
        prompt = 'The following are multiple choice reading comprehension questions (with answers).\n\n'.format(
            self._format_subject(subset_name)
        )
        few_shot_prompts = [self._generate_prompt(input_d=sample, include_answer=True) for sample in few_shot_list]

        context: str = '\n'.join(few_shot_prompts) + '\n'
        context += self._generate_prompt(input_d=input_d, include_answer=False)
        context = prompt + context

        full_prompt: str = context.strip() + self._generate_prompt(input_d=input_d, include_answer=False)

        return {'data': [full_prompt], 'multi_choices': self.choices}

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        return input_d.get('answer', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input. Depending on the dataset.
            eval_type: The evaluation type. e.g. 'checkpoint' or 'service' or 'custom'.

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        if eval_type == 'checkpoint':
            return result
        elif eval_type == 'service':        # TODO: to be implemented
            return result
        elif eval_type == 'custom':         # TODO: to be implemented
            return result
        else:
            raise ValueError(f'Unknown eval_type: {eval_type}')

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
        Generate report for the evaluation.

        Args:
            subset_score_map: The subset-score mapping. e.g. {subset_name: (score, num), ...}
            report_name: The user-defined report name.

        Returns:
        {
            "name":"RACE",
            "metric":"WeightedAverageAccuracy",
            "score":0.3389,
            "category":[
                {
                   "name":"High",
                   "score":0.2528,
                   "subset":[
                       {
                           "name":"high",
                           "score":0.2528
                       }
                   ]
                }
            ],
            "total_num":59
        }
        """
        total_num: int = sum([num for _, num in subset_score_map.values()])
        weighted_avg_acc: float = sum([score * num for score, num in subset_score_map.values()]) / total_num

        # Get domain-subject mapping
        subject_review_map = {}
        for subset_name, (subset_score, num) in subset_score_map.items():
            domain_name: str = SUBJECT_MAPPING.get(subset_name)
            if domain_name in subject_review_map:
                subject_review_map[domain_name].append((subset_name, subset_score, num))
            else:
                subject_review_map[domain_name] = [(subset_name, subset_score, num)]

        # Get domain score
        category_list = []
        for domain_name, domain_res_list in subject_review_map.items():
            domain_weighted_avg_acc = sum([score * num for _, score, num in domain_res_list]) / \
                                     sum([num for _, _, num in domain_res_list])
            domain_weighted_avg_acc = normalize_score(score=domain_weighted_avg_acc)
            category_list.append({'name': domain_name,
                                  'score': normalize_score(score=domain_weighted_avg_acc),
                                  'subset': [{'name': subset_name, 'score': subset_score}
                                             for subset_name, subset_score, _ in domain_res_list]})

        # Get final dict of report
        res_map = dict(name=report_name or 'race',
                       metric=self.metric_list[0]['name'],
                       score=weighted_avg_acc,
                       category=category_list,
                       total_num=total_num)

        return res_map

    @classmethod
    def _generate_prompt(cls, input_d: dict, include_answer=True) -> str:

        input_choices: list = input_d['options']

        example: str = 'Article:\n{}\nQuestion:\n{}'.format(input_d['article'], input_d['question'])
        for j in range(len(cls.choices)):
            example += '\n{}. {}'.format(cls.choices[j], input_choices[j])

        example += '\nAnswer:'
        if include_answer:
            example += ' {}\n\n'.format(input_d['answer'])

        return example

    @classmethod
    def _format_subject(cls, subject):
        l = subject.split('_')
        s = ''
        for entry in l:
            s += ' ' + entry
        return s
