# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import random
import re

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import AnswerKeys
from evalscope.metrics import exact_match
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

# BBH multiple choice subset list
MULTIPLE_CHOICE = 'multiple_choice'
MULTIPLE_CHOICE_LIST = [
    'temporal_sequences',
    'disambiguation_qa',
    'date_understanding',
    'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes',
    'snarks',
    'ruin_names',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'movie_recommendation',
    'salient_translation_error_detection',
    'reasoning_about_colored_objects',
]

# The free form subset list of BBH dataset
FREE_FORM = 'free_form'
FREE_FORM_LIST = [
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

# BBH sub-task type
TASK_TYPE = 'task_type'
SUBSET_LIST = MULTIPLE_CHOICE_LIST + FREE_FORM_LIST


@Benchmark.register(
    name='bbh',
    pretty_name='BBH',
    dataset_id='modelscope/bbh',
    subset_list=SUBSET_LIST,
    metric_list=['AverageAccuracy'],
    few_shot_num=3,
    train_split=None,
    eval_split='test',
    prompt_template="Q: {query}\nA: Let's think step by step.",
)
class BBHAdapter(DataAdapter):
    """
    Adapter for BBH free-form and multiple-choices sub-tasks.
    """

    def __init__(self, **kwargs):

        few_shot_num = kwargs.get('few_shot_num', 3)

        if few_shot_num != 3 and few_shot_num != 0:
            logger.error(f'BBH uses 3-shot examples with CoT or 0-shot by system, but got {few_shot_num}. '
                         f'Use 3-shot by default.')
            kwargs['few_shot_num'] = 3

        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            for split_name in [self.eval_split]:
                if os.path.exists(dataset_name_or_path):
                    file_path = os.path.join(dataset_name_or_path, f'{subset_name}.json')
                else:
                    file_path: str = os.path.join(work_dir, dataset_name_or_path, f'{subset_name}.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        examples = json.load(f)['examples']
                        if subset_name in data_dict:
                            data_dict[subset_name].update({split_name: examples})
                        else:
                            data_dict[subset_name] = {split_name: examples}

        return data_dict

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
        if len(few_shot_list) > 0:
            cot_prompts = 'Follow the given examples and answer the question.\n' + few_shot_list[0]
        else:
            cot_prompts = ''
        full_prompt = cot_prompts + self.prompt_template.format(query=input_d['input'])

        return self.gen_prompt_data(full_prompt)

    def gen_prompts(self, data_dict: dict) -> dict:
        """
        Generate dataset prompts from raw input, unify the prompt format for different datasets.

        Args:
            data_dict:  Refer to the output of load method: evalscope.benchmarks.benchmark.Benchmark.load

        Returns:
            {'subset_name': [prompt_d_1, prompt_d_2, ...]}
            prompt_d_i (dict): refer to the output of gen_prompt method.

        e.g. train -- few-shot data, test -- target dataset to evaluate.
        """
        res_dict: dict = {}

        if self.few_shot_num < 0:
            raise ValueError(f'Invalid shot_num: {self.few_shot_num} for few-shot evaluation.')

        logger.info(f'Use default settings: '
                    f'> few_shot_num: {self.few_shot_num}, '
                    f'> few_shot_split: {self.train_split}, '
                    f'> target_eval_split: {self.eval_split}')

        for sub_name, sub_data_dict in data_dict.items():
            few_shot_data = []
            if self.few_shot_num > 0:
                with open(
                        os.path.join(os.path.dirname(__file__), 'cot_prompts', f'{sub_name}.txt'), 'r',
                        encoding='utf-8') as f:
                    cot_prompt_str = f.read()
                few_shot_data = [cot_prompt_str]

            res_dict[sub_name] = []
            for sample_d in sub_data_dict[self.eval_split]:
                prompt_d = self.gen_prompt(input_d=sample_d, few_shot_list=few_shot_data)
                sample_d_new = sample_d.copy()
                if sub_name in MULTIPLE_CHOICE_LIST:
                    sample_d_new[TASK_TYPE] = MULTIPLE_CHOICE
                elif sub_name in FREE_FORM_LIST:
                    sample_d_new[TASK_TYPE] = FREE_FORM
                else:
                    raise ValueError(f'Invalid subset name: {sub_name}')

                prompt_d[AnswerKeys.RAW_INPUT] = sample_d_new
                res_dict[sub_name].append(prompt_d)

        return res_dict

    def get_gold_answer(self, input_d: dict) -> str:
        # Get the gold choice
        gold = input_d.get('target', '')
        # remove brackets
        if gold is None:
            logger.error(f'BBHAdapter: gold is None.')
        gold = gold.replace('(', '').replace(')', '')
        return gold

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
        # Note: to use same extraction method for both of checkpoint/service/custom.
        task_type: str = raw_input_d.get(TASK_TYPE)

        if task_type == MULTIPLE_CHOICE:
            return self._extract_mc_answer(result)
        elif task_type == FREE_FORM:
            return self._extract_ff_answer(result)
        else:
            raise ValueError(f'Invalid task type: {task_type}')

    def match(self, gold: str, pred: str) -> float:
        return exact_match(gold=gold, pred=pred)

    @classmethod
    def _extract_mc_answer(cls, ans: str) -> str:
        """
        Extract the answer from the model output for Multiple choice task.
        """
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        match = re.search(r'\(([A-Z])\)*', ans)
        if match:
            return match.group(1)
        match = re.search(r'([A-Z])', ans)
        if match:
            return match.group(1)
        return ans

    @classmethod
    def _extract_ff_answer(cls, ans: str):
        """
        Extract the answer from the model output for Free-form task.
        """
        pattern = r'answer is\s+(.*?)\.'

        match = re.search(pattern, ans)
        if match:
            res = match.group(1)
            return res

        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        ans = ans.split('\n')[0]
        if ans.endswith('.'):
            ans = ans[:-1]
        return ans
