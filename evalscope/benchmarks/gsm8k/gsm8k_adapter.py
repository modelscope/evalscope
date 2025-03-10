# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI, Inc. and its affiliates.
# flake8: noqa
import math
import os
import re

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()


@Benchmark.register(
    name='gsm8k',
    pretty_name='GSM8K',
    dataset_id='modelscope/gsm8k',
    subset_list=['main'],
    metric_list=['AverageAccuracy'],
    few_shot_num=4,
    train_split=None,
    eval_split='test',
    prompt_template="Question: {query}\nLet's think step by step\nAnswer:",
)
class GSM8KAdapter(DataAdapter):

    def __init__(self, **kwargs):
        """
        Data adapter for GSM8K dataset.

        Args:
            subset_list (list): Subset list for the dataset. Default: ['main']
            metric_list (list): Metric list for the dataset. Default: [{'name': 'AverageAccuracy', 'object': mean}]
            few_shot_num (int): Number of few-shot examples. Default: 4
            train_split (str): Train split name. Default: 'train'
            eval_split (str): The target eval split name. Default: 'test'
            **kwargs: ...
        """
        few_shot_num = kwargs.get('few_shot_num', 4)
        if few_shot_num != 4 and few_shot_num != 0:
            logger.error(f'GSM8K uses 4-shot examples with CoT or 0-shot by system, but got {few_shot_num}. '
                         f'Use 4-shot by default.')
            kwargs['few_shot_num'] = 4

        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}
            for split in [self.train_split, self.eval_split]:
                if os.path.exists(dataset_name_or_path):
                    file_path = os.path.join(dataset_name_or_path, f'{split}.jsonl')
                else:
                    file_path = os.path.join(work_dir, dataset_name_or_path, f'{split}.jsonl')
                if os.path.exists(file_path):
                    data_dict[subset_name][split] = jsonl_to_list(file_path)

        return data_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate prompt for the model.

        Args:
            input_d (dict): The raw input. A single data format of the GSM8K:
            {
                "question": "Janet\\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers\' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers\' market?",
                "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\\u2019s market.\\n#### 18"
            }
        """
        use_fewshot = self.few_shot_num > 0
        context = self._generate_prompt(use_fewshot=use_fewshot)

        full_prompt = context + self.prompt_template.format(query=input_d['question'])

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        ans: str = input_d.get('answer', '')
        ans = self.extract_answer(ans).strip()
        if not ans:
            logger.error(f'No ground truth answer found in the input: {input_d}')
        return ans

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d (dict): The raw input. Depending on the dataset.
            eval_type (str): 'checkpoint' or 'service'

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        # Note: to use same extraction method for both of checkpoint and custom.
        return self.extract_answer(result)

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and predicted answer.

        Args:
            gold (str): The golden answer. Note: to be extracted.
            pred (str): The extracted prediction. Usually a string for chat/multiple-choice-questions.
                        e.g. 'B'
        """

        def number_equal(gold_ans, pred_ans):
            if pred_ans is None:
                return False
            try:
                return math.isclose(eval(gold_ans), eval(pred_ans), rel_tol=0, abs_tol=1e-4)
            except:
                logger.warning(f'##report##Cannot compare two numbers: gold_ans={gold_ans}, pred_ans={pred_ans}')
                return False

        return number_equal(gold_ans=gold, pred_ans=pred)

    @classmethod
    def _generate_prompt(cls, use_fewshot: bool = True) -> str:
        if use_fewshot:
            # Use 4-shot examples by system
            context = (
                "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\n"
                'Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\n'
                "Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\n"
                "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n\n"
                "Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\n"
                "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n\n"
                "Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\n"
                'For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n\n'
            )
        else:
            context = ''
        return context

    @staticmethod
    def extract_answer(s: str) -> str:
        _PAT_LAST_DIGIT = re.compile(r'([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)')
        match = list(_PAT_LAST_DIGIT.finditer(s))
        if match:
            last_digit = match[-1].group().replace(',', '').replace('+', '').strip().strip('.')
            # print(f"The last digit in {s} is {last_digit}")
        else:
            last_digit = None
            print(f'No digits found in {s!r}', flush=True)

        return last_digit
