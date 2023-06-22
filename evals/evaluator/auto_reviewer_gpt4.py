# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import time

import pandas as pd
from evals.constants import ArenaWinner
from evals.evaluator import BaseReviewer
from evals.predictors.openai_gpt_predictor import OpenaiGptPredictor
from evals.utils.arena_utils import get_battle_pairs, merge_ques_ans, shuffle_pairwise_preferences
from evals.utils.logger import get_logger
from evals.utils.utils import jsonl_dump_data, jsonl_to_list, random_seeded_choice

logger = get_logger()


class AutoReviewerGpt4(BaseReviewer):
    """
    Auto-review target answers(models) pairwise with GPT-4.

    Args:
        prompt_file: path to prompt templates file.
        answer_file_list: list of paths to answer files.
        review_file: path to review result file.
        reviewer_args: config for reviewer(GPT-4).

    Examples:
        >>> from evals.evaluator.auto_reviewer_gpt4 import AutoReviewerGpt4
        >>> input_kwargs = dict(prompt_file='/path/to/prompt_file.jsonl', answer_file_list=['/path/to/ans1_file.jsonl',
            '/path/to/ans2_file.jsonl', ...], review_file='/path/to/review_file.jsonl',
            reviewer_args={'max_tokens': 1024, 'temperature': 0.2})
        >>> auto_reviewer = AutoReviewerGpt4(**input_kwargs)
        >>> auto_reviewer.run()

    """

    MODEL_NAME = 'gpt-4'

    def __init__(self, prompt_file: str, answer_file_list: list, reference_file: str,
                 review_file: str, reviewer_args: dict, **kwargs):
        super().__init__(**kwargs)

        self.review_file = review_file
        self.prompt_list = jsonl_to_list(prompt_file)
        self.answer_list = [
            jsonl_to_list(answer_file) for answer_file in answer_file_list
        ]
        if reference_file:
            self.answer_list.append(jsonl_to_list(reference_file))
            self.reference_idx = len(self.answer_list) - 1

        self.reviewer_args = reviewer_args if reviewer_args \
            else self._get_default_args()
        self.is_randomize_output_order = self.reviewer_args.pop(
            'is_randomize_output_order', True)
        self.seed = self.reviewer_args.pop('seed', 123)
        self.gpt_predictor = OpenaiGptPredictor(**self.reviewer_args)

    @staticmethod
    def _get_default_args():
        return dict(max_tokens=1024, temperature=0.2)

    @staticmethod
    def gen_prompt(prompts_list: list, category: str, ques: str, ans1: str,
                   ans2: str):
        """
        Generate prompt for Auto-reviewer with GPT-4.
        """

        # Default to general category (idx 0)
        target_prompt_dict = prompts_list[0]
        for item in prompts_list:
            if item['category'] == category:
                target_prompt_dict = item
                break

        sys_prompt = target_prompt_dict['system_prompt']
        prompt_template = target_prompt_dict['prompt_template']
        defaults = target_prompt_dict['defaults']

        user_prompt = prompt_template.format(
            question=ques, answer_1=ans1, answer_2=ans2, **defaults)

        return sys_prompt, user_prompt

    @staticmethod
    def parse_score(review):
        try:
            score_pair = review.split('\n')[0]
            score_pair = score_pair.replace(',', ' ')
            sp = score_pair.split(' ')
            if len(sp) == 2:
                return [float(sp[0]), float(sp[1])]
            else:
                raise Exception('Invalid score pair.')
        except Exception as e:
            logger.error(f'{e}\nContent: {review}\n'
                         'You must manually fix the score pair.')
            return [-1, -1]

    def get_answer_dummy(self, sys_prompt: str, user_prompt: str) -> list:
        logger.info('Get dummy scores for input prompt ...')
        return [round(random.random(), 2), round(random.random(), 2)]

    def get_answer(self, sys_prompt: str, user_prompt: str) -> list:

        input_msg = dict(sys_prompt=sys_prompt, user_prompt=user_prompt)
        input_msg.update(self.reviewer_args)

        # Call GPT-4 predictor
        resp = self.gpt_predictor.predict(**input_msg)
        ans_text = resp['ans_text']
        # model_id = resp['model_id']

        score_pair = AutoReviewerGpt4.parse_score(ans_text)
        return score_pair

    def get_reviews(self, item: pd.Series) -> dict:

        input_msg = dict(
            ques=item[0]['text'],
            category=item[0]['category'],
            ans1=item[0]['answer'],
            ans2=item[1]['answer'])

        model_a = item[0]['model_id']
        model_b = item[1]['model_id']

        sys_prompt, user_prompt = AutoReviewerGpt4.gen_prompt(
            prompts_list=self.prompt_list, **input_msg)

        scores = self.get_answer_dummy(sys_prompt,
                                       user_prompt)  # TODO: ONLY FOR TEST

        # scores = self.get_answer(sys_prompt, user_prompt)

        def get_winner(scores_pair: list) -> str:
            if scores_pair[0] > scores_pair[1]:
                return ArenaWinner.MODEL_A
            elif scores_pair[0] < scores_pair[1]:
                return ArenaWinner.MODEL_B
            else:
                if scores_pair[0] == scores_pair[1] == -1:
                    return ArenaWinner.UNKNOWN
                return ArenaWinner.TIE

        review_result = dict(
            model_a=model_a,
            model_b=model_b,
            scores=scores,
            win=get_winner(scores),
            anony=True,
            tstamp=time.time(),
            language='NA',
            question_id=item[0]['question_id'],
            category=input_msg['category'],
            question=input_msg['ques'])
        return review_result

    def run(self):
        print('Run battles for models ...')

        merge_key = 'question_id'
        merged_ans_df = merge_ques_ans(self.answer_list, merge_key=merge_key)
        merged_ans_df = merged_ans_df.drop(columns=['question_id'])

        battle_pairs = get_battle_pairs(merged_ans_df.columns, self.reference_idx)

        res_list = []
        for t in battle_pairs:
            pair_df = merged_ans_df[list(t)]
            if self.is_randomize_output_order:
                pair_df.columns = ['output_1', 'output_2']
                pair_df["is_switched_outputs"] = pair_df.apply(
                    lambda x: random_seeded_choice(
                        seed="is_switched_outputs" + x[0]["text"] + str(self.seed),
                        choices=[False, True],
                    ),
                    axis=1,
                )
                pair_df = shuffle_pairwise_preferences(
                    pair_df, pair_df["is_switched_outputs"]
                )
                
            pair_df_combine = pair_df.apply(
                lambda x: self.get_reviews(x), axis=1)
            res_list.extend(pair_df_combine.to_list())

        os.makedirs(os.path.dirname(self.review_file), exist_ok=True)
        jsonl_dump_data(res_list, self.review_file)
