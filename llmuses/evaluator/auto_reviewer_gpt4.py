# Copyright (c) Alibaba, Inc. and its affiliates.
# isort:skip_file

import os
import random
import time
from functools import partial

import pandas as pd

from llmuses.constants import ArenaMode, EvalTaskConfig, FnCompletionParser, PositionBiasMitigation
from llmuses.evaluator import BaseReviewer
from llmuses.predictors.openai_gpt_predictor import OpenaiGptPredictor
from llmuses.utils import completion_parsers
from llmuses.utils.arena_utils import (BattlePairSelection, get_battle_pairs,
                                       merge_ques_ans,
                                       shuffle_pairwise_preferences)
from llmuses.utils.logger import get_logger
from llmuses.utils.utils import jsonl_dump_data, jsonl_to_list, random_seeded_choice

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
        >>> from llmuses.evaluator.auto_reviewer_gpt4 import AutoReviewerGpt4
        >>> input_kwargs = dict(prompt_file='/path/to/prompt_file.jsonl', answer_file_list=['/path/to/ans1_file.jsonl',
            '/path/to/ans2_file.jsonl', ...], review_file='/path/to/review_file.jsonl',
            reviewer_args={'max_tokens': 1024, 'temperature': 0.2})
        >>> auto_reviewer = AutoReviewerGpt4(**input_kwargs)
        >>> auto_reviewer.run()

    """

    MODEL_NAME = 'gpt-4'

    def __init__(self, prompt_file: str, answer_file_list: list,
                 baseline_file: str, reference_file: str, review_file: str,
                 reviewer_args: dict, cache_file: str, **kwargs):
        super().__init__(**kwargs)

        self.review_file = review_file
        self.prompt_list = jsonl_to_list(prompt_file)
        self.answer_list = [
            jsonl_to_list(answer_file) for answer_file in answer_file_list
        ]
        self.reference_list = jsonl_to_list(
            reference_file) if reference_file else []
        self.cache_list = jsonl_to_list(
            cache_file) if cache_file and os.path.isfile(cache_file) else []

        self.reviewer_args = reviewer_args if reviewer_args \
            else self._get_default_args()

        self.mode = self.reviewer_args.pop('mode', ArenaMode.PAIRWISE_ALL)
        if self.mode == ArenaMode.PAIRWISE_BASELINE:
            assert baseline_file is not None, 'baseline_file is required for PAIRWISE_BASELINE mode'
            self.answer_list.append(jsonl_to_list(baseline_file))
            self.baseline_idx = len(self.answer_list) - 1

        self.position_bias_mitigation = self.reviewer_args.pop(
            EvalTaskConfig.POSITION_BIAS_MITIGATION,
            PositionBiasMitigation.NONE)
        if self.position_bias_mitigation == PositionBiasMitigation.RANDOMIZE_ORDER:
            self.random_seed = self.reviewer_args.pop(
                EvalTaskConfig.RANDOM_SEED, 123)

        fn_completion_parser = self.reviewer_args.pop(
            EvalTaskConfig.FN_COMPLETION_PARSER,
            FnCompletionParser.LMSYS_PARSER)
        completion_parser_kwargs = self.reviewer_args.pop(
            EvalTaskConfig.COMPLETION_PARSER_KWARGS, {})
        if isinstance(fn_completion_parser, str):
            fn_completion_parser = getattr(completion_parsers,
                                           fn_completion_parser)

        self.fn_completion_parser = partial(fn_completion_parser,
                                            **completion_parser_kwargs)
        self.gpt_predictor = OpenaiGptPredictor(**self.reviewer_args)

    @staticmethod
    def _get_default_args():
        return dict(
            model=AutoReviewerGpt4.MODEL_NAME,
            max_tokens=1024,
            temperature=0.2,
            mode=ArenaMode.PAIRWISE_ALL,
            position_bias_mitigation=PositionBiasMitigation.NONE,
            fn_completion_parser=FnCompletionParser.LMSYS_PARSER,
        )

    @staticmethod
    def gen_prompt(prompts_list: list,
                   type: str,
                   category: str,
                   ques: str,
                   ans1: str,
                   ans2: str = None,
                   ans_ref: str = None):
        """
        Generate prompt for Auto-reviewer with GPT-4.
        """

        # Default to general category (idx 0)
        target_prompt_dict = prompts_list[0]
        for item in prompts_list:
            is_category_match = category in item['category'] if isinstance(
                item['category'], list) else item['category'] == category
            is_type_match = item.get('type', 'pairwise') == type
            if is_category_match and is_type_match:
                target_prompt_dict = item
                break
            elif is_type_match and target_prompt_dict.get('type',
                                                          'pairwise') != type:
                target_prompt_dict = item  # fallback to type match

        sys_prompt = target_prompt_dict['system_prompt']
        prompt_template = target_prompt_dict['prompt_template']
        defaults = target_prompt_dict.get('defaults', dict({}))
        output_format = target_prompt_dict.get('output_format',
                                               '[[rating_a,rating_b]]')

        if type == ArenaMode.SINGLE:
            user_prompt = prompt_template.format(
                question=ques, answer=ans1, ref_answer_1=ans_ref, **defaults)
        else:
            user_prompt = prompt_template.format(
                question=ques,
                answer_a=ans1,
                answer_b=ans2,
                ref_answer_1=ans_ref,
                **defaults)

        return sys_prompt, user_prompt, output_format

    def get_answer_dummy(self, sys_prompt: str, user_prompt: str,
                         output_format) -> list:
        logger.info('Get dummy scores for input prompt ...')
        if output_format == '[[rating]]':
            return f'[[{round(random.random(), 2)}]]'
        if output_format == '[[rating_a,rating_b]]':
            ans_list = [round(random.random(), 2), round(random.random(), 2)]
            return ' '.join(str(element) for element in ans_list)
        elif output_format == '[[A]]':
            return random.choice(['[[A]]', '[[B]]', '[[C]]'])
        elif output_format == "[{'model': <model-name>, 'rank': <model-rank>}, " \
                              "{'model': <model-name>, 'rank': <model-rank>}]":
            rank_1 = random.choice([1, 2])
            rank_2 = 1 if rank_1 == 2 else 2
            return f"[{{'model': 'model_a', 'rank': {rank_1}}}, {{'model': 'model_b', 'rank': {rank_2}}}]"

    def get_answer(self, sys_prompt: str, user_prompt: str) -> list:

        input_msg = dict(sys_prompt=sys_prompt, user_prompt=user_prompt)
        input_msg.update(self.reviewer_args)
        input_msg['model'] = self.MODEL_NAME

        # Call GPT-4 predictor
        resp = self.gpt_predictor.predict(**input_msg)
        ans_text = resp['ans_text']
        # model_id = resp['model_id']

        return ans_text

    def get_review_cache(self, model_a, model_b, question) -> list:
        if model_b:
            cache_hit = next(
                (r for r in self.cache_list if r['model_a'] == model_a
                 and r['model_b'] == model_b and r['question'] == question),
                None)
        else:
            cache_hit = next(
                (r for r in self.cache_list
                 if r['model'] == model_a and r['question'] == question), None)
        return cache_hit

    def run_review_pair(self, model_a, model_b, question, category, ans1,
                        ans2) -> dict:
        input_msg = dict(
            ques=question, category=category, ans1=ans1, ans2=ans2)

        if self.reference_list:
            ans_ref = next((ref for ref in self.reference_list
                            if ref.get('text') == question), None)
            assert ans_ref['answer']
            input_msg['ans_ref'] = ans_ref['answer']

        sys_prompt, user_prompt, output_format = AutoReviewerGpt4.gen_prompt(
            prompts_list=self.prompt_list,
            type='single' if self.mode == ArenaMode.SINGLE else 'pairwise',
            **input_msg)

        # TODO: ONLY FOR TEST
        # review_text = self.get_answer_dummy(sys_prompt, user_prompt, output_format)

        review_text = self.get_answer(sys_prompt, user_prompt)

        result = self.fn_completion_parser(
            review_text, output_format=output_format)
        if not isinstance(result, tuple):
            result = (result, None)
        return review_text, *result

    def run_review_single(self, model, question, category, answer) -> dict:
        input_msg = dict(ques=question, category=category, ans1=answer)

        if self.reference_list:
            ans_ref = next((ref for ref in self.reference_list
                            if ref.get('text') == question), None)
            assert ans_ref['answer']
            input_msg['ans_ref'] = ans_ref['answer']

        sys_prompt, user_prompt, output_format = AutoReviewerGpt4.gen_prompt(
            prompts_list=self.prompt_list,
            type='single' if self.mode == ArenaMode.SINGLE else 'pairwise',
            **input_msg)

        # TODO: ONLY FOR TEST
        review_text = self.get_answer_dummy(sys_prompt, user_prompt,
                                            output_format)

        # review_text = self.get_answer(sys_prompt, user_prompt)

        score = self.fn_completion_parser(review_text, output_format)
        return review_text, score

    def get_review_pair(self, item: pd.Series) -> dict:
        model_a = item[0]['model_id']
        model_b = item[1]['model_id']
        question = item[0]['text']
        question_id = item[0]['question_id']
        category = item[0]['category']
        ans1 = item[0]['answer']
        ans2 = item[1]['answer']

        review_cache = self.get_review_cache(model_a, model_b, question)
        if review_cache:
            logger.info(f'Use cache review for {model_a} vs {model_b} ...')
            return review_cache

        if self.position_bias_mitigation == PositionBiasMitigation.SWAP_POSITION:
            review_text_1, winner_1, score_1 = self.run_review_pair(
                model_a, model_b, question, category, ans1, ans2)
            review_text_2, winner_2, score_2 = self.run_review_pair(
                model_b, model_a, question, category, ans2, ans1)

            # Swap winner for the second round.
            if winner_2 == 'model_a':
                winner_2 = 'model_b'
            elif winner_2 == 'model_b':
                winner_2 = 'model_a'
            review_result = dict(
                model_a=model_a,
                model_b=model_b,
                win_1=winner_1,
                win_2=winner_2,
                anony=True,
                tstamp=time.time(),
                language=item[0].get('language', 'NA'),
                question_id=question_id,
                category=category,
                question=question,
                review_text_1=review_text_1,
                review_text_2=review_text_2)
        else:
            review_text, winner, scores = self.run_review_pair(
                model_a, model_b, question, category, ans1, ans2)
            review_result = dict(
                model_a=model_a,
                model_b=model_b,
                scores=scores,
                win=winner,
                anony=True,
                tstamp=time.time(),
                language=item[0].get('language', 'NA'),
                question_id=question_id,
                category=category,
                question=question,
                review_text=review_text)
        return review_result

    def get_review_single(self, row: pd.Series):
        item = row[0]
        model = item['model_id']
        question = item['text']
        question_id = item['question_id']
        category = item['category']
        answer = item['answer']

        review_cache = self.get_review_cache(model, None, question)
        if review_cache:
            logger.info(f'Use cache review for {model} ...')
            return review_cache

        review_text, score = self.run_review_single(model, question, category,
                                                    answer)

        review_result = dict(
            model=model,
            score=score,
            anony=True,
            tstamp=time.time(),
            language=item.get('language', 'NA'),
            question_id=question_id,
            category=category,
            question=question,
            review_text=review_text)
        return review_result

    def run(self):
        print('Run battles for models ...')

        os.makedirs(os.path.dirname(self.review_file), exist_ok=True)

        merge_key = 'question_id'
        merged_ans_df = merge_ques_ans(self.answer_list, merge_key=merge_key)
        merged_ans_df = merged_ans_df.drop(columns=['question_id'])

        if self.mode == ArenaMode.PAIRWISE_ALL:
            battle_pairs = get_battle_pairs(merged_ans_df.columns)

            # tmp_file = '~/workspace/work/maas/llm-eval/llmuses/registry/data/arena/reports/elo_rating_origin.csv'
            # origin_elo_rating = pd.read_csv(tmp_file)
            # origin_elo_rating = origin_elo_rating[['Model', 'Elo_Rating']]
            # res = list(origin_elo_rating[['Model', 'Elo_Rating']].apply(lambda x: {x[0]: x[1]}, axis=1))
            # model_elo_map = dict()
            # for item in res:
            #     model_elo_map.update(item)
            #
            # battle_selection = BattlePairSelection(model_elo_map)
            # battle_pairs = battle_selection.volatility_index(frac=0.5, swap=True)
            # print('>>>battle_pairs: \n', battle_pairs)
            #
            # sys.exit(0)

        elif self.mode == ArenaMode.PAIRWISE_BASELINE:
            battle_pairs = get_battle_pairs(merged_ans_df.columns,
                                            self.baseline_idx)
        elif self.mode == ArenaMode.SINGLE:
            battle_pairs = [(col, ) for col in merged_ans_df.columns]
        else:
            raise Exception(f'NotSupported mode: {self.mode}')

        res_list = []
        for t in battle_pairs:
            pair_df = merged_ans_df[list(t)]
            if self.position_bias_mitigation == PositionBiasMitigation.RANDOMIZE_ORDER:
                pair_df.columns = ['output_1', 'output_2']
                pair_df['is_switched_outputs'] = pair_df.apply(
                    lambda x: random_seeded_choice(
                        seed='is_switched_outputs' + x[0]['text'] + str(
                            self.random_seed),
                        choices=[False, True],
                    ),
                    axis=1,
                )
                pair_df = shuffle_pairwise_preferences(
                    pair_df, pair_df['is_switched_outputs'])

            for index, row in pair_df.iterrows():
                row_result = self.get_review_pair(
                    row
                ) if self.mode != ArenaMode.SINGLE else self.get_review_single(
                    row)
                res_list.append(row_result)
        jsonl_dump_data(res_list, self.review_file)
