# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa

import os
import pandas as pd
import random
import sys
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, List, Tuple

from evalscope.constants import ArenaMode, EvalConfigKeys, FnCompletionParser, PositionBiasMitigation
from evalscope.models import OpenAIModel
from evalscope.utils import completion_parsers, random_seeded_choice
from evalscope.utils.arena_utils import get_battle_pairs, merge_ques_ans, shuffle_pairwise_preferences
from evalscope.utils.io_utils import dump_jsonl_data, jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()


class BaseReviewer(ABC):

    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run pairwise battles with given models.
        """
        raise NotImplementedError('run() method must be implemented in your subclass.')


class AutoReviewerGpt4(BaseReviewer):
    """
    Auto-review target answers(models) pairwise with GPT-4.

    Args:
        prompt_file: path to prompt templates file.
        answer_file_list: list of paths to answer files.
        review_result_file: path to review result file.
        reviewer_args: config for reviewer(GPT-4).

    Examples:
        >>> from evalscope.evaluator.reviewer.auto_reviewer import AutoReviewerGpt4
        >>> input_kwargs = dict(prompt_file='/path/to/prompt_file.jsonl', answer_file_list=['/path/to/ans1_file.jsonl',
            '/path/to/ans2_file.jsonl', ...], review_file='/path/to/review_file.jsonl',
            reviewer_args={'model': 'gpt-4', 'mode': 'single'})
        >>> auto_reviewer = AutoReviewerGpt4(**input_kwargs)
        >>> auto_reviewer.run(dry_run=False)
    """

    MODEL_NAME = 'gpt-4'

    def __init__(self,
                 prompt_file: str,
                 answer_file_list: list,
                 review_result_file: str,
                 baseline_file: str = None,
                 reference_file: str = None,
                 reviewer_args: dict = None,
                 cache_file: str = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.review_result_file = review_result_file
        self.prompt_list = jsonl_to_list(prompt_file)
        self.answer_list = [jsonl_to_list(answer_file) for answer_file in answer_file_list]
        self.reference_list = jsonl_to_list(reference_file) if reference_file else []
        self.cache_list = jsonl_to_list(cache_file) if cache_file and os.path.isfile(cache_file) else []

        self.reviewer_args = reviewer_args if reviewer_args \
            else self._get_default_args()

        self.review_mode = self.reviewer_args.pop('mode', ArenaMode.PAIRWISE)
        if self.review_mode == ArenaMode.PAIRWISE_BASELINE:
            assert baseline_file is not None, f'baseline_file is required for {ArenaMode.PAIRWISE_BASELINE} mode'
            self.answer_list.append(jsonl_to_list(baseline_file))
            self.baseline_idx = len(self.answer_list) - 1

        self.position_bias_mitigation = self.reviewer_args.pop(EvalConfigKeys.POSITION_BIAS_MITIGATION,
                                                               PositionBiasMitigation.NONE)
        if self.position_bias_mitigation == PositionBiasMitigation.RANDOMIZE_ORDER:
            self.random_seed = self.reviewer_args.pop(EvalConfigKeys.RANDOM_SEED, 123)

        fn_completion_parser = self.reviewer_args.pop(EvalConfigKeys.FN_COMPLETION_PARSER,
                                                      FnCompletionParser.LMSYS_PARSER)
        completion_parser_kwargs = self.reviewer_args.pop(EvalConfigKeys.COMPLETION_PARSER_KWARGS, {})
        if isinstance(fn_completion_parser, str):
            fn_completion_parser = getattr(completion_parsers, fn_completion_parser)

        self.fn_completion_parser = partial(fn_completion_parser, **completion_parser_kwargs)
        self.gpt_predictor = OpenAIModel(model_cfg=self.reviewer_args)

    @staticmethod
    def _get_default_args():
        return dict(
            model=AutoReviewerGpt4.MODEL_NAME,
            mode=ArenaMode.PAIRWISE,
            position_bias_mitigation=PositionBiasMitigation.NONE,
            fn_completion_parser=FnCompletionParser.LMSYS_PARSER,
            random_seed=123,
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
            is_category_match = category in item['category'] if isinstance(item['category'],
                                                                           list) else item['category'] == category
            is_type_match = item.get('type', ArenaMode.PAIRWISE) == type
            if is_category_match and is_type_match:
                target_prompt_dict = item
                break
            elif is_type_match and target_prompt_dict.get('type', ArenaMode.PAIRWISE) != type:
                target_prompt_dict = item  # fallback to type match

        sys_prompt = target_prompt_dict['system_prompt']
        prompt_template = target_prompt_dict['prompt_template']
        defaults = target_prompt_dict.get('defaults', dict({}))
        output_format = target_prompt_dict.get('output_format', '[[rating_a,rating_b]]')

        if type == ArenaMode.SINGLE:
            user_prompt = prompt_template.format(question=ques, answer=ans1, ref_answer_1=ans_ref, **defaults)
        else:
            user_prompt = prompt_template.format(
                question=ques, answer_a=ans1, answer_b=ans2, ref_answer_1=ans_ref, **defaults)

        return sys_prompt, user_prompt, output_format

    def get_review_cache(self, model_a, model_b, question) -> list:
        if model_b:
            cache_hit = next((r for r in self.cache_list
                              if r['model_a'] == model_a and r['model_b'] == model_b and r['question'] == question),
                             None)
        else:
            cache_hit = next((r for r in self.cache_list if r['model'] == model_a and r['question'] == question), None)
        return cache_hit

    def get_review_pair(self, item: List[dict], dry_run=False, **kwargs) -> dict:

        question = item[0]['text']
        question_id = item[0]['question_id']
        category = item[0]['category']

        model_a = item[0]['model_id']
        model_b = item[1]['model_id']

        ans1 = item[0]['answer']
        ans2 = item[1]['answer']

        review_cache = self.get_review_cache(model_a, model_b, question)
        if review_cache:
            logger.info(f'Use cache review for {model_a} vs {model_b} ...')
            return review_cache

        if self.position_bias_mitigation == PositionBiasMitigation.SWAP_POSITION:
            review_text_1, winner_1, score_1 = self._get_review_pair(
                model_a, model_b, question, category, ans1, ans2, dry_run=dry_run, **kwargs)
            review_text_2, winner_2, score_2 = self._get_review_pair(
                model_b, model_a, question, category, ans2, ans1, dry_run=dry_run, **kwargs)

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
            review_text, winner, scores = self._get_review_pair(
                model_a, model_b, question, category, ans1, ans2, dry_run=dry_run, **kwargs)

            if dry_run:
                scores = [round(random.random(), 1), round(random.random(), 1)]
                winner = 'model_a' if scores[0] > scores[1] else 'model_b'

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

    def get_review_single(self, row: List[dict], dry_run: bool = False, **kwargs):
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

        review_text, score = self._get_review_single(model, question, category, answer, dry_run=dry_run, **kwargs)

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

    def _get_review_pair(self,
                         model_a,
                         model_b,
                         question,
                         category,
                         ans1,
                         ans2,
                         dry_run=False,
                         **kwargs) -> Tuple[str, Any]:
        input_msg = dict(ques=question, category=category, ans1=ans1, ans2=ans2)

        if self.reference_list:
            ans_ref = next((ref for ref in self.reference_list if ref.get('text') == question), None)
            assert ans_ref['answer']
            input_msg['ans_ref'] = ans_ref['answer']

        sys_prompt, user_prompt, output_format = AutoReviewerGpt4.gen_prompt(
            prompts_list=self.prompt_list,
            type=ArenaMode.SINGLE if self.review_mode == ArenaMode.SINGLE else ArenaMode.PAIRWISE,
            **input_msg)

        if dry_run:
            review_text = self._get_reviewer_prediction_dummy(sys_prompt, user_prompt, output_format)
        else:
            review_text = self._get_reviewer_prediction(sys_prompt, user_prompt, **kwargs)

        result = self.fn_completion_parser(review_text, output_format=output_format)
        if not isinstance(result, tuple):
            result = (result, None)
        return review_text, *result

    def _get_review_single(self, model, question, category, answer, dry_run=False, **kwargs) -> Tuple[str, Any]:
        input_msg = dict(ques=question, category=category, ans1=answer)

        if self.reference_list:
            ans_ref = next((ref for ref in self.reference_list if ref.get('text') == question), None)
            assert ans_ref['answer']
            input_msg['ans_ref'] = ans_ref['answer']

        sys_prompt, user_prompt, output_format = AutoReviewerGpt4.gen_prompt(
            prompts_list=self.prompt_list,
            type=ArenaMode.SINGLE if self.review_mode == ArenaMode.SINGLE else ArenaMode.PAIRWISE,
            **input_msg)

        if dry_run:
            review_text = self._get_reviewer_prediction_dummy(sys_prompt, user_prompt, output_format)
        else:
            review_text = self._get_reviewer_prediction(sys_prompt, user_prompt, **kwargs)

        score = self.fn_completion_parser(review_text, output_format)
        return review_text, score

    def _get_reviewer_prediction_dummy(self, sys_prompt: str, user_prompt: str, output_format) -> str:
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

    def _get_reviewer_prediction(self, sys_prompt: str, user_prompt: str, **kwargs) -> str:

        input_msg = dict(sys_prompt=sys_prompt, user_prompt=user_prompt)

        # Call GPT-4 predictor
        # TODO: Add more reviewer implementation
        resp = self.gpt_predictor.predict(model_id=self.MODEL_NAME, inputs=input_msg, **kwargs)

        if resp is None or len(resp) == 0:
            logger.error(f'Failed to get response from {self.MODEL_NAME} for input: {input_msg}')

        ans_text = resp['ans_text']
        # model_id = resp['model_id']

        return ans_text

    def run(self, dry_run: bool = False, **kwargs):
        print(f'Run battles for models with dry_run={dry_run} ...')

        os.makedirs(os.path.dirname(self.review_result_file), exist_ok=True)

        if len(self.answer_list) == 0:
            raise Exception('The answer list cannot be empty.')

        merge_key = 'question_id'
        merged_ans_df = merge_ques_ans(self.answer_list, merge_key=merge_key)
        merged_ans_df = merged_ans_df.drop(columns=['question_id'])

        if self.review_mode == ArenaMode.PAIRWISE:
            battle_pairs = get_battle_pairs(merged_ans_df.columns)
        elif self.review_mode == ArenaMode.PAIRWISE_BASELINE:
            battle_pairs = get_battle_pairs(merged_ans_df.columns, self.baseline_idx)
        elif self.review_mode == ArenaMode.SINGLE:
            battle_pairs = [(col, ) for col in merged_ans_df.columns]
        else:
            raise Exception(f'NotSupported mode: {self.review_mode}')

        res_list = []
        for t in battle_pairs:
            pair_df = merged_ans_df[list(t)]
            if self.position_bias_mitigation == PositionBiasMitigation.RANDOMIZE_ORDER:
                pair_df.columns = ['output_1', 'output_2']
                pair_df['is_switched_outputs'] = pair_df.apply(
                    lambda x: random_seeded_choice(
                        seed='is_switched_outputs' + x[0]['text'] + str(self.random_seed),
                        choices=[False, True],
                    ),
                    axis=1,
                )
                pair_df = shuffle_pairwise_preferences(pair_df, pair_df['is_switched_outputs'])

            for index, row in pair_df.iterrows():
                row_result = self.get_review_pair(row.to_list(), dry_run=dry_run, **kwargs) \
                    if self.review_mode != ArenaMode.SINGLE \
                    else self.get_review_single(row.to_list(), dry_run=dry_run, **kwargs)
                res_list.append(row_result)
        dump_jsonl_data(res_list, self.review_result_file)


if __name__ == '__main__':
    from pathlib import Path

    work_path = os.path.join(Path(__file__).absolute().parent, '../../../')
    prompt_template_path = os.path.join(work_path, 'evalscope/registry/data/prompt_template/prompt_templates.jsonl')
    answer_file_list = [
        os.path.join(work_path, 'outputs/arena/default/answers/answer_chatglm2-6b.jsonl'),
        os.path.join(work_path, 'outputs/arena/default/answers/answer_llama2-7b.jsonl')
    ]
    review_result_file_path = os.path.join(work_path, 'outputs/arena/default/reviews/review_gpt4.jsonl')

    input_kwargs = dict(
        prompt_file=prompt_template_path,
        answer_file_list=answer_file_list,
        review_result_file=review_result_file_path,
        reviewer_args={},
        baseline_file='',
        reference_file='',
        cache_file='',
    )

    auto_reviewer = AutoReviewerGpt4(**input_kwargs)
    auto_reviewer.run(dry_run=True)
