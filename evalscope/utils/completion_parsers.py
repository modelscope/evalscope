# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa

import ast
import re

# from . import utils as ann_utils
from evalscope.constants import ArenaWinner
from evalscope.utils.logger import get_logger

logger = get_logger()

one_score_pattern = re.compile('\[\[(\d+\.?\d*)\]\]')
one_score_pattern_backup = re.compile('\[(\d+\.?\d*)\]')


# modified from: https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/eval_gpt_review.py#L47
# does not work with batched completions
def lmsys_parser(completion, output_format):
    if output_format == '[[rating]]':
        match = re.search(one_score_pattern, completion)
        if not match:
            match = re.search(one_score_pattern_backup, completion)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            logger.error(f'Content: {completion}\n'
                         'You must manually fix the score.')
            rating = -1

        return rating
    if output_format == '[[rating_a,rating_b]]':
        try:
            score_pair = completion.split('\n')[0]
            score_pair = score_pair.replace(',', ' ')
            sp = score_pair.split(' ')
            if len(sp) == 2:
                score_1 = float(sp[0])
                score_2 = float(sp[1])
                if score_1 > score_2:
                    winner = ArenaWinner.MODEL_A
                elif score_1 < score_2:
                    winner = ArenaWinner.MODEL_B
                else:
                    if score_1 == score_1 == -1:
                        winner = ArenaWinner.UNKNOWN
                    winner = ArenaWinner.TIE
                return winner, [score_1, score_2]
            else:
                raise Exception('Invalid score pair.')
        except Exception as e:
            logger.error(f'{e}\nContent: {completion}\nYou must manually fix the score pair.')
            return ArenaWinner.UNKNOWN, [-1, -1]
    elif output_format == '[[A]]':
        if '[[A]]' in completion:
            winner = ArenaWinner.MODEL_A
        elif '[[B]]' in completion:
            winner = ArenaWinner.MODEL_B
        elif '[[C]]' in completion:
            winner = ArenaWinner.TIE
        else:
            logger.error(f'\nContent: {completion}\nYou must manually fix the score.')
            winner = ArenaWinner.UNKNOWN
        return winner


def ranking_parser(completion, **kwargs):
    try:
        if isinstance(completion, str):
            ordered_completions = ast.literal_eval(completion)
        else:
            ordered_completions = completion

        rank = [c for c in ordered_completions if c['model'] == 'model_a'][0]['rank']
        assert rank in [1, 2]

        return ArenaWinner.MODEL_A if rank == 1 else ArenaWinner.MODEL_B
    except Exception as e:
        logger.error(f'{e}\nContent: {completion}\n'
                     'You must manually fix the score pair.')
        return ArenaWinner.UNKNOWN
