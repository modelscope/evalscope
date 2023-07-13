import ast
import copy
from typing import Any
import re
import numpy as np

from . import utils as ann_utils
from evals.constants import ArenaWinner
from evals.utils.logger import get_logger

logger = get_logger()

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

def regex_parser(completion: str, outputs_to_match: dict[str, Any]) -> list[Any]:
    """Parse a single batch of completions, by returning a sequence of keys in the order in which outputs_to_match
    was matched.

    Parameters
    ----------
    completion : str
        Completion to parse.

    outputs_to_match : dict[str, Any]
        Dictionary of compiled regex to match. Keys are the keys to return in the order in which they are matched.
        The values can be either a compiled regex or a string. If a string, it will be compiled to a regex and that will
        be modified inplace.

    Examples
    --------
    >>> completion = '\n(b)\n\n### Best output for example 8:\n(a)\n\n### Best output for example 9:\n(b)\n\n### Best
    output for example 10:\n(a)\n\n### Best output for example 11:\n(a)'
    >>> regex_parser(completion, {1: '\n\(a\)', 2: '\n\(b\)'})
    [2, 1, 2, 1, 1]
    >>> regex_parser(' (a)', {1: ' \(a\)', 2: ' \(b\)'})
    [1]
    >>> completion = '### Preferred output in JSON format for example 4:\r\n{{\r\n"Concise explanation": "Both
    outputs are incorrect, but Output (a) is less confusing and more concise.",\r\n"Output (a) is better than Output
    (b)": true\r\n}}\r\n\r\n### Preferred output in JSON format for example 5:\r\n{{\r\n"Concise explanation": "Both
    outputs are incomplete, but Output (b) seems to start with a more relevant source.",\r\n"Output (a) is better
    than Output (b)": false\r\n}}\r\n\r\n### Preferred output in JSON format for example 6:\r\n{{\r\n"Concise
    explanation": "Both outputs are incorrect, but Output (a) is less confusing and more concise.",\r\n"Output (a) is
    better than Output (b)": true\r\n}}\r\n\r\n### Preferred output in JSON format for example 7:\r\n{{\r\n"Concise
    explanation": "Both outputs are incomplete, but Output (b) seems to start with a more relevant source.",
    \r\n"Output (a) is better than Output (b)": false\r\n}}'
    >>> regex_parser(completion, {1: ' true', 2: ' false'})
    [1, 2, 1, 2]
    """
    for k, v in outputs_to_match.items():
        if not isinstance(v, re.Pattern):
            # inplace modification, which is bad practice but useful to speedup
            outputs_to_match[k] = re.compile(v)

    completion = copy.deepcopy(completion)
    responses = []
    while True:
        match, key = ann_utils._find_first_match(completion, outputs_to_match)
        if not match:
            break
        responses.append(key)
        # avoid matching the same output twice
        completion = completion[match.end():]
    return responses


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
            logger.error(f'{e}\nContent: {completion}\n'
                        'You must manually fix the score pair.')
            return ArenaWinner.UNKNOWN, [-1, -1]
    elif output_format == '[[A]]':
        if "[[A]]" in completion:
            winner = ArenaWinner.MODEL_A
        elif "[[B]]" in completion:
            winner = ArenaWinner.MODEL_B
        elif "[[C]]" in completion:
            winner = ArenaWinner.TIE
        else:
            logger.error(f'{e}\nContent: {completion}\n'
                        'You must manually fix the score.')
            winner = ArenaWinner.UNKNOWN
        return winner


def ranking_parser(completion, **kwargs):
    try:
        if isinstance(completion, str):
            ordered_completions = ast.literal_eval(completion)
        else:
            ordered_completions = completion

        rank = [c for c in ordered_completions if c["model"] == "model_a"][0]["rank"]
        assert rank in [1, 2]

        return ArenaWinner.MODEL_A if rank == 1 else ArenaWinner.MODEL_B
    except Exception as e:
        logger.error(f"{e}\nContent: {completion}\n" "You must manually fix the score pair.")
        return ArenaWinner.UNKNOWN
