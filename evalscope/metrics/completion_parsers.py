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


class ResponseParser:

    @staticmethod
    def parse_first_capital(text: str, options: list[str]) -> str:
        for t in text:
            if t.isupper() and (t in options):
                return t
        return ''

    @staticmethod
    def parse_last_capital(text: str, options: list[str]) -> str:
        for t in text[::-1]:
            if t.isupper() and (t in options):
                return t
        return ''

    @staticmethod
    def parse_first_option_with_choices(text: str, options: list[str]) -> str:
        """
        Find first valid option for text.

        Args:
            text: The text to parse.
            options: The options to find. e.g. ['A', 'B', 'C', 'D']
        """
        options_concat = ResponseParser.process_options(options)

        patterns = [
            rf'答案是?\s?([{options_concat}])',
            rf'答案是?\s?：([{options_concat}])',
            rf'答案是?\s?:([{options_concat}])',
            rf'答案应该?是\s?([{options_concat}])',
            rf'答案应该?选\s?([{options_concat}])',
            rf'答案为\s?([{options_concat}])',
            rf'答案选\s?([{options_concat}])',
            rf'选择?\s?([{options_concat}])',
            rf'故选?\s?([{options_concat}])'
            rf'只有选?项?\s?([{options_concat}])\s?是?对',
            rf'只有选?项?\s?([{options_concat}])\s?是?错',
            rf'只有选?项?\s?([{options_concat}])\s?不?正确',
            rf'只有选?项?\s?([{options_concat}])\s?错误',
            rf'说法不?对选?项?的?是\s?([{options_concat}])',
            rf'说法不?正确选?项?的?是\s?([{options_concat}])',
            rf'说法错误选?项?的?是\s?([{options_concat}])',
            rf'([{options_concat}])\s?是正确的',
            rf'([{options_concat}])\s?是正确答案',
            rf'选项\s?([{options_concat}])\s?正确',
            rf'所以答\s?([{options_concat}])',
            rf'所以\s?([{options_concat}][.。$]?$)',
            rf'所有\s?([{options_concat}][.。$]?$)',
            rf'[\s，：:,]([{options_concat}])[。，,\.]?$',
            rf'[\s，,：:][故即]([{options_concat}])[。\.]?$',
            rf'[\s，,：:]因此([{options_concat}])[。\.]?$',
            rf'[是为。]\s?([{options_concat}])[。\.]?$',
            rf'因此\s?([{options_concat}])[。\.]?$',
            rf'显然\s?([{options_concat}])[。\.]?$',
            rf'答案是\s?(\S+)(?:。|$)',
            rf'答案应该是\s?(\S+)(?:。|$)',
            rf'答案为\s?(\S+)(?:。|$)',
            rf'答案是(.*?)[{options_concat}]',
            rf'答案为(.*?)[{options_concat}]',
            rf'固选(.*?)[{options_concat}]',
            rf'答案应该是(.*?)[{options_concat}]',
            rf'[Tt]he answer is \(?[{options_concat}]\)?',
            rf'[Tt]he correct answer is [{options_concat}]',
            rf'[Tt]he correct answer is:\n[{options_concat}]',
            rf'(\s|^)[{options_concat}][\s。，,\.$]',  # noqa
            rf'^选项\s?([{options_concat}])',
            rf'^([{options_concat}])\s?选?项',
            rf'(\s|^)[{options_concat}][\s。，,：:\.$]',
            rf'(\s|^)[{options_concat}](\s|$)',
            rf'[{options_concat}]',
        ]

        regexes = [re.compile(pattern) for pattern in patterns]
        for regex in regexes:
            match = regex.search(text)
            if match:
                outputs = match.group(0)
                for i in options:
                    if i in outputs:
                        return i
        # If no match found, try to find the last capital letter in the text
        last_capital = ResponseParser.parse_last_capital(text, options)
        if last_capital:
            return last_capital
        return 'No valid option found'

    @staticmethod
    def parse_first_option(text: str, options: list[str]) -> str:
        """
        Find first valid option for text.

        Args:
            text: The text to parse.
        """
        options_pattern = ResponseParser.process_options(options)

        patterns = [
            rf'[Aa]nswer:\s*({options_pattern})',
            rf'ANSWER:\s*({options_pattern})',
            rf'answer is \(?({options_pattern})\)?',
            rf'[Tt]he correct answer is:\s*({options_pattern})',
            rf'[Tt]he correct answer is:\n\s*({options_pattern})',
            rf'[Tt]he correct answer is:\n\n-\s*({options_pattern})',
            rf'[Tt]he answer might be:\n\n-\s*({options_pattern})',
            rf'[Tt]he answer is \s*({options_pattern})',
        ]

        regexes = [re.compile(pattern) for pattern in patterns]
        for regex in regexes:
            matches = regex.search(text)
            if matches:
                return matches.group(1)
        # If no match found, try to find the last capital letter in the text
        last_capital = ResponseParser.parse_last_capital(text, options)
        if last_capital:
            return last_capital
        return 'No valid option found'

    @staticmethod
    def parse_bracketed_answer(text: str, options: list[str]) -> str:
        options = ResponseParser.process_options(options)
        # Match the first occurrence of the options in angle brackets
        match = re.search(rf'<({options})>', text)
        if match:
            return match.group(1)
        return 'No valid option found'

    @staticmethod
    def process_options(options: list[str]) -> str:
        # Escape each option to ensure special characters in options are treated literally
        escaped_options = [re.escape(option) for option in options]
        # Join options into a regex pattern separated by '|', to match any of the options
        options_pattern = '|'.join(escaped_options)
        return options_pattern


if __name__ == '__main__':
    result = '**Answer: A **Answer: C**'
    options = ['A', 'B', 'C', 'D']
    parsed_result = ResponseParser.parse_first_option(result, options)
    print(f'Parsed result: {parsed_result}')  # Should print 'C'
