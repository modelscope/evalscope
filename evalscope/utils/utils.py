# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) OpenCompass.

import functools
import hashlib
import importlib
import importlib.util
import numpy as np
import os
import random
import re
import torch
from inspect import signature
from typing import Any, Dict, List, Tuple, Union

from evalscope.utils.logger import get_logger

logger = get_logger()

TEST_LEVEL_LIST = [0, 1]

# Example: export TEST_LEVEL_LIST=0,1
TEST_LEVEL_LIST_STR = 'TEST_LEVEL_LIST'


def test_level_list():
    global TEST_LEVEL_LIST
    if TEST_LEVEL_LIST_STR in os.environ:
        TEST_LEVEL_LIST = [int(x) for x in os.environ[TEST_LEVEL_LIST_STR].split(',')]

    return TEST_LEVEL_LIST


def get_obj_from_cfg(eval_class_ref: Any, *args, **kwargs) -> Any:
    module_name, spliter, cls_name = eval_class_ref.partition(':')

    try:
        obj_cls = importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f'{e}')
        raise e

    if spliter:
        for attr in cls_name.split('.'):
            obj_cls = getattr(obj_cls, attr)

    return functools.partial(obj_cls, *args, **kwargs)


def random_seeded_choice(seed: Union[int, str, float], choices, **kwargs):
    """Random choice with a (potentially string) seed."""
    return random.Random(seed).choices(choices, k=1, **kwargs)[0]


def gen_hash(name: str, bits: int = 32):
    return hashlib.md5(name.encode(encoding='UTF-8')).hexdigest()[:bits]


def dict_torch_dtype_to_str(d: Dict[str, Any]) -> dict:
    """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.

        Refer to: https://github.com/huggingface/transformers/pull/16065/files for details.
        """
    if d.get('torch_dtype', None) is not None and not isinstance(d['torch_dtype'], str):
        d['torch_dtype'] = str(d['torch_dtype']).split('.')[1]

    for value in d.values():
        if isinstance(value, dict):
            dict_torch_dtype_to_str(value)

    return d


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

def normalize_score(score: Union[float, dict], keep_num: int = 4) -> Union[float, dict]:
    """
    Normalize score.

    Args:
        score: input score, could be float or dict. e.g. 0.12345678 or {'acc': 0.12345678, 'f1': 0.12345678}
        keep_num: number of digits to keep.

    Returns:
        Union[float, dict]: normalized score. e.g. 0.1234 or {'acc': 0.1234, 'f1': 0.1234}
    """
    if isinstance(score, float):
        score = round(score, keep_num)
    elif isinstance(score, dict):
        score = {k: round(v, keep_num) for k, v in score.items()}
    else:
        logger.warning(f'Unknown score type: {type(score)}')

    return score


def is_module_installed(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_module_path(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec and spec.origin:
        return os.path.abspath(spec.origin)
    else:
        raise ValueError(f'Cannot find module: {module_name}')


def get_valid_list(input_list, candidate_list):
    """
    Get the valid and invalid list from input_list based on candidate_list.
    Args:
        input_list: The input list.
        candidate_list: The candidate list.

    Returns:
        valid_list: The valid list.
        invalid_list: The invalid list.
    """
    return [i for i in input_list if i in candidate_list], \
           [i for i in input_list if i not in candidate_list]


def get_latest_folder_path(work_dir):
    from datetime import datetime

    # Get all subdirectories in the work_dir
    folders = [f for f in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, f))]

    # Get the timestamp（YYYYMMDD_HHMMSS）
    timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')

    # Filter out the folders
    timestamped_folders = [f for f in folders if timestamp_pattern.match(f)]

    if not timestamped_folders:
        print(f'>> No timestamped folders found in {work_dir}!')
        return None

    # timestamp parser
    def parse_timestamp(folder_name):
        return datetime.strptime(folder_name, '%Y%m%d_%H%M%S')

    # Find the latest folder
    latest_folder = max(timestamped_folders, key=parse_timestamp)

    return os.path.join(work_dir, latest_folder)


def csv_to_list(file_path: str) -> List[dict]:
    import csv

    with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        result = [row for row in csv_reader]

    return result


def seed_everything(seed: int):
    """Set all random seeds to a fixed value for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_supported_params(func):
    """Get the supported parameters of a function."""
    sig = signature(func)
    return list(sig.parameters.keys())

def parse_int_or_float(num):
    number = float(num)
    if number.is_integer():
        return int(number)
    return number

if __name__ == '__main__':
    options = ['A', 'B', 'C', 'D']
    answers = ['Context .... ANSWER: A', 'answer: A']
    for answer in answers:
        print(ResponseParser.parse_first_option(answer, options))
