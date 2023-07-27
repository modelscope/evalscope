# Copyright (c) Alibaba, Inc. and its affiliates.

import functools
import importlib
import os
import random
from typing import Any, Union

import jsonlines as jsonl
import pandas as pd
import pyarrow as pa
import yaml

from llmuses.constants import DumpMode
from llmuses.utils.logger import get_logger

logger = get_logger()

TEST_LEVEL_LIST = [0, 1]

# Example: export TEST_LEVEL_LIST=0,1
TEST_LEVEL_LIST_STR = 'TEST_LEVEL_LIST'


def test_level_list():
    global TEST_LEVEL_LIST
    if TEST_LEVEL_LIST_STR in os.environ:
        TEST_LEVEL_LIST = [
            int(x) for x in os.environ[TEST_LEVEL_LIST_STR].split(',')
        ]

    return TEST_LEVEL_LIST


def jsonl_to_list(jsonl_file):
    """
    Read jsonl file to list.

    Args:
        jsonl_file: jsonl file path.

    Returns:
        list: list of lines. Each line is a dict.
    """
    res_list = []
    with jsonl.open(jsonl_file, mode='r') as reader:
        for line in reader.iter(
                type=dict, allow_none=True, skip_invalid=False):
            res_list.append(line)
    return res_list


def jsonl_to_reader(jsonl_file):
    """
    Read jsonl file to reader object.

    Args:
        jsonl_file: jsonl file path.

    Returns:
        reader: jsonl reader object.
    """
    with jsonl.open(jsonl_file, mode='r') as reader:
        return reader


def jsonl_to_csv():
    pass


def jsonl_dump_data(data_list, jsonl_file, dump_mode=DumpMode.OVERWRITE):
    """
    Dump data to jsonl file.

    Args:
        data_list: data list to be dumped.  [{'a': 'aaa'}, ...]
        jsonl_file: jsonl file path.
        dump_mode: dump mode. It can be 'overwrite' or 'append'.
    """
    if not jsonl_file:
        raise ValueError('output file must be provided.')

    jsonl_file = os.path.expanduser(jsonl_file)

    if dump_mode == DumpMode.OVERWRITE:
        dump_mode = 'w'
    elif dump_mode == DumpMode.APPEND:
        dump_mode = 'a'
    with jsonl.open(jsonl_file, mode=dump_mode) as writer:
        writer.write_all(data_list)
    logger.info(f'Dump data to {jsonl_file} successfully.')


def yaml_to_dict(yaml_file) -> dict:
    """
    Read yaml file to dict.
    """
    with open(yaml_file, 'r') as f:
        try:
            stream = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(e)
            raise e

    return stream


def get_obj_from_cfg(eval_class_ref: Any, *args, **kwargs) -> Any:
    module_name, spliter, cls_name = eval_class_ref.partition(':')

    try:
        obj_cls = importlib.import_module(module_name)
    except ImportError as e:
        logger.error(e)
        raise e

    if spliter:
        for attr in cls_name.split('.'):
            obj_cls = getattr(obj_cls, attr)

    return functools.partial(obj_cls, *args, **kwargs)


def markdown_table(header_l, data_l):
    md_str = f'| {" | ".join(header_l)} |'
    md_str += f'\n| {" | ".join(["---"] * len(header_l))} |'
    for data in data_l:
        if isinstance(data, str):
            data = [data]
        assert len(data) <= len(header_l)
        tmp = data + [''] * (len(header_l) - len(data))
        md_str += f'\n| {" | ".join(tmp)} |'
    return md_str


def random_seeded_choice(seed: Union[int, str, float], choices, **kwargs):
    """Random choice with a (potentially string) seed."""
    return random.Random(seed).choices(choices, k=1, **kwargs)[0]
