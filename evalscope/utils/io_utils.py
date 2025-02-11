import json
import jsonlines as jsonl
import os
import yaml

from evalscope.constants import DumpMode
from evalscope.utils.logger import get_logger

logger = get_logger()


class OutputsStructure:
    LOGS_DIR = 'logs'
    PREDICTIONS_DIR = 'predictions'
    REVIEWS_DIR = 'reviews'
    REPORTS_DIR = 'reports'
    CONFIGS_DIR = 'configs'

    def __init__(self, outputs_dir: str, is_make=True):
        self.outputs_dir = outputs_dir
        self.is_make = is_make
        self._dirs = {
            'logs_dir': None,
            'predictions_dir': None,
            'reviews_dir': None,
            'reports_dir': None,
            'configs_dir': None
        }

    def _get_dir(self, attr_name, dir_name):
        if self._dirs[attr_name] is None:
            dir_path = os.path.join(self.outputs_dir, dir_name)
            if self.is_make:
                os.makedirs(dir_path, exist_ok=True)
            self._dirs[attr_name] = dir_path
        return self._dirs[attr_name]

    @property
    def logs_dir(self):
        return self._get_dir('logs_dir', OutputsStructure.LOGS_DIR)

    @property
    def predictions_dir(self):
        return self._get_dir('predictions_dir', OutputsStructure.PREDICTIONS_DIR)

    @property
    def reviews_dir(self):
        return self._get_dir('reviews_dir', OutputsStructure.REVIEWS_DIR)

    @property
    def reports_dir(self):
        return self._get_dir('reports_dir', OutputsStructure.REPORTS_DIR)

    @property
    def configs_dir(self):
        return self._get_dir('configs_dir', OutputsStructure.CONFIGS_DIR)


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
        for line in reader.iter(type=dict, allow_none=True, skip_invalid=False):
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


def dump_jsonl_data(data_list, jsonl_file, dump_mode=DumpMode.OVERWRITE):
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

    if not isinstance(data_list, list):
        data_list = [data_list]

    if dump_mode == DumpMode.OVERWRITE:
        dump_mode = 'w'
    elif dump_mode == DumpMode.APPEND:
        dump_mode = 'a'
    with jsonl.open(jsonl_file, mode=dump_mode) as writer:
        writer.write_all(data_list)


def jsonl_to_csv():
    pass


def yaml_to_dict(yaml_file) -> dict:
    """
    Read yaml file to dict.
    """
    with open(yaml_file, 'r') as f:
        try:
            stream = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f'{e}')
            raise e

    return stream


def dict_to_yaml(d: dict, yaml_file: str):
    """
    Dump dict to yaml file.
    """
    with open(yaml_file, 'w') as f:
        yaml.dump(d, f, default_flow_style=False, allow_unicode=True)


def json_to_dict(json_file) -> dict:
    """
    Read json file to dict.
    """
    with open(json_file, 'r') as f:
        try:
            stream = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f'{e}')
            raise e

    return stream


def are_paths_same(path1, path2):
    """
    Check if two paths are the same.
    """
    real_path1 = os.path.realpath(os.path.abspath(os.path.expanduser(path1)))
    real_path2 = os.path.realpath(os.path.abspath(os.path.expanduser(path2)))

    return real_path1 == real_path2


def dict_to_json(d: dict, json_file: str):
    """
    Dump dict to json file.
    """
    with open(json_file, 'w') as f:
        json.dump(d, f, indent=4, ensure_ascii=False)
