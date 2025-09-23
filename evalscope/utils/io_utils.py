import base64
import csv
import hashlib
import io
import json
import jsonlines as jsonl
import os
import re
import string
import unicodedata
import yaml
from datetime import datetime
from io import BytesIO
from PIL import Image

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

    def _get_dir(self, attr_name, dir_name) -> str:
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
    try:
        res_list = []
        with jsonl.open(jsonl_file, mode='r') as reader:
            for line in reader.iter(type=dict, allow_none=True, skip_invalid=False):
                res_list.append(line)
    except Exception:
        # Fallback to reading line by line
        res_list = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    res_list.append(json.loads(line.strip()))
    if not res_list:
        logger.warning(f'No data found in {jsonl_file}.')
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

    # Convert non-serializable types to serializable ones
    data_list = convert_normal_types(data_list)

    if dump_mode == DumpMode.OVERWRITE:
        dump_mode = 'w'
    elif dump_mode == DumpMode.APPEND:
        dump_mode = 'a'
    with jsonl.open(jsonl_file, mode=dump_mode) as writer:
        writer.write_all(data_list)


def jsonl_to_csv(jsonl_file, csv_file):
    """
    Convert jsonl file to csv file.

    Args:
        jsonl_file: jsonl file path.
        csv_file: csv file path.
    """
    data = jsonl_to_list(jsonl_file)
    if not data:
        logger.warning(f'No data found in {jsonl_file}.')
        return

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data[0].keys())  # Write header
        for item in data:
            writer.writerow(item.values())


def csv_to_list(csv_file) -> list:
    """
    Read csv file to list.

    Args:
        csv_file: csv file path.

    Returns:
        list: list of lines. Each line is a dict.
    """
    res_list = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            res_list.append(row)
    return res_list


def csv_to_jsonl(csv_file, jsonl_file):
    """
    Convert csv file to jsonl file.

    Args:
        csv_file: csv file path.
        jsonl_file: jsonl file path.
    """
    data = csv_to_list(csv_file)
    if not data:
        logger.warning(f'No data found in {csv_file}.')
        return

    dump_jsonl_data(data, jsonl_file, dump_mode=DumpMode.OVERWRITE)


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


def gen_hash(name: str, bits: int = 32):
    return hashlib.md5(name.encode(encoding='UTF-8')).hexdigest()[:bits]


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


def PIL_to_base64(image: Image.Image, format: str = 'JPEG', add_header: bool = False) -> str:
    """
    Convert a PIL Image to a base64 encoded string.

    Args:
        image (Image.Image): The PIL Image to convert.
        format (str): The format to save the image in. Default is 'JPEG'.
        add_header (bool): Whether to add the base64 header. Default is False.

    Returns:
        str: Base64 encoded string of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    if add_header:
        img_str = f'data:image/{format.lower()};base64,{img_str}'
    return img_str


def bytes_to_base64(bytes_data: bytes, *, format: str = 'png', add_header: bool = False, content_type='image') -> str:
    """Convert bytes to a base64 encoded string.

    Args:
        bytes_data (bytes): The bytes to convert.
        format (str): The format of the image. Default is 'png'.
        add_header (bool): Whether to add the base64 header. Default is False.
        content_type (str): The type of the data, 'image' or 'audio'. Default is 'image'.

    Returns:
        str: Base64 encoded string of the bytes.
    """
    base64_str = base64.b64encode(bytes_data).decode('utf-8')
    if add_header:
        base64_str = f'data:{content_type}/{format};base64,{base64_str}'
    return base64_str


def base64_to_PIL(base64_str):
    """Convert a base64 encoded string to a PIL Image.

    Args:
        base64_str (str): The base64 encoded string.

    Returns:
        Image.Image: The decoded PIL Image.
    """
    # remove header
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]

    # decode
    img_data = base64.b64decode(base64_str)
    img_file = io.BytesIO(img_data)
    img = Image.open(img_file)
    return img


def safe_filename(s: str, max_length: int = 255) -> str:
    """
    Convert a string into a safe filename by removing or replacing unsafe characters.

    Args:
        s (str): The input string to convert
        max_length (int): Maximum length of the resulting filename (default 255)

    Returns:
        str: A safe filename string

    Examples:
        >>> safe_filename("Hello/World?.txt")
        'Hello_World.txt'
    """
    # normalize unicode characters
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ASCII', 'ignore').decode('ASCII')

    # remove or replace unsafe characters
    # Keep only alphanumeric characters, dots, dashes, and underscores
    safe_chars = string.ascii_letters + string.digits + '.-_'
    s = ''.join(c if c in safe_chars else '_' for c in s)

    # remove consecutive underscores
    s = re.sub(r'_+', '_', s)

    # remove leading/trailing periods and underscores
    s = s.strip('._')

    # handle empty string case
    if not s:
        s = 'untitled'

    # handle starting with a period (hidden files)
    if s.startswith('.'):
        s = '_' + s

    # enforce length limit
    if len(s) > max_length:
        # If we need to truncate, preserve the file extension if present
        name, ext = os.path.splitext(s)
        ext_len = len(ext)
        if ext_len > 0:
            max_name_length = max_length - ext_len
            s = name[:max_name_length] + ext
        else:
            s = s[:max_length]

    return s


def convert_normal_types(obj):
    """Recursively convert numpy types and datetime objects to native Python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_normal_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_normal_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_normal_types(item) for item in obj)
    else:
        return obj
