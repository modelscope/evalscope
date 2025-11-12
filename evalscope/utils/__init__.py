# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from .import_utils import _LazyModule

if TYPE_CHECKING:
    from .argument_utils import BaseArgument, get_supported_params, parse_int_or_float
    from .deprecation_utils import deprecated
    from .function_utils import run_once, thread_safe
    from .import_utils import get_module_path, is_module_installed
    from .io_utils import (
        OutputsStructure,
        csv_to_jsonl,
        csv_to_list,
        dict_to_yaml,
        gen_hash,
        get_latest_folder_path,
        get_valid_list,
        json_to_dict,
        jsonl_to_csv,
        jsonl_to_list,
        safe_filename,
        yaml_to_dict,
    )
    from .logger import configure_logging, get_logger
    from .model_utils import EvalBackend, dict_torch_dtype_to_str, fix_do_sample_warning, get_device, seed_everything

else:
    _import_structure = {
        'argument_utils': [
            'BaseArgument',
            'parse_int_or_float',
            'get_supported_params',
        ],
        'model_utils': [
            'EvalBackend',
            'get_device',
            'seed_everything',
            'dict_torch_dtype_to_str',
            'fix_do_sample_warning',
        ],
        'import_utils': [
            'is_module_installed',
            'get_module_path',
        ],
        'function_utils': [
            'thread_safe',
            'run_once',
        ],
        'io_utils': [
            'OutputsStructure',
            'csv_to_list',
            'json_to_dict',
            'yaml_to_dict',
            'get_latest_folder_path',
            'gen_hash',
            'dict_to_yaml',
            'csv_to_jsonl',
            'jsonl_to_csv',
            'jsonl_to_list',
            'gen_hash',
            'get_valid_list',
            'safe_filename',
            'thread_safe',
        ],
        'deprecation_utils': [
            'deprecated',
        ],
        'logger': [
            'get_logger',
            'configure_logging',
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
