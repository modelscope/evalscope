from typing import Optional, Union
from evalscope.utils import is_module_installed, get_valid_list
from evalscope.backend.base import BackendManager
from evalscope.utils.logger import get_logger
from functools import partial
import subprocess
import copy


logger = get_logger()

