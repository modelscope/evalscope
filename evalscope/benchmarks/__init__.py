# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import importlib
import os

from evalscope.benchmarks.benchmark import Benchmark, BenchmarkMeta
from evalscope.benchmarks.data_adapter import DataAdapter
from evalscope.utils import get_logger

logger = get_logger()

# Using glob to find all files matching the pattern
pattern = os.path.join(os.path.dirname(__file__), '*', '**', '*_adapter.py')
files = glob.glob(pattern, recursive=True)

for file_path in files:
    if file_path.endswith('.py') and not os.path.basename(file_path).startswith('_'):
        # Convert file path to a module path
        relative_path = os.path.relpath(file_path, os.path.dirname(__file__))
        module_path = relative_path[:-3].replace(os.path.sep, '.')  # strip '.py' and convert to module path
        full_path = f'evalscope.benchmarks.{module_path}'
        importlib.import_module(full_path)
        # print(f'Importing {full_path}')
