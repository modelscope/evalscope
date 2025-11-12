# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import importlib
import os
import time

from evalscope.utils import get_logger

logger = get_logger()

# Using glob to find all files matching the pattern
pattern = os.path.join(os.path.dirname(__file__), '*', '**', '*_adapter.py')
files = glob.glob(pattern, recursive=True)

import_times = []

for file_path in files:
    if file_path.endswith('.py') and not os.path.basename(file_path).startswith('_'):
        # Convert file path to a module path
        relative_path = os.path.relpath(file_path, os.path.dirname(__file__))
        module_path = relative_path[:-3].replace(os.path.sep, '.')  # strip '.py' and convert to module path
        full_path = f'evalscope.benchmarks.{module_path}'

        start_time = time.perf_counter()
        importlib.import_module(full_path)
        end_time = time.perf_counter()

        import_times.append((full_path, end_time - start_time))

# Sort by import time in descending order
import_times.sort(key=lambda x: x[1], reverse=True)

# Log the sorted import times
for module, duration in import_times:
    logger.debug(f'Module {module} imported in {duration:.6f} seconds')
