# Copyright (c) Alibaba, Inc. and its affiliates.
"""Utility modules for EvalScope service."""

from .benchmarks import (
    DEFAULT_MULTIMODAL_BENCHMARKS,
    DEFAULT_TEXT_BENCHMARKS,
    build_benchmark_entry,
    parse_benchmark_description,
)
from .log import OUTPUT_DIR, create_log_file, get_log_content, validate_task_id
from .process import run_eval_wrapper, run_in_subprocess, run_perf_wrapper

__all__ = [
    'OUTPUT_DIR',
    'create_log_file',
    'get_log_content',
    'validate_task_id',
    'run_eval_wrapper',
    'run_perf_wrapper',
    'run_in_subprocess',
    'DEFAULT_TEXT_BENCHMARKS',
    'DEFAULT_MULTIMODAL_BENCHMARKS',
    'build_benchmark_entry',
    'parse_benchmark_description',
]
