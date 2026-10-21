# Copyright (c) Alibaba, Inc. and its affiliates.
"""Utility modules for EvalScope service."""

from .benchmarks import (
    DEFAULT_MULTIMODAL_BENCHMARKS,
    DEFAULT_TEXT_BENCHMARKS,
    build_benchmark_entry,
    discover_all_benchmarks,
    parse_benchmark_description,
)
from .log import OUTPUT_DIR, create_log_file, get_log_content, read_task_manifest, validate_task_id, write_task_manifest
from .process import (
    ActiveTask,
    TaskStoppedError,
    active_task_ids,
    list_active_tasks,
    run_eval_wrapper,
    run_in_subprocess,
    run_perf_wrapper,
    sanitize_endpoint,
    serialize_result,
    stop_process,
)

__all__ = [
    'OUTPUT_DIR',
    'create_log_file',
    'get_log_content',
    'validate_task_id',
    'read_task_manifest',
    'write_task_manifest',
    'active_task_ids',
    'list_active_tasks',
    'sanitize_endpoint',
    'ActiveTask',
    'run_eval_wrapper',
    'run_perf_wrapper',
    'serialize_result',
    'stop_process',
    'run_in_subprocess',
    'TaskStoppedError',
    'DEFAULT_TEXT_BENCHMARKS',
    'DEFAULT_MULTIMODAL_BENCHMARKS',
    'build_benchmark_entry',
    'discover_all_benchmarks',
    'parse_benchmark_description',
]
