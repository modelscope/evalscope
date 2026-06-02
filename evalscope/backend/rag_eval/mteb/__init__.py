# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.backend.rag_eval.mteb.arguments import CustomTaskConfig, MTEBEvalConfig, MTEBModelConfig, MTEBToolConfig
from evalscope.backend.rag_eval.mteb.custom_task import build_custom_task
from evalscope.backend.rag_eval.mteb.data_loader import patch_tasks_for_modelscope
from evalscope.backend.rag_eval.mteb.task_template import one_stage_eval, run_mteb_eval, two_stage_eval

__all__ = [
    'CustomTaskConfig',
    'MTEBModelConfig',
    'MTEBEvalConfig',
    'MTEBToolConfig',
    'run_mteb_eval',
    'one_stage_eval',
    'two_stage_eval',
    'patch_tasks_for_modelscope',
    'build_custom_task',
]
