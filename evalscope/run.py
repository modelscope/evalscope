# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Run evaluation for LLMs.
"""
import logging
import os.path
import torch
from argparse import Namespace
from datetime import datetime
from typing import List, Optional, Union

from evalscope.arguments import parse_args
from evalscope.config import TaskConfig, parse_task_config
from evalscope.constants import DEFAULT_MODEL_REVISION, DEFAULT_WORK_DIR, EvalBackend, EvalType
from evalscope.evaluator import Evaluator
from evalscope.models.custom import CustomModel
from evalscope.utils import import_module_util, seed_everything
from evalscope.utils.io_utils import OutputsStructure, are_paths_same
from evalscope.utils.logger import configure_logging, get_logger

logger = get_logger()

BENCHMARK_PATH_PREFIX = 'evalscope.benchmarks.'
MEMBERS_TO_IMPORT = ['DATASET_ID', 'SUBSET_LIST', 'DataAdapterClass', 'ModelAdapterClass']


def run_task(task_cfg: Union[str, dict, TaskConfig, List[TaskConfig], Namespace]) -> Union[dict, List[dict]]:
    """Run evaluation task(s) based on the provided configuration."""
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # If task_cfg is a list, run each task individually
    if isinstance(task_cfg, list):
        return [run_single_task(cfg, run_time) for cfg in task_cfg]

    task_cfg = parse_task_config(task_cfg)
    return run_single_task(task_cfg, run_time)


def run_single_task(task_cfg: TaskConfig, run_time: str) -> dict:
    """Run a single evaluation task."""
    seed_everything(task_cfg.seed)
    outputs = setup_work_directory(task_cfg, run_time)
    configure_logging(task_cfg.debug, os.path.join(outputs.logs_dir, 'eval_log.log'))

    task_cfg.dump_yaml(outputs.configs_dir)
    logger.info(task_cfg)

    if task_cfg.eval_backend != EvalBackend.NATIVE:
        return run_non_native_backend(task_cfg)
    else:
        return evaluate_model(task_cfg, outputs)


def setup_work_directory(task_cfg: TaskConfig, run_time: str):
    """Set the working directory for the task."""
    if task_cfg.use_cache:
        task_cfg.work_dir = task_cfg.use_cache
        logger.info(f'Set resume from {task_cfg.work_dir}')
    elif are_paths_same(task_cfg.work_dir, DEFAULT_WORK_DIR):
        task_cfg.work_dir = os.path.join(task_cfg.work_dir, run_time)

    outputs = OutputsStructure(outputs_dir=task_cfg.work_dir)

    if task_cfg.eval_backend == EvalBackend.OPEN_COMPASS:
        task_cfg.eval_config['time_str'] = run_time
    elif task_cfg.eval_backend == EvalBackend.VLM_EVAL_KIT:
        task_cfg.eval_config['work_dir'] = task_cfg.work_dir
    return outputs


def run_non_native_backend(task_cfg: TaskConfig) -> dict:
    """Run evaluation using a non-native backend."""
    eval_backend = task_cfg.eval_backend
    eval_config = task_cfg.eval_config

    if eval_config is None:
        logger.warning(f'Got eval_backend {eval_backend}, but eval_config is not provided.')

    backend_manager_class = get_backend_manager_class(eval_backend)
    backend_manager = backend_manager_class(config=eval_config)
    backend_manager.run()

    return dict()


def get_backend_manager_class(eval_backend: EvalBackend):
    """Get the backend manager class based on the evaluation backend."""
    if eval_backend == EvalBackend.OPEN_COMPASS:
        from evalscope.backend.opencompass import OpenCompassBackendManager
        return OpenCompassBackendManager
    elif eval_backend == EvalBackend.VLM_EVAL_KIT:
        from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
        return VLMEvalKitBackendManager
    elif eval_backend == EvalBackend.RAG_EVAL:
        from evalscope.backend.rag_eval import RAGEvalBackendManager
        return RAGEvalBackendManager
    elif eval_backend == EvalBackend.THIRD_PARTY:
        raise NotImplementedError(f'Not implemented for evaluation backend {eval_backend}')


def evaluate_model(task_cfg: TaskConfig, outputs: OutputsStructure) -> dict:
    """Evaluate the model based on the provided task configuration."""
    # Initialize evaluator
    eval_results = {}

    for dataset_name in task_cfg.datasets:
        evaluator = create_evaluator(task_cfg, dataset_name, outputs)
        res_dict = evaluator.eval(infer_cfg=task_cfg.generation_config, debug=task_cfg.debug, limit=task_cfg.limit)
        eval_results[dataset_name] = res_dict

    return eval_results


def create_evaluator(task_cfg: TaskConfig, dataset_name: str, outputs: OutputsStructure):
    """Create an evaluator object for the specified dataset."""
    imported_modules = import_module_util(BENCHMARK_PATH_PREFIX, dataset_name, MEMBERS_TO_IMPORT)
    model_adapter = initialize_model_adapter(task_cfg, dataset_name, imported_modules)

    dataset_config = task_cfg.dataset_args.get(dataset_name, {})
    dataset_name_or_path = dataset_config.get('local_path') or imported_modules['DATASET_ID']
    in_prompt_template = dataset_config.get('prompt_template', '')
    few_shot_num = dataset_config.get('few_shot_num', None)
    few_shot_random = dataset_config.get('few_shot_random', True)

    data_adapter = imported_modules['DataAdapterClass'](
        few_shot_num=few_shot_num,
        few_shot_random=few_shot_random,
        prompt_template=in_prompt_template,
        outputs=outputs,
    )
    in_subset_list = dataset_config.get('subset_list', imported_modules['SUBSET_LIST'])

    logger.info(f'Evaluating on subsets for {dataset_name}: {in_subset_list}\n')

    return Evaluator(
        dataset_name_or_path=dataset_name_or_path,
        subset_list=in_subset_list,
        data_adapter=data_adapter,
        model_adapter=model_adapter,
        use_cache=task_cfg.use_cache,
        outputs=outputs,
        datasets_dir=task_cfg.dataset_dir,
        datasets_hub=task_cfg.dataset_hub,
        stage=task_cfg.stage,
        eval_type=task_cfg.eval_type,
        overall_task_cfg=task_cfg,
    )


def initialize_model_adapter(task_cfg: TaskConfig, dataset_name: str, imported_modules):
    """Initialize the model adapter based on the task configuration."""
    if task_cfg.dry_run:
        from evalscope.models.dummy_chat_model import DummyChatModel
        return DummyChatModel(model_cfg=dict())
    elif task_cfg.eval_type == EvalType.CUSTOM:
        if not isinstance(task_cfg.model, CustomModel):
            raise ValueError(f'Expected evalscope.models.custom.CustomModel, but got {type(task_cfg.model)}.')
        from evalscope.models.model_adapter import CustomModelAdapter
        return CustomModelAdapter(custom_model=task_cfg.model)
    else:
        device_map = task_cfg.model_args.get('device_map', 'auto') if torch.cuda.is_available() else None
        model_precision = task_cfg.model_args.get('precision', torch.float16)
        if isinstance(model_precision, str) and model_precision != 'auto':
            model_precision = eval(model_precision)
        return imported_modules['ModelAdapterClass'](
            model_id=task_cfg.model,
            model_revision=task_cfg.model_args.get('revision', DEFAULT_MODEL_REVISION),
            device_map=device_map,
            torch_dtype=model_precision,
            generation_config=task_cfg.generation_config,
            chat_template=task_cfg.chat_template)


def main():
    args = parse_args()
    run_task(args)


if __name__ == '__main__':
    main()
