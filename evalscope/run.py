# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Run evaluation for LLMs.
"""
import os.path
import torch
from argparse import Namespace
from datetime import datetime
from typing import List, Optional, Union

from evalscope.arguments import parse_args
from evalscope.benchmarks import Benchmark, BenchmarkMeta
from evalscope.config import TaskConfig, parse_task_config
from evalscope.constants import DEFAULT_MODEL_REVISION, DEFAULT_WORK_DIR, EvalBackend, EvalType
from evalscope.evaluator import Evaluator
from evalscope.models import CustomModel, LocalModel
from evalscope.utils import seed_everything
from evalscope.utils.io_utils import OutputsStructure, are_paths_same
from evalscope.utils.logger import configure_logging, get_logger

logger = get_logger()


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
    base_model = get_base_model(task_cfg)
    for dataset_name in task_cfg.datasets:
        evaluator = create_evaluator(task_cfg, dataset_name, outputs, base_model)
        res_dict = evaluator.eval(infer_cfg=task_cfg.generation_config, debug=task_cfg.debug, limit=task_cfg.limit)
        eval_results[dataset_name] = res_dict

    return eval_results


def create_evaluator(task_cfg: TaskConfig, dataset_name: str, outputs: OutputsStructure, base_model: LocalModel):
    """Create an evaluator object for the specified dataset."""
    benchmark: BenchmarkMeta = Benchmark.get(dataset_name)

    data_adapter = benchmark.get_data_adapter(config=task_cfg.dataset_args)
    model_adapter = initialize_model_adapter(task_cfg, benchmark.model_adapter, base_model)

    return Evaluator(
        dataset_name_or_path=benchmark.dataset_id,
        data_adapter=data_adapter,
        subset_list=benchmark.subset_list,
        model_adapter=model_adapter,
        outputs=outputs,
        task_cfg=task_cfg,
    )


def get_base_model(task_cfg: TaskConfig) -> Optional[LocalModel]:
    """Get the base local model for the task. If the task is not checkpoint-based, return None.
       Avoids loading model multiple times for different datasets.
    """
    if task_cfg.eval_type != EvalType.CHECKPOINT:
        return None
    else:
        device_map = task_cfg.model_args.get('device_map', 'auto') if torch.cuda.is_available() else None
        cache_dir = task_cfg.model_args.get('cache_dir', None)
        model_precision = task_cfg.model_args.get('precision', torch.float16)
        model_revision = task_cfg.model_args.get('revision', DEFAULT_MODEL_REVISION)
        if isinstance(model_precision, str) and model_precision != 'auto':
            model_precision = eval(model_precision)

        base_model = LocalModel(
            model_id=task_cfg.model,
            model_revision=model_revision,
            device_map=device_map,
            torch_dtype=model_precision,
            cache_dir=cache_dir)
        return base_model


def initialize_model_adapter(task_cfg: TaskConfig, model_adapter_cls, base_model: LocalModel):
    """Initialize the model adapter based on the task configuration."""
    if task_cfg.dry_run:
        from evalscope.models.model import DummyChatModel
        return DummyChatModel(model_cfg=dict())
    elif task_cfg.eval_type == EvalType.CUSTOM:
        if not isinstance(task_cfg.model, CustomModel):
            raise ValueError(f'Expected evalscope.models.custom.CustomModel, but got {type(task_cfg.model)}.')
        from evalscope.models import CustomModelAdapter
        return CustomModelAdapter(custom_model=task_cfg.model)
    elif task_cfg.eval_type == EvalType.SERVICE:
        from evalscope.models import ServerModelAdapter
        return ServerModelAdapter(api_url=task_cfg.api_url, model_id=task_cfg.model, api_key=task_cfg.api_key)
    else:
        return model_adapter_cls(
            model=base_model or get_base_model(task_cfg),
            generation_config=task_cfg.generation_config,
            chat_template=task_cfg.chat_template)


def main():
    args = parse_args()
    run_task(args)


if __name__ == '__main__':
    main()
