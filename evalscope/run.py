# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Run evaluation for LLMs.
"""
import os
from argparse import Namespace
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

from evalscope.config import TaskConfig, parse_task_config
from evalscope.constants import DEFAULT_WORK_DIR, DataCollection, EvalBackend
from evalscope.utils.io_utils import OutputsStructure, are_paths_same
from evalscope.utils.logger import configure_logging, get_logger
from evalscope.utils.model_utils import seed_everything

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
    if task_cfg.seed is not None:
        seed_everything(task_cfg.seed)
    outputs = setup_work_directory(task_cfg, run_time)
    configure_logging(task_cfg.debug, os.path.join(outputs.logs_dir, 'eval_log.log'))

    if task_cfg.eval_backend != EvalBackend.NATIVE:
        result = run_non_native_backend(task_cfg, outputs)
    else:
        logger.info('Running with native backend')
        result = evaluate_model(task_cfg, outputs)

        logger.info(f'Finished evaluation for {task_cfg.model_id} on {task_cfg.datasets}')
        logger.info(f'Output directory: {outputs.outputs_dir}')

    return result


def setup_work_directory(task_cfg: TaskConfig, run_time: str):
    """Set the working directory for the task."""
    # use cache
    if task_cfg.use_cache:
        task_cfg.work_dir = task_cfg.use_cache
        logger.info(f'Set resume from {task_cfg.work_dir}')
    elif not task_cfg.no_timestamp:
        task_cfg.work_dir = os.path.join(task_cfg.work_dir, run_time)

    outputs = OutputsStructure(outputs_dir=task_cfg.work_dir)

    # Unify the output directory structure
    if task_cfg.eval_backend == EvalBackend.OPEN_COMPASS:
        task_cfg.eval_config['time_str'] = run_time
    elif task_cfg.eval_backend == EvalBackend.VLM_EVAL_KIT:
        task_cfg.eval_config['work_dir'] = task_cfg.work_dir
    elif task_cfg.eval_backend == EvalBackend.RAG_EVAL:
        from evalscope.backend.rag_eval import Tools
        if task_cfg.eval_config['tool'].lower() == Tools.MTEB:
            task_cfg.eval_config['eval']['output_folder'] = task_cfg.work_dir
        elif task_cfg.eval_config['tool'].lower() == Tools.CLIP_BENCHMARK:
            task_cfg.eval_config['eval']['output_dir'] = task_cfg.work_dir
    return outputs


def run_non_native_backend(task_cfg: TaskConfig, outputs: OutputsStructure) -> dict:
    """Run evaluation using a non-native backend."""
    eval_backend = task_cfg.eval_backend
    eval_config = task_cfg.eval_config

    if eval_config is None:
        logger.warning(f'Got eval_backend {eval_backend}, but eval_config is not provided.')

    backend_manager_class = get_backend_manager_class(eval_backend)
    backend_manager = backend_manager_class(config=eval_config)

    task_cfg.dump_yaml(outputs.configs_dir)
    logger.info(task_cfg)

    backend_manager.run()

    return dict()


def get_backend_manager_class(eval_backend: EvalBackend):
    """Get the backend manager class based on the evaluation backend."""
    if eval_backend == EvalBackend.OPEN_COMPASS:
        logger.info('Using OpenCompassBackendManager')
        from evalscope.backend.opencompass import OpenCompassBackendManager
        return OpenCompassBackendManager
    elif eval_backend == EvalBackend.VLM_EVAL_KIT:
        logger.info('Using VLMEvalKitBackendManager')
        from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
        return VLMEvalKitBackendManager
    elif eval_backend == EvalBackend.RAG_EVAL:
        logger.info('Using RAGEvalBackendManager')
        from evalscope.backend.rag_eval import RAGEvalBackendManager
        return RAGEvalBackendManager
    elif eval_backend == EvalBackend.THIRD_PARTY:
        raise NotImplementedError(f'Not implemented for evaluation backend {eval_backend}')


def evaluate_model(task_config: TaskConfig, outputs: OutputsStructure) -> dict:
    """Evaluate the model based on the provided task configuration."""
    from evalscope.api.evaluator import Evaluator
    from evalscope.api.model.lazy_model import LazyModel
    from evalscope.api.registry import get_benchmark
    from evalscope.evaluator import DefaultEvaluator
    from evalscope.report import gen_table

    # Initialize evaluator
    eval_results = {}
    # Initialize model with lazy loading (model will only be loaded when actually needed)
    model = LazyModel(task_config=task_config)
    # Initialize evaluators for each dataset
    evaluators: List[Evaluator] = []
    for dataset_name in task_config.datasets:
        # Create evaluator for each dataset
        benchmark = get_benchmark(dataset_name, task_config)
        evaluator = DefaultEvaluator(
            task_config=task_config,
            model=model,
            benchmark=benchmark,
            outputs=outputs,
        )
        evaluators.append(evaluator)

        # Update task_config.dataset_args with benchmark metadata, except for DataCollection
        if dataset_name != DataCollection.NAME:
            task_config.dataset_args[dataset_name] = benchmark.to_dict()

    # dump task_cfg to outputs.configs_dir after creating evaluators
    task_config.dump_yaml(outputs.configs_dir)
    logger.info(task_config)

    # Run evaluation for each evaluator
    for evaluator in evaluators:
        res_dict = evaluator.eval()
        eval_results[evaluator.benchmark.name] = res_dict

    # Make overall report
    try:
        report_table: str = gen_table(reports_path_list=[outputs.reports_dir], add_overall_metric=True)
        logger.info(f'Overall report table: \n{report_table} \n')
    except Exception:
        logger.error('Failed to generate report table.')
    # Clean up
    if model is not None:
        import gc

        del model
        del evaluators
        gc.collect()

        from evalscope.utils.import_utils import check_import
        if check_import('torch', raise_warning=False):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return eval_results


def main():
    from evalscope.arguments import parse_args
    args = parse_args()
    run_task(args)


if __name__ == '__main__':
    main()
