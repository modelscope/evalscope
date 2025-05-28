# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Run evaluation for LLMs.
"""
import os
from argparse import Namespace
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

from evalscope.config import TaskConfig, parse_task_config
from evalscope.constants import DataCollection, EvalBackend
from evalscope.utils import seed_everything
from evalscope.utils.io_utils import OutputsStructure
from evalscope.utils.logger import configure_logging, get_logger

if TYPE_CHECKING:
    from evalscope.models import LocalModel

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
    # elif are_paths_same(task_cfg.work_dir, DEFAULT_WORK_DIR):
    else:
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
    from evalscope.models import get_local_model
    from evalscope.report import gen_table

    # Initialize evaluator
    eval_results = {}
    base_model = get_local_model(task_cfg)
    evaluators = []
    for dataset_name in task_cfg.datasets:
        evaluator = create_evaluator(task_cfg, dataset_name, outputs, base_model)
        evaluators.append(evaluator)

    # dump task_cfg to outputs.configs_dir after creating evaluators
    task_cfg.dump_yaml(outputs.configs_dir)
    logger.info(task_cfg)

    # Run evaluation for each evaluator
    for evaluator in evaluators:
        res_dict = evaluator.eval()
        eval_results[evaluator.dataset_name] = res_dict

    # Make overall report
    try:
        report_table: str = gen_table([outputs.reports_dir])
        logger.info(f'Overall report table: \n{report_table} \n')
    except Exception:
        logger.error('Failed to generate report table.')

    # Clean up
    if base_model is not None:
        import gc
        import torch

        del base_model
        del evaluators
        torch.cuda.empty_cache()
        gc.collect()

    return eval_results


def create_evaluator(task_cfg: TaskConfig, dataset_name: str, outputs: OutputsStructure, base_model: 'LocalModel'):
    """Create an evaluator object for the specified dataset."""
    from evalscope.benchmarks import Benchmark, BenchmarkMeta
    from evalscope.evaluator import Evaluator
    from evalscope.models import initialize_model_adapter

    benchmark: BenchmarkMeta = Benchmark.get(dataset_name)

    if dataset_name == DataCollection.NAME:
        # EvaluatorCollection is a collection of evaluators
        from evalscope.collections import EvaluatorCollection
        data_adapter = benchmark.get_data_adapter(config=task_cfg.dataset_args.get(dataset_name, {}))
        return EvaluatorCollection(task_cfg, data_adapter, outputs, base_model)

    # Initialize data adapter first to update config
    data_adapter = benchmark.get_data_adapter(config=task_cfg.dataset_args.get(dataset_name, {}))
    # Initialize model adapter
    model_adapter = initialize_model_adapter(task_cfg, data_adapter, base_model)

    # update task_cfg.dataset_args
    task_cfg.dataset_args[dataset_name] = benchmark.to_string_dict()

    return Evaluator(
        data_adapter=data_adapter,
        model_adapter=model_adapter,
        outputs=outputs,
        task_cfg=task_cfg,
    )


def main():
    from evalscope.arguments import parse_args
    args = parse_args()
    run_task(args)


if __name__ == '__main__':
    main()
