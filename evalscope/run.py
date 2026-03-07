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
from evalscope.utils.io_utils import OutputsStructure
from evalscope.utils.logger import configure_logging, get_logger
from evalscope.utils.model_utils import seed_everything

logger = get_logger()


def run_task(task_cfg: Union[str, dict, TaskConfig, List[TaskConfig], Namespace]) -> Union[dict, List[dict]]:
    """Run evaluation task(s) based on the provided configuration."""
    # If task_cfg is a list, run each task individually
    if isinstance(task_cfg, list):
        return [run_task(cfg) for cfg in task_cfg]

    task_cfg = parse_task_config(task_cfg)
    return run_single_task(task_cfg)


def run_single_task(task_cfg: TaskConfig) -> dict:
    """Run a single evaluation task."""
    if task_cfg.seed is not None:
        seed_everything(task_cfg.seed)
    outputs = setup_work_directory(task_cfg)
    configure_logging(task_cfg.debug, os.path.join(outputs.logs_dir, 'eval_log.log'))

    if task_cfg.eval_backend != EvalBackend.NATIVE:
        result = run_non_native_backend(task_cfg, outputs)
    else:
        logger.info('Running with native backend')
        result = evaluate_model(task_cfg, outputs)

        logger.info(f'Finished evaluation for {task_cfg.model_id} on {task_cfg.datasets}')
        logger.info(f'Output directory: {outputs.outputs_dir}')

    return result


def setup_work_directory(task_cfg: TaskConfig):
    """Set the working directory for the task."""
    # Get current time
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
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


def compute_eval_total_count(task_config: 'TaskConfig') -> Optional[int]:
    """Estimate the total number of evaluation samples for a task configuration.

    Reads per-subset ``sample_count`` values from the bundled ``_meta`` JSON
    files and applies ``limit`` (per-subset cap) and ``repeats`` (multiplier).
    Returns ``None`` if the count cannot be determined (e.g. missing meta file
    or unknown dataset).

    Calculation per dataset::

        effective = min(sample_count, limit)  # apply limit first
        contribution = effective * repeats    # then multiply by repeats

    """
    from evalscope.utils.resource_utils import load_benchmark_data
    total = 0

    for dataset_name in task_config.datasets:
        entry = load_benchmark_data(dataset_name).get(dataset_name, {})
        if not entry.get('statistics'):
            logger.debug(f'No meta file found for dataset "{dataset_name}", skipping total_count estimate.')
            return None

        subset_stats = entry.get('statistics', {}).get('subset_stats', [])
        if not subset_stats:
            logger.debug(f'No subset_stats in meta for "{dataset_name}", skipping total_count estimate.')
            return None

        subset_count_map = {s['name']: s['sample_count'] for s in subset_stats}

        # Determine which subsets are active (user override or full list)
        dataset_args = task_config.dataset_args.get(dataset_name, {})
        active_subsets = dataset_args.get('subset_list', None)
        if active_subsets is None:
            active_subsets = entry.get('meta', {}).get('subset_list', list(subset_count_map.keys()))

        limit = task_config.limit
        repeats = task_config.repeats

        for subset in active_subsets:
            sample_count = subset_count_map.get(subset)
            if sample_count is None:
                continue
            # Apply limit per subset first
            if limit is not None:
                if isinstance(limit, float):
                    effective = int(sample_count * limit)
                else:
                    effective = min(sample_count, int(limit))
            else:
                effective = sample_count
            # Then multiply by repeats
            total += effective * repeats

    return total if total > 0 else None


def evaluate_model(task_config: TaskConfig, outputs: OutputsStructure) -> dict:
    """Evaluate the model based on the provided task configuration."""
    from evalscope.api.evaluator import Evaluator
    from evalscope.api.model.lazy_model import LazyModel
    from evalscope.api.registry import get_benchmark
    from evalscope.evaluator import DefaultEvaluator
    from evalscope.report import gen_html_report_file, gen_table
    from evalscope.utils.tqdm_utils import TqdmLogging as tqdm
    from evalscope.utils.tqdm_utils import make_tracker

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

    tracker_ctx = make_tracker(
        task_config.enable_progress_tracker,
        work_dir=outputs.outputs_dir,
        pipeline='eval',
        total_count=compute_eval_total_count(task_config),
    )
    # Run evaluation for each evaluator (outermost progress stage)
    with tracker_ctx:
        with tqdm(
            evaluators,
            desc='Running[eval]',
            total=len(evaluators),
            unit='benchmark',
            logger=logger,
        ) as pbar:
            for evaluator in pbar:
                res_dict = evaluator.eval()
                eval_results[evaluator.benchmark_name] = res_dict

    # Make overall report
    try:
        report_table: str = gen_table(reports_path_list=[outputs.reports_dir], add_overall_metric=True)
        logger.info(f'Overall report table: \n{report_table} \n')
    except Exception:
        logger.error('Failed to generate report table.')

    # Generate interactive HTML report if enabled
    if task_config.generate_html_report:
        try:
            html_path = gen_html_report_file(outputs.reports_dir)
            logger.info(f'HTML report generated: {html_path}')
        except Exception as e:
            logger.error(f'Failed to generate HTML report: {e}')
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
