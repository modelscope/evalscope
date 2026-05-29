# Copyright (c) Alibaba, Inc. and its affiliates.
"""MTEB evaluation entry point.

Implements the MTEB 2.x evaluation flow with optional two-stage
(Encoder + CrossEncoder) reranking and ModelScope data loading.
"""
import mteb
import os
from pathlib import Path
from tabulate import tabulate
from typing import List

from evalscope.backend.rag_eval.models import load_model
from evalscope.backend.rag_eval.mteb.arguments import MTEBEvalConfig, MTEBModelConfig, MTEBToolConfig
from evalscope.backend.rag_eval.mteb.data_loader import patch_tasks_for_modelscope
from evalscope.utils.logger import get_logger

logger = get_logger()


def run_mteb_eval(config: MTEBToolConfig):
    """Main entry point for MTEB evaluation.

    Dispatch logic:
        - 1 encoder + 1 cross-encoder, with at least one Retrieval task
          → two-stage evaluation (encoder retrieval → cross-encoder rerank)
        - 1 encoder + 1 cross-encoder without Retrieval tasks
          → run cross-encoder directly (single-stage on Reranking tasks)
        - otherwise → run each model independently in single-stage mode
    """
    eval_args = config.eval
    models = config.models

    encoders = [m for m in models if not m.is_cross_encoder]
    rerankers = [m for m in models if m.is_cross_encoder]

    if len(encoders) == 1 and len(rerankers) == 1:
        tasks = resolve_tasks(eval_args)
        has_retrieval = any(getattr(t.metadata, 'type', None) == 'Retrieval' for t in tasks)
        if has_retrieval:
            return two_stage_eval(encoders[0], rerankers[0], eval_args)
        else:
            return one_stage_eval(rerankers[0], eval_args)

    results = None
    for model_config in models:
        results = one_stage_eval(model_config, eval_args)
    return results


def resolve_tasks(eval_args: MTEBEvalConfig) -> list:
    """Resolve MTEB tasks from the evaluation configuration.

    Resolution order:
        1. ``task_names`` (explicit list of task names)
        2. ``task_types`` and/or ``languages`` (filter-based)
        3. ``custom_tasks`` only (no standard tasks specified)

    If both standard tasks and ``custom_tasks`` are specified, custom tasks
    are appended to the resolved standard tasks.
    """
    if eval_args.task_names:
        tasks = list(mteb.get_tasks(tasks=eval_args.task_names))
    elif eval_args.task_types or eval_args.languages:
        kwargs = {}
        if eval_args.task_types:
            kwargs['task_types'] = eval_args.task_types
        if eval_args.languages:
            kwargs['languages'] = eval_args.languages
        tasks = list(mteb.get_tasks(**kwargs))
    elif eval_args.custom_tasks:
        tasks = []
    else:
        raise ValueError(
            "Must specify either 'task_names', 'task_types'/'languages', "
            "or 'custom_tasks' in eval config."
        )

    if eval_args.custom_tasks:
        from evalscope.backend.rag_eval.mteb.custom_task import build_custom_task
        for task_config in eval_args.custom_tasks:
            tasks.append(build_custom_task(task_config))

    return tasks


def _build_evaluate_kwargs(eval_args: MTEBEvalConfig, output_folder: str, prediction_folder=None) -> dict:
    """Build the kwargs dict passed to ``mteb.evaluate``."""
    from mteb import ResultCache

    eval_kwargs: dict = {
        'cache': ResultCache(cache_path=output_folder),
        'overwrite_strategy': 'always' if eval_args.overwrite_results else 'only-missing',
    }
    if eval_args.encode_kwargs:
        eval_kwargs['encode_kwargs'] = eval_args.encode_kwargs
    if prediction_folder is not None:
        eval_kwargs['prediction_folder'] = prediction_folder
    return eval_kwargs


def one_stage_eval(model_args: MTEBModelConfig, eval_args: MTEBEvalConfig):
    """Run single-model MTEB evaluation."""
    model = load_model(model_args)

    tasks = resolve_tasks(eval_args)

    if eval_args.hub == 'modelscope':
        patch_tasks_for_modelscope(tasks, limits=eval_args.limits)

    eval_kwargs = _build_evaluate_kwargs(eval_args, eval_args.output_folder)

    results = mteb.evaluate(
        model=model,
        tasks=tasks,
        **eval_kwargs,
    )

    show_results(eval_args.output_folder, model, results.task_results)
    return results


def two_stage_eval(
    encoder_args: MTEBModelConfig,
    reranker_args: MTEBModelConfig,
    eval_args: MTEBEvalConfig,
):
    """Run two-stage evaluation: encoder retrieval, then cross-encoder rerank.

    Stage 1: encoder runs over Retrieval tasks with ``prediction_folder`` set
        so that retrieval predictions are persisted.
    Stage 2: each task is converted via ``task.convert_to_reranking`` to a
        reranking task seeded with stage 1 predictions, then the cross-encoder
        is evaluated on the converted tasks.
    """
    encoder = load_model(encoder_args)
    reranker = load_model(reranker_args)

    tasks = resolve_tasks(eval_args)

    if eval_args.hub == 'modelscope':
        patch_tasks_for_modelscope(tasks, limits=eval_args.limits)

    stage1_path = os.path.join(eval_args.output_folder, 'stage1')
    stage2_path = os.path.join(eval_args.output_folder, 'stage2')
    stage1_predictions = Path(stage1_path) / 'predictions'
    stage2_predictions = Path(stage2_path) / 'predictions'

    logger.info('=== Stage 1: Encoder retrieval ===')
    eval_kwargs_s1 = _build_evaluate_kwargs(
        eval_args,
        stage1_path,
        prediction_folder=stage1_predictions,
    )
    mteb.evaluate(
        model=encoder,
        tasks=tasks,
        **eval_kwargs_s1,
    )

    logger.info('=== Stage 2: CrossEncoder reranking ===')
    for task in tasks:
        if getattr(task.metadata, 'type', None) == 'Retrieval':
            task.convert_to_reranking(
                top_ranked_path=str(stage1_predictions),
                top_k=eval_args.top_k,
            )

    eval_kwargs_s2 = _build_evaluate_kwargs(
        eval_args,
        stage2_path,
        prediction_folder=stage2_predictions,
    )
    results = mteb.evaluate(
        model=reranker,
        tasks=tasks,
        **eval_kwargs_s2,
    )

    show_results(stage2_path, reranker, results.task_results)
    return results


def show_results(output_folder: str, model, results: List) -> None:
    """Display evaluation results in a formatted table."""
    model_name = getattr(model, 'model_name_or_path', 'unknown')
    if hasattr(model, 'mteb_model_meta'):
        try:
            model_name = model.mteb_model_meta.model_name_as_path()
        except Exception:
            pass

    data = []
    for result in results:
        try:
            main_res = result.only_main_score()
        except Exception:
            main_res = result
        scores = getattr(main_res, 'scores', {}) or {}
        for split, sub_scores in scores.items():
            for sub_score in sub_scores:
                data.append({
                    'Model': str(model_name).replace('eval__', ''),
                    'Task Type': getattr(main_res, 'task_type', 'N/A'),
                    'Task': getattr(main_res, 'task_name', 'N/A'),
                    'Split': split,
                    'Subset': sub_score.get('hf_subset', 'default'),
                    'Main Score': sub_score.get('main_score', 'N/A'),
                })

    if data:
        logger.info(f'Evaluation results:\n{tabulate(data, headers="keys", tablefmt="grid")}')
    logger.info(f'Results saved in: {os.path.abspath(output_folder)}')
