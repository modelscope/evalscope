import mteb
import os
from mteb.task_selection import results_to_dataframe

from evalscope.backend.rag_eval import EmbeddingModel, cmteb
from evalscope.utils.logger import get_logger

logger = get_logger()


def show_results(output_folder, model, results):
    model_name = model.mteb_model_meta.model_name_as_path()
    revision = model.mteb_model_meta.revision

    results_df = results_to_dataframe({model_name: {revision: results}})

    save_path = os.path.join(
        output_folder,
        model_name,
        revision,
    )
    logger.info(f'Evaluation results:\n{results_df.to_markdown()}')
    logger.info(f'Evaluation results saved in {os.path.abspath(save_path)}')


def one_stage_eval(
    model_args,
    eval_args,
) -> None:
    # load model
    model = EmbeddingModel.load(**model_args)
    custom_dataset_path = eval_args.pop('dataset_path', None)
    # load task first to update instructions
    tasks = cmteb.TaskBase.get_tasks(task_names=eval_args['tasks'], dataset_path=custom_dataset_path)
    evaluation = mteb.MTEB(tasks=tasks)

    # run evaluation
    results = evaluation.run(model, **eval_args)

    # save and log results
    show_results(eval_args['output_folder'], model, results)


def two_stage_eval(
    model1_args,
    model2_args,
    eval_args,
) -> None:
    """a two-stage run with the second stage reading results saved from the first stage."""
    # load model
    dual_encoder = EmbeddingModel.load(**model1_args)
    cross_encoder = EmbeddingModel.load(**model2_args)

    first_stage_path = f"{eval_args['output_folder']}/stage1"
    second_stage_path = f"{eval_args['output_folder']}/stage2"

    tasks = cmteb.TaskBase.get_tasks(task_names=eval_args['tasks'])
    for task in tasks:
        evaluation = mteb.MTEB(tasks=[task])

        # stage 1: run dual encoder
        evaluation.run(
            dual_encoder,
            save_predictions=True,
            output_folder=first_stage_path,
            overwrite_results=True,
            hub=eval_args['hub'],
            limits=eval_args['limits'],
        )
        # stage 2: run cross encoder
        results = evaluation.run(
            cross_encoder,
            top_k=eval_args['top_k'],
            save_predictions=True,
            output_folder=second_stage_path,
            previous_results=f'{first_stage_path}/{task.metadata.name}_default_predictions.json',
            overwrite_results=True,
            hub=eval_args['hub'],
            limits=eval_args['limits'],
        )

        # save and log results
        show_results(second_stage_path, cross_encoder, results)
