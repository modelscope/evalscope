import json
import os
import torch
from itertools import product

from evalscope.backend.rag_eval.clip_benchmark.arguments import Arguments
from evalscope.backend.rag_eval.clip_benchmark.dataset_builder import (build_dataset, get_dataloader,
                                                                       get_dataset_default_task)
from evalscope.backend.rag_eval.clip_benchmark.tasks import image_caption, zeroshot_classification, zeroshot_retrieval
from evalscope.backend.rag_eval.utils.clip import VisionModel
from evalscope.utils.logger import get_logger

logger = get_logger()


def evaluate(args: Arguments):
    models = args.models
    dataset_names = args.dataset_name
    data_dir = args.data_dir
    split = args.split
    batch_size = args.batch_size
    num_workers = args.num_workers
    verbose = args.verbose
    input_task = args.task
    output_dir = args.output_dir
    cache_dir = args.cache_dir
    skip_existing = args.skip_existing
    limit = args.limit

    # Iterate over model and dataset combinations
    for model_cfg, dataset_name in product(models, dataset_names):
        task = input_task or get_dataset_default_task(dataset_name)
        model_name = os.path.basename(model_cfg['model_name'])

        output_path = os.path.join(output_dir, model_name)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f'{dataset_name}_{task}.json')

        # Skip evaluation if the result already exists and skip_existing is True
        if os.path.exists(output_file) and skip_existing:
            if verbose:
                logger.info(f'Skip {output_dir}, exists already.')
            return

        # Determine device (CPU or GPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_cfg['device'] = device
        # Initialize the model
        model = VisionModel.load(**model_cfg)

        # Build the dataset
        dataset = build_dataset(
            dataset_name=dataset_name,
            root=data_dir,
            transform=model.transform,
            split=split,
            wds_cache_dir=f'{cache_dir}/{dataset_name}',
        )

        # Create the dataloader
        dataloader = get_dataloader(dataset_name, dataset, batch_size, num_workers)

        # Evaluate based on the task
        if task == 'zeroshot_classification':
            zeroshot_templates = (dataset.templates if hasattr(dataset, 'templates') else None)
            if verbose:
                logger.info(f'Zero-shot templates: {zeroshot_templates}')
            classnames = dataset.classes if hasattr(dataset, 'classes') else None
            assert (zeroshot_templates is not None
                    and classnames is not None), 'Dataset does not support classification'
            metrics = zeroshot_classification.evaluate(
                model,
                dataloader,
                classnames,
                zeroshot_templates,
                device=device,
                verbose=verbose,
                limit=limit,
            )
        elif task == 'zeroshot_retrieval':
            metrics = zeroshot_retrieval.evaluate(model, dataloader, recall_k_list=[5], device=device, limit=limit)
        elif task == 'image_caption':
            output_path = os.path.join(output_path, dataset_name, 'retrieval_data')
            metrics = image_caption.evaluate(model, dataloader, limit=limit, output_path=output_path)

        # Prepare dump data
        dump = {
            'dataset': dataset_name,
            'model': model_name,
            'task': task,
            'metrics': metrics,
        }

        if verbose:
            logger.info(f'Evaluation results: {dump}')

        # Write the results to output file
        if verbose:
            logger.info(f'Dump results to: {output_file}')
        with open(output_file, 'w') as f:
            json.dump(dump, f)


if __name__ == '__main__':
    evaluate()
