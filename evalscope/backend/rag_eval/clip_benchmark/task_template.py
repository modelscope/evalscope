import os
import torch
import json
from itertools import product
from torch.utils.data import DataLoader
from evalscope.backend.rag_eval.clip_benchmark.dataset_builder import (
    build_dataset,
    get_dataset_default_task,
)
from evalscope.backend.rag_eval.clip_benchmark.metrics import (
    zeroshot_classification,
    zeroshot_retrieval,
)
from evalscope.backend.rag_eval.clip_benchmark.arguments import Arguments
from evalscope.backend.rag_eval.utils.clip import CLIPModel
from evalscope.utils.logger import get_logger

logger = get_logger()


def evaluate(args: Arguments):
    model_name_or_paths = args.model_name_or_path
    dataset_names = args.dataset_name
    data_dir = args.data_dir
    split = args.split
    batch_size = args.batch_size
    num_workers = args.num_workers
    verbose = args.verbose
    output_dir = args.output_dir
    cache_dir = args.cache_dir
    skip_existing = args.skip_existing

    # Iterate over model and dataset combinations
    for model_name_or_path, dataset_name in product(model_name_or_paths, dataset_names):
        task = get_dataset_default_task(dataset_name)
        model_name = os.path.basename(model_name_or_path)

        output_path = os.path.join(output_dir, model_name)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{dataset_name}_{task}.json")

        # Skip evaluation if the result already exists and skip_existing is True
        if os.path.exists(output_file) and skip_existing:
            if verbose:
                logger.info(f"Skip {output_dir}, exists already.")
            return

        # Determine device (CPU or GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the model
        model = CLIPModel(model_name_or_path, device=device)

        # Build the dataset
        dataset = build_dataset(
            dataset_name=dataset_name,
            root=data_dir,
            transform=model.transform,
            split=split,
            wds_cache_dir=f"{cache_dir}/{dataset_name}",
        )

        # Create the dataloader
        dataloader = DataLoader(
            dataset.batched(batch_size),
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Evaluate based on the task
        if task == "zeroshot_classification":
            zeroshot_templates = (
                dataset.templates if hasattr(dataset, "templates") else None
            )
            if verbose:
                logger.info(f"Zero-shot templates: {zeroshot_templates}")
            classnames = dataset.classes if hasattr(dataset, "classes") else None
            assert (
                zeroshot_templates is not None and classnames is not None
            ), "Dataset does not support classification"
            metrics = zeroshot_classification.evaluate(
                model,
                dataloader,
                classnames,
                zeroshot_templates,
                device=device,
                verbose=verbose,
            )
        elif task == "zeroshot_retrieval":
            metrics = zeroshot_retrieval.evaluate(
                model, dataloader, recall_k_list=[5], device=device
            )

        # Prepare dump data
        dump = {
            "dataset": dataset_name,
            "model": model_name_or_path,
            "task": task,
            "metrics": metrics,
        }

        if verbose:
            logger.info(f"Evaluation results: {dump}")

        # Write the results to output file
        if verbose:
            logger.info(f"Dump results to: {output_file}")
        with open(output_file, "w") as f:
            json.dump(dump, f)


if __name__ == "__main__":
    evaluate()
