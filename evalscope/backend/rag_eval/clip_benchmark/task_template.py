import os
import torch
import json
from torch.utils.data import DataLoader
from evalscope.backend.rag_eval.clip_benchmark.dataset_builder import (
    build_dataset,
    get_dataset_default_task,
)
from evalscope.backend.rag_eval.clip_benchmark.metrics import (
    zeroshot_classification,
    zeroshot_retrieval,
)
from evalscope.backend.rag_eval.utils.clip import CLIPModel
from evalscope.utils.logger import get_logger

logger = get_logger()


def evaluate():
    model_name = "AI-ModelScope/clip-vit-large-patch14-336"
    # dataset_name = "flickr8k"
    dataset_name = "mnist"
    batch_size = 128
    num_workers = 0
    verbose = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = "outputs"

    task = get_dataset_default_task(dataset_name)

    model = CLIPModel(model_name, device=device)

    dataset = build_dataset(
        dataset_name=dataset_name,
        root=f"https://modelscope.cn/datasets/clip-benchmark/wds_{dataset_name}/resolve/master",
        transform=model.transform,
        split="test",
        wds_cache_dir=f"cache/{dataset_name}",
    )

    dataloader = DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )

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

    dump = {
        "dataset": dataset_name,
        "model": model_name,
        "task": task,
        "metrics": metrics,
    }

    if verbose:
        logger.info(f"Dump results to: {output}")

    output_path = os.path.join(output, model_name)
    with open(f"{output_path}/{task}_result.json", "w") as f:
        json.dump(dump, f)


if __name__ == "__main__":
    evaluate()
