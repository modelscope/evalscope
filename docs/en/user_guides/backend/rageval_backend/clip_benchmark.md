(clip_benchmark)=

# CLIP Benchmark

This framework supports the [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), which aims to provide a unified framework and benchmark for evaluating and analyzing CLIP (Contrastive Language-Image Pretraining) and its variants. Currently, the framework supports 43 evaluation datasets, including zero-shot retrieval tasks (metric: recall@k) and zero-shot classification tasks (metric: acc@k).

## Environment Preparation

Install the required packages:

```bash
pip install evalscope[rag] -U
```

## Quick Start

The following example shows how to evaluate a CLIP model with minimal configuration:

```python
from evalscope.run import run_task

task_cfg = {
    "work_dir": "outputs",
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "clip_benchmark",
        "eval": {
            "models": [
                {
                    "model_name": "AI-ModelScope/chinese-clip-vit-large-patch14-336px",
                }
            ],
            "dataset_name": ["muge"],
            "split": "test",
        },
    },
}

run_task(task_cfg=task_cfg)
```

Output evaluation results:

```{code-block} json
:caption: outputs/chinese-clip-vit-large-patch14-336px/muge_zeroshot_retrieval.json

{"dataset": "muge", "model": "AI-ModelScope/chinese-clip-vit-large-patch14-336px", "task": "zeroshot_retrieval", "metrics": {"image_retrieval_recall@5": 0.8935546875, "text_retrieval_recall@5": 0.876953125}}
```

Key parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | `List[dict]` | Model configuration list. `model_name` is the model name or path, supports automatic download from ModelScope |
| `dataset_name` | `List[str]` | Dataset name list, see [Supported Datasets](#supported-datasets) |
| `split` | `str` | Dataset split, default `"test"` |

## Advanced: Multi-Model / Multi-Dataset Batch Evaluation

When evaluating multiple models or datasets simultaneously, extend the configuration:

```python
from evalscope.run import run_task

task_cfg = {
    "work_dir": "outputs",
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "clip_benchmark",
        "eval": {
            "models": [
                {
                    "model_name": "AI-ModelScope/chinese-clip-vit-large-patch14-336px",
                }
            ],
            "dataset_name": ["muge", "flickr8k"],
            "split": "test",
            "batch_size": 128,
            "num_workers": 1,
            "verbose": True,
            "skip_existing": False,
            "cache_dir": "cache",
            "limit": 1000,
        },
    },
}

run_task(task_cfg=task_cfg)
```

Additional parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `128` | Data loading batch size |
| `num_workers` | `int` | `1` | Number of data loading workers |
| `verbose` | `bool` | `True` | Enable verbose logging |
| `skip_existing` | `bool` | `False` | Skip processing if output already exists |
| `cache_dir` | `str` | `"cache"` | Dataset cache directory |
| `limit` | `Optional[int]` | `None` | Limit the number of samples to process |

## Supported Datasets

| Dataset Name                                                                                                   | Task Type              | Notes                      |
|---------------------------------------------------------------------------------------------------------------|------------------------|----------------------------|
| [muge](https://modelscope.cn/datasets/clip-benchmark/muge/)                                                   | zeroshot_retrieval     | Chinese Multimodal Dataset |
| [flickr30k](https://modelscope.cn/datasets/clip-benchmark/flickr30k/)                                         | zeroshot_retrieval     |                            |
| [flickr8k](https://modelscope.cn/datasets/clip-benchmark/flickr8k/)                                           | zeroshot_retrieval     |                            |
| [mscoco_captions](https://modelscope.cn/datasets/clip-benchmark/mscoco_captions/)                             | zeroshot_retrieval     |                            |
| [mscoco_captions2017](https://modelscope.cn/datasets/clip-benchmark/mscoco_captions2017/)                     | zeroshot_retrieval     |                            |
| [imagenet1k](https://modelscope.cn/datasets/clip-benchmark/imagenet1k/)                                       | zeroshot_classification|                            |
| [imagenetv2](https://modelscope.cn/datasets/clip-benchmark/imagenetv2/)                                       | zeroshot_classification|                            |
| [imagenet_sketch](https://modelscope.cn/datasets/clip-benchmark/imagenet_sketch/)                             | zeroshot_classification|                            |
| [imagenet-a](https://modelscope.cn/datasets/clip-benchmark/imagenet-a/)                                       | zeroshot_classification|                            |
| [imagenet-r](https://modelscope.cn/datasets/clip-benchmark/imagenet-r/)                                       | zeroshot_classification|                            |
| [imagenet-o](https://modelscope.cn/datasets/clip-benchmark/imagenet-o/)                                       | zeroshot_classification|                            |
| [objectnet](https://modelscope.cn/datasets/clip-benchmark/objectnet/)                                         | zeroshot_classification|                            |
| [fer2013](https://modelscope.cn/datasets/clip-benchmark/fer2013/)                                             | zeroshot_classification|                            |
| [voc2007](https://modelscope.cn/datasets/clip-benchmark/voc2007/)                                             | zeroshot_classification|                            |
| [voc2007_multilabel](https://modelscope.cn/datasets/clip-benchmark/voc2007_multilabel/)                       | zeroshot_classification|                            |
| [sun397](https://modelscope.cn/datasets/clip-benchmark/sun397/)                                               | zeroshot_classification|                            |
| [cars](https://modelscope.cn/datasets/clip-benchmark/cars/)                                                   | zeroshot_classification|                            |
| [fgvc_aircraft](https://modelscope.cn/datasets/clip-benchmark/fgvc_aircraft/)                                 | zeroshot_classification|                            |
| [mnist](https://modelscope.cn/datasets/clip-benchmark/mnist/)                                                 | zeroshot_classification|                            |
| [stl10](https://modelscope.cn/datasets/clip-benchmark/stl10/)                                                 | zeroshot_classification|                            |
| [gtsrb](https://modelscope.cn/datasets/clip-benchmark/gtsrb/)                                                 | zeroshot_classification|                            |
| [country211](https://modelscope.cn/datasets/clip-benchmark/country211/)                                       | zeroshot_classification|                            |
| [renderedsst2](https://modelscope.cn/datasets/clip-benchmark/renderedsst2/)                                   | zeroshot_classification|                            |
| [vtab_caltech101](https://modelscope.cn/datasets/clip-benchmark/vtab_caltech101/)                             | zeroshot_classification|                            |
| [vtab_cifar10](https://modelscope.cn/datasets/clip-benchmark/vtab_cifar10/)                                   | zeroshot_classification|                            |
| [vtab_cifar100](https://modelscope.cn/datasets/clip-benchmark/vtab_cifar100/)                                 | zeroshot_classification|                            |
| [vtab_clevr_count_all](https://modelscope.cn/datasets/clip-benchmark/vtab_clevr_count_all/)                   | zeroshot_classification|                            |
| [vtab_clevr_closest_object_distance](https://modelscope.cn/datasets/clip-benchmark/vtab_clevr_closest_object_distance/) | zeroshot_classification|                            |
| [vtab_diabetic_retinopathy](https://modelscope.cn/datasets/clip-benchmark/vtab_diabetic_retinopathy/)         | zeroshot_classification|                            |
| [vtab_dmlab](https://modelscope.cn/datasets/clip-benchmark/vtab_dmlab/)                                       | zeroshot_classification|                            |
| [vtab_dsprites_label_orientation](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_orientation/) | zeroshot_classification|                            |
| [vtab_dsprites_label_x_position](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_x_position/) | zeroshot_classification|                            |
| [vtab_dsprites_label_y_position](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_y_position/) | zeroshot_classification|                            |
| [vtab_dtd](https://modelscope.cn/datasets/clip-benchmark/vtab_dtd/)                                           | zeroshot_classification|                            |
| [vtab_eurosat](https://modelscope.cn/datasets/clip-benchmark/vtab_eurosat/)                                   | zeroshot_classification|                            |
| [vtab_kitti_closest_vehicle_distance](https://modelscope.cn/datasets/clip-benchmark/vtab_kitti_closest_vehicle_distance/) | zeroshot_classification|                            |
| [vtab_flowers](https://modelscope.cn/datasets/clip-benchmark/vtab_flowers/)                                   | zeroshot_classification|                            |
| [vtab_pets](https://modelscope.cn/datasets/clip-benchmark/vtab_pets/)                                         | zeroshot_classification|                            |
| [vtab_pcam](https://modelscope.cn/datasets/clip-benchmark/vtab_pcam/)                                         | zeroshot_classification|                            |
| [vtab_resisc45](https://modelscope.cn/datasets/clip-benchmark/vtab_resisc45/)                                 | zeroshot_classification|                            |
| [vtab_smallnorb_label_azimuth](https://modelscope.cn/datasets/clip-benchmark/vtab_smallnorb_label_azimuth/)   | zeroshot_classification|                            |
| [vtab_smallnorb_label_elevation](https://modelscope.cn/datasets/clip-benchmark/vtab_smallnorb_label_elevation/) | zeroshot_classification|                            |
| [vtab_svhn](https://modelscope.cn/datasets/clip-benchmark/vtab_svhn/)                                         | zeroshot_classification|                            |

## Full Parameter Reference

- `eval_backend`: Default value is `RAGEval`, indicating the use of the RAGEval evaluation backend.
- `eval_config`: A dictionary containing the following fields:
    - `tool`: Evaluation tool, using `clip_benchmark`.
    - `eval`: Evaluation configuration, containing the following fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `List[dict]` | `[]` | Model configuration list. `model_name` is the model name or path, supports automatic download from ModelScope |
| `dataset_name` | `List[str]` | `[]` | Dataset name list, see [Supported Datasets](#supported-datasets) |
| `split` | `str` | `"test"` | Dataset split |
| `task` | `Optional[str]` | `None` | Task type, auto-inferred by default |
| `batch_size` | `int` | `128` | Data loading batch size |
| `num_workers` | `int` | `1` | Number of data loading workers |
| `verbose` | `bool` | `True` | Enable verbose logging |
| `output_dir` | `str` | `"outputs"` | Output directory for evaluation results |
| `cache_dir` | `str` | `"cache"` | Dataset cache directory |
| `skip_existing` | `bool` | `False` | Skip processing if output already exists |
| `data_dir` | `Optional[str]` | `None` | Custom data directory |
| `limit` | `Optional[int]` | `None` | Limit the number of samples to process |

## FAQ

### Dataset Download Failure

If ModelScope dataset download fails, try configuring a mirror or manually downloading the dataset, then specify the local path via the `data_dir` parameter.

### Slow Evaluation

- Increase `batch_size` (default 128) to improve throughput, mind the GPU memory limit
- Increase `num_workers` (default 1) to speed up data loading
- Use `skip_existing: true` to skip already completed evaluations

### Metric Definitions

- **zeroshot_classification**: reports `acc1` (Top-1 accuracy) and `acc5` (Top-5 accuracy)
- **zeroshot_retrieval**: reports `text_retrieval_recall@k` and `image_retrieval_recall@k`

## Custom Evaluation Dataset

```{seealso}
[Custom Image-Text Dataset](../../../advanced_guides/custom_dataset/clip.md)
```
