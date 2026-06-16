(clip_benchmark)=

# CLIP Benchmark

本框架支持 [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)，其旨在为评测和分析 CLIP（Contrastive Language-Image Pretraining）及其变体提供一个统一的框架和基准，目前框架支持 43 个评测数据集，包括 zero-shot retrieval 任务（评价指标为 recall@k）和 zero-shot classification 任务（评价指标为 acc@k）。

## 环境准备

安装依赖包：

```bash
pip install evalscope[rag] -U
```

## 快速开始

以下示例展示如何用最小配置评测一个 CLIP 模型：

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

输出评测结果如下：

```{code-block} json
:caption: outputs/chinese-clip-vit-large-patch14-336px/muge_zeroshot_retrieval.json

{"dataset": "muge", "model": "AI-ModelScope/chinese-clip-vit-large-patch14-336px", "task": "zeroshot_retrieval", "metrics": {"image_retrieval_recall@5": 0.8935546875, "text_retrieval_recall@5": 0.876953125}}
```

关键参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| `models` | `List[dict]` | 模型配置列表，`model_name` 为模型名称或路径，支持从 ModelScope 自动下载 |
| `dataset_name` | `List[str]` | 数据集名称列表，参见[支持的数据集](#支持的数据集) |
| `split` | `str` | 数据集的划分部分，默认 `"test"` |

## 进阶：多模型/多数据集批量评测

当需要同时评测多个模型或多个数据集时，可以扩展配置：

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

额外参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | `int` | `128` | 数据加载的批量大小 |
| `num_workers` | `int` | `1` | 数据加载的工作线程数 |
| `verbose` | `bool` | `True` | 是否启用详细日志记录 |
| `skip_existing` | `bool` | `False` | 如果输出已存在，是否跳过处理 |
| `cache_dir` | `str` | `"cache"` | 数据集缓存目录 |
| `limit` | `Optional[int]` | `None` | 限制处理样本的数量 |

## 支持的数据集

| 数据集名称                                                                                                     | 任务类型               | 备注 |
|--------------------------------------------------------------------------------------------------------------|--------------------|------|
| [muge](https://modelscope.cn/datasets/clip-benchmark/muge/)                                                   | zeroshot_retrieval |  中文多模态图文数据集    |
| [flickr30k](https://modelscope.cn/datasets/clip-benchmark/flickr30k/)                                         | zeroshot_retrieval |      |
| [flickr8k](https://modelscope.cn/datasets/clip-benchmark/flickr8k/)                                           | zeroshot_retrieval |      |
| [mscoco_captions](https://modelscope.cn/datasets/clip-benchmark/mscoco_captions/)                             | zeroshot_retrieval |      |
| [mscoco_captions2017](https://modelscope.cn/datasets/clip-benchmark/mscoco_captions2017/)                     | zeroshot_retrieval |      |
| [imagenet1k](https://modelscope.cn/datasets/clip-benchmark/imagenet1k/)                                       | zeroshot_classification |      |
| [imagenetv2](https://modelscope.cn/datasets/clip-benchmark/imagenetv2/)                                       | zeroshot_classification |      |
| [imagenet_sketch](https://modelscope.cn/datasets/clip-benchmark/imagenet_sketch/)                             | zeroshot_classification |      |
| [imagenet-a](https://modelscope.cn/datasets/clip-benchmark/imagenet-a/)                                       | zeroshot_classification |      |
| [imagenet-r](https://modelscope.cn/datasets/clip-benchmark/imagenet-r/)                                       | zeroshot_classification |      |
| [imagenet-o](https://modelscope.cn/datasets/clip-benchmark/imagenet-o/)                                       | zeroshot_classification |      |
| [objectnet](https://modelscope.cn/datasets/clip-benchmark/objectnet/)                                         | zeroshot_classification |      |
| [fer2013](https://modelscope.cn/datasets/clip-benchmark/fer2013/)                                             | zeroshot_classification |      |
| [voc2007](https://modelscope.cn/datasets/clip-benchmark/voc2007/)                                             | zeroshot_classification |      |
| [voc2007_multilabel](https://modelscope.cn/datasets/clip-benchmark/voc2007_multilabel/)                       | zeroshot_classification |      |
| [sun397](https://modelscope.cn/datasets/clip-benchmark/sun397/)                                               | zeroshot_classification |      |
| [cars](https://modelscope.cn/datasets/clip-benchmark/cars/)                                                   | zeroshot_classification |      |
| [fgvc_aircraft](https://modelscope.cn/datasets/clip-benchmark/fgvc_aircraft/)                                 | zeroshot_classification |      |
| [mnist](https://modelscope.cn/datasets/clip-benchmark/mnist/)                                                 | zeroshot_classification |      |
| [stl10](https://modelscope.cn/datasets/clip-benchmark/stl10/)                                                 | zeroshot_classification |      |
| [gtsrb](https://modelscope.cn/datasets/clip-benchmark/gtsrb/)                                                 | zeroshot_classification |      |
| [country211](https://modelscope.cn/datasets/clip-benchmark/country211/)                                       | zeroshot_classification |      |
| [renderedsst2](https://modelscope.cn/datasets/clip-benchmark/renderedsst2/)                                   | zeroshot_classification |      |
| [vtab_caltech101](https://modelscope.cn/datasets/clip-benchmark/vtab_caltech101/)                             | zeroshot_classification |      |
| [vtab_cifar10](https://modelscope.cn/datasets/clip-benchmark/vtab_cifar10/)                                   | zeroshot_classification |      |
| [vtab_cifar100](https://modelscope.cn/datasets/clip-benchmark/vtab_cifar100/)                                 | zeroshot_classification |      |
| [vtab_clevr_count_all](https://modelscope.cn/datasets/clip-benchmark/vtab_clevr_count_all/)                   | zeroshot_classification |      |
| [vtab_clevr_closest_object_distance](https://modelscope.cn/datasets/clip-benchmark/vtab_clevr_closest_object_distance/) | zeroshot_classification |      |
| [vtab_diabetic_retinopathy](https://modelscope.cn/datasets/clip-benchmark/vtab_diabetic_retinopathy/)         | zeroshot_classification |      |
| [vtab_dmlab](https://modelscope.cn/datasets/clip-benchmark/vtab_dmlab/)                                       | zeroshot_classification |      |
| [vtab_dsprites_label_orientation](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_orientation/) | zeroshot_classification |      |
| [vtab_dsprites_label_x_position](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_x_position/) | zeroshot_classification |      |
| [vtab_dsprites_label_y_position](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_y_position/) | zeroshot_classification |      |
| [vtab_dtd](https://modelscope.cn/datasets/clip-benchmark/vtab_dtd/)                                           | zeroshot_classification |      |
| [vtab_eurosat](https://modelscope.cn/datasets/clip-benchmark/vtab_eurosat/)                                   | zeroshot_classification |      |
| [vtab_kitti_closest_vehicle_distance](https://modelscope.cn/datasets/clip-benchmark/vtab_kitti_closest_vehicle_distance/) | zeroshot_classification |      |
| [vtab_flowers](https://modelscope.cn/datasets/clip-benchmark/vtab_flowers/)                                   | zeroshot_classification |      |
| [vtab_pets](https://modelscope.cn/datasets/clip-benchmark/vtab_pets/)                                         | zeroshot_classification |      |
| [vtab_pcam](https://modelscope.cn/datasets/clip-benchmark/vtab_pcam/)                                         | zeroshot_classification |      |
| [vtab_resisc45](https://modelscope.cn/datasets/clip-benchmark/vtab_resisc45/)                                 | zeroshot_classification |      |
| [vtab_smallnorb_label_azimuth](https://modelscope.cn/datasets/clip-benchmark/vtab_smallnorb_label_azimuth/)   | zeroshot_classification |      |
| [vtab_smallnorb_label_elevation](https://modelscope.cn/datasets/clip-benchmark/vtab_smallnorb_label_elevation/) | zeroshot_classification |      |
| [vtab_svhn](https://modelscope.cn/datasets/clip-benchmark/vtab_svhn/)                                         | zeroshot_classification |      |

## 完整参数参考

- `eval_backend`：默认值为 `RAGEval`，表示使用 RAGEval 评测后端。
- `eval_config`：字典，包含以下字段：
    - `tool`：评测工具，使用 `clip_benchmark`。
    - `eval`：评测配置，包含以下字段：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `models` | `List[dict]` | `[]` | 模型配置列表，`model_name` 为模型名称或路径，支持从 ModelScope 自动下载 |
| `dataset_name` | `List[str]` | `[]` | 数据集名称列表，参见[支持的数据集](#支持的数据集) |
| `split` | `str` | `"test"` | 数据集的划分部分 |
| `task` | `Optional[str]` | `None` | 任务类型，默认自动推断 |
| `batch_size` | `int` | `128` | 数据加载的批量大小 |
| `num_workers` | `int` | `1` | 数据加载的工作线程数 |
| `verbose` | `bool` | `True` | 是否启用详细日志记录 |
| `output_dir` | `str` | `"outputs"` | 评测结果输出目录 |
| `cache_dir` | `str` | `"cache"` | 数据集缓存目录 |
| `skip_existing` | `bool` | `False` | 如果输出已存在，是否跳过处理 |
| `data_dir` | `Optional[str]` | `None` | 自定义数据目录 |
| `limit` | `Optional[int]` | `None` | 限制处理样本的数量 |

## 常见问题

### 数据集下载失败

如果 ModelScope 数据集下载失败，可尝试设置镜像或手动下载数据集，然后通过 `data_dir` 参数指定本地路径。

### 评测速度慢

- 增大 `batch_size`（默认 128）提升吞吐量，注意 GPU 显存限制
- 增大 `num_workers`（默认 1）加速数据加载
- 使用 `skip_existing: true` 跳过已完成的评测

### 结果指标含义

- **zeroshot_classification**: 报告 `acc1`（Top-1 准确率）和 `acc5`（Top-5 准确率）
- **zeroshot_retrieval**: 报告 `text_retrieval_recall@k` 和 `image_retrieval_recall@k`

## 自定义评测数据集

```{seealso}
[自定义图文评测数据集](../../../advanced_guides/custom_dataset/clip.md)
```
