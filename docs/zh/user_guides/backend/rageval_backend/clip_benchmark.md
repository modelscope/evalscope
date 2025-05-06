(clip_benchmark)=

# CLIP Benchmark
本框架支持[CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)，其旨在为评测和分析CLIP（Contrastive Language-Image Pretraining）及其变体提供一个统一的框架和基准，目前框架支持43个评测数据集，包括zero-shot retireval任务，评价指标为recall@k；zero-shot classification任务，评价指标为acc@k。

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


## 环境准备
安装依赖包
```bash
pip install evalscope[rag] -U
```

## 配置评测参数

```python
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
```

参数说明：
- `eval_backend`：默认值为 `RAGEval`，表示使用 RAGEval 评测后端。
- `eval_config`：字典，包含以下字段：
    - `tool`：评测工具，使用 `clip_benchmark`。
    - `eval`：字典，包含以下字段：
        - `models`：模型配置列表，包含以下字段：
            - `model_name`: `str` 模型名称或路径，例如 `AI-ModelScope/chinese-clip-vit-large-patch14-336px`，支持从modelscope仓库自动下载模型。
        - `dataset_name`: `List[str]` 数据集名称列表，例如 `["muge", "flickr8k", "mnist"]`，参见[任务列表](#支持的数据集)。
        - `split`: `str` 数据集的划分部分，默认为 `test`。
        - `batch_size`: `int` 数据加载的批量大小，默认为 `128`。
        - `num_workers`: `int` 数据加载的工作线程数，默认为 `1`。
        - `verbose`: `bool` 是否启用详细日志记录，默认为 `True`。
        - `skip_existing`: `bool` 如果输出已经存在，是否跳过处理，默认为 `False`。
        - `cache_dir`: `str` 数据集缓存目录，默认为 `cache`。
        - `limit`: `Optional[int]` 限制处理样本的数量，默认为 `None`，例如 `1000`。

## 运行评测任务

```python
from evalscope.run import run_task

run_task(task_cfg=task_cfg) 
```

输出评测结果如下：

```{code-block} json
:caption: outputs/chinese-clip-vit-large-patch14-336px/muge_zeroshot_retrieval.json

{"dataset": "muge", "model": "AI-ModelScope/chinese-clip-vit-large-patch14-336px", "task": "zeroshot_retrieval", "metrics": {"image_retrieval_recall@5": 0.8935546875, "text_retrieval_recall@5": 0.876953125}}
```

## 自定义评测数据集

```{seealso}
[自定义图文评测数据集](../../../advanced_guides/custom_dataset/clip.md)
```
