(clip_benchmark)=

# CLIP Benchmark
本框架支持[CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)，其旨在为评估和分析CLIP（Contrastive Language-Image Pretraining）及其变体提供一个统一的框架和基准，目前框架支持43个评估数据集，包括zero-shot retireval任务，评价指标为recall@k；zero-shot classification任务，评价指标为acc@k。

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
            "output_dir": "outputs",
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
        - `output_dir`: `str` 输出目录，默认为 `outputs`。
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
本框架支持自定义评测数据集，只需要按如下格式准备数据集并配置即可。

### 图文检索数据集

**1. 准备数据集**
准备如下格式的`image_queries.jsonl`(文件名称固定)图文检索数据集：

```{code-block} json
:caption: custom_eval/multimodal/text-image-retrieval/image_queries.jsonl

{"image_path": "custom_eval/multimodal/images/dog.jpg", "query": ["dog"]}
{"image_path": "custom_eval/multimodal/images/AMNH.jpg", "query": ["building"]}
{"image_path": "custom_eval/multimodal/images/tokyo.jpg", "query": ["city", "tokyo"]}
{"image_path": "custom_eval/multimodal/images/tesla.jpg", "query": ["car", "tesla"]}
{"image_path": "custom_eval/multimodal/images/running.jpg", "query": ["man", "running"]}
```
其中：
- `image_path`: 图像路径，支持本地路径。
- `query`: 图文检索的文本描述，支持多个描述，例如 `["dog", "cat"]`。

**2. 配置评测参数**

```python
task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "clip_benchmark",
        "eval": {
            "models": [
                {
                    "model_name": "AI-ModelScope/chinese-clip-vit-large-patch14-336px",
                }
            ],
            "dataset_name": ["custom"],
            "data_dir": "custom_eval/multimodal/text-image-retrieval",
            "split": "test",
            "batch_size": 128,
            "num_workers": 1,
            "verbose": True,
            "skip_existing": False,
            "limit": 1000,
        },
    },
}
```
其中：
- `dataset_name`: 数据集名称，必须指定为`custom`。
- `data_dir`: 数据集目录，包含 `image_queries.jsonl`文件。

**3. 运行评测任务**
```python
from evalscope.run import run_task

run_task(task_cfg=task_cfg)
```

输出评测结果如下：
```json
{"dataset": "custom", "model": "AI-ModelScope/chinese-clip-vit-large-patch14-336px", "task": "zeroshot_retrieval", "metrics": {"image_retrieval_recall@5": 1.0, "text_retrieval_recall@5": 1.0}}
```

### 图文检索转文本检索
为方便评估不同的多模态检索方法，本框架支持通过多模态大模型，将图文检索问题转化为文本检索问题，再进行文本检索评测。

**1. 准备数据集**

支持[图文检索数据集](#支持的数据集)和[自定义图文检索数据集](#图文检索数据集)。

**2. 配置评测参数**
```python
task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "clip_benchmark",
        "eval": {
            "models": [
                {
                    "model_name": "internvl2-8b",
                    "api_base": "http://localhost:8008/v1",
                    "api_key": "xxx",
                    "prompt": "用中文描述这张图片",
                }
            ],
            "dataset_name": ["muge"],
            "split": "test",
            "task": "image_caption",
            "batch_size": 2,
            "num_workers": 1,
            "verbose": True,
            "skip_existing": False,
            "limit": 10,
        },
    },
}
```
参数说明：
- `models`列表中需配置多模态大模型：
    - `model_name`: 多模态大模型名称，例如 `internvl2-8b`。
    - `api_base`: 多模态大模型的API地址，例如 `http://localhost:8008/v1`。
    - `api_key`: 多模态大模型的API密钥，例如 `xxx`。
    - `prompt`为多模态大模型输入的提示词，例如 `"用中文描述这张图片"`。
- `task`: 评测任务，必须指定为`image_caption`。

**3. 运行转换任务**
```python
from evalscope.run import run_task
run_task(task_cfg=task_cfg)
```

输出结果如下：
```
2024-10-22 19:56:09,832 - evalscope - INFO - Write files to outputs/internvl2-8b/muge/retrieval_data
2024-10-22 19:56:10,543 - evalscope - INFO - Evaluation results: {'dataset': 'muge', 'model': 'internvl2-8b', 'task': 'image_caption', 'metrics': {'convertion_successful': True, 'save_path': 'outputs/internvl2-8b/muge/retrieval_data'}}
2024-10-22 19:56:10,544 - evalscope - INFO - Dump results to: outputs/internvl2-8b/muge_image_caption.json
```

输出文件目录结构如下：
```
muge
├── retrieval_data
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels
│       └── test.tsv
└── muge_image_caption.json
```
文件具体内容如下：
```{code-block} json
:caption: outputs/internvl2-8b/muge/retrieval_data/corpus.jsonl

{"_id":0,"text":"这是一张展示澳亚奇品牌的产品广告图片，图片中包含了六罐澳亚奇品牌的饮料，饮料罐上印有品牌的名称和图案。饮料罐排列在纸箱上，纸箱上也有品牌名称和图案。整个包装以红色和黄色为主基调，给人以醒目和吸引人的感觉。"}
{"_id":1,"text":"这是一副时尚的眼镜，镜框是金属材质的，颜色为玫瑰金色，镜腿部分是黑色的。镜腿的内侧有品牌标志，看起来像是“The Row”。这款眼镜的设计比较现代，适合日常佩戴。"}
{"_id":2,"text":"这张图片展示了一位女性，她正在用手机拍摄自己的侧脸自拍。她有长长的棕色头发，并佩戴着一对精美的耳环。耳环的设计有点像是字母“A”。背景是室内环境，可以看到淡蓝色墙壁和浅色的柜子。"}
{"_id":3,"text":"这是一张黑色塑料瓶的图片，瓶身上贴有红色标签，标签上有白色和黄色的文字。标签上内容包括产品名称、品牌和一些图案。瓶口是红色和灰色的盖子。"}
{"_id":4,"text":"这是一张客厅的照片，里面有一把单人沙发椅。沙发的靠背和坐垫上有黑白相间的斑马纹图案，椅子的框架是黑色的木制结构，带有卷曲的扶手。沙发的腿部是黑色的，造型优雅。沙发布置在一个铺有地毯的地板上，背景中可以看到部分沙发和装饰画，整个房间的装饰风格显得温馨且现代。"}
{"_id":5,"text":"这是一张一次性纸杯的图片。纸杯呈圆柱形，杯壁较为光滑，没有明显的装饰或花纹。杯口部分略微向外扩展，便于抓握。杯子整体呈浅灰色或乳白色，质地看起来较为轻薄。这种纸杯常用于盛装饮料或冷食，适合一次性使用。"}
{"_id":6,"text":"这张图片展示的是四个卡通人物，背景有五彩斑斓的光芒。从左到右，这四个角色分别是：\n\n1. 一个穿着蓝色服装、戴着紫色头巾和发饰的角色。\n2. 一个穿着蓝绿色服装、戴着蓝色发饰和翅膀的角色。\n3. 一个穿着粉红色服装、带着红色头饰和翅膀的角色。\n4. 一个穿着红色和白色服装、戴着红色头饰的角色。\n\n背景中，有“新格林童话”和“NEW GREEN”的字样。"}
{"_id":7,"text":"这是一张展示手中握着蓝色葡萄的照片。手的主人穿着绿色的毛衣，手指修长。葡萄颜色深蓝，表面光滑，每颗葡萄看起来都十分饱满多汁。旁边有一些绿色叶子和干燥的枝条做装饰。背景是一张木质的桌子，整体画面给人一种自然清新的感觉。"}
{"_id":8,"text":"这张图片展示了一个可爱的小马克杯，杯身是浅绿色，配有圆弧形的手柄。杯子上绘有可爱的卡通图案，包括一只戴着耳机的小兔子，并配有“热爱学习”字样，旁边还有两只小耳朵和几颗星星。整个马克杯的设计简洁可爱，适合用作日常饮品盛器。"}
{"_id":9,"text":"这是一张展示塑料包装中大量线状物体的图片。这些线状物体堆叠在一起，看起来像是一些纤维或麻线，可能是用于编织或加工的。"}
```

```{code-block} json
:caption: outputs/internvl2-8b/muge/retrieval_data/queries.jsonl

{"_id":0,"text":"酸角汁饮料 整箱 云南"}
{"_id":1,"text":"达芬奇眼镜"}
{"_id":2,"text":"水钻蝴蝶结耳钉"}
{"_id":3,"text":"邓州黄酒"}
{"_id":4,"text":"斑马纹老虎椅"}
{"_id":5,"text":"布丁杯模具"}
{"_id":6,"text":"光之美少女盒蛋"}
{"_id":7,"text":"蓝莓模型"}
{"_id":8,"text":"少女心喝水杯"}
{"_id":9,"text":"炸面"}
```


```{code-block}
:caption: outputs/internvl2-8b/muge/retrieval_data/qrels/test.tsv

query-id	corpus-id	score
0	0	1
1	1	1
2	2	1
3	3	1
4	4	1
5	5	1
6	6	1
7	7	1
8	8	1
9	9	1
```

**4. 执行文本检索任务**

有了数据集之后，按照CMTEB的教程，可以执行文本检索任务。
```{seealso}
参考[自定义文本检索文本评估](./mteb.md#自定义文本检索评估)
```