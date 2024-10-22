(clip_benchmark)=

# CLIP Benchmark

This framework supports the [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), which aims to provide a unified framework and benchmark for evaluating and analyzing CLIP (Contrastive Language-Image Pretraining) and its variants. Currently, the framework supports 43 evaluation datasets, including zero-shot retrieval tasks with the evaluation metric of recall@k, and zero-shot classification tasks with the evaluation metric of acc@k.

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

## Environment Preparation

Install the required packages
```bash
pip install evalscope[rag] -U
```

## Configure Evaluation Parameters

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

### Parameter Description
- `eval_backend`: Default value is `RAGEval`, indicating the use of the RAGEval evaluation backend.
- `eval_config`: A dictionary containing the following fields:
    - `tool`: The evaluation tool, using `clip_benchmark`.
    - `eval`: A dictionary containing the following fields:
        - `models`: A list of model configurations, each with the following fields:
            - `model_name`: `str` The model name or path, e.g., `AI-ModelScope/chinese-clip-vit-large-patch14-336px`. Supports automatic downloading from the ModelScope repository.
        - `dataset_name`: `List[str]` A list of dataset names, e.g., `["muge", "flickr8k", "mnist"]`. See [Task List](#supported-datasets).
        - `split`: `str` The split of the dataset to use, default is `test`.
        - `batch_size`: `int` Batch size for data loading, default is `128`.
        - `num_workers`: `int` Number of worker threads for data loading, default is `1`.
        - `verbose`: `bool` Whether to enable detailed logging, default is `True`.
        - `skip_existing`: `bool` Whether to skip processing if output already exists, default is `False`.
        - `output_dir`: `str` Output directory, default is `outputs`.
        - `cache_dir`: `str` Dataset cache directory, default is `cache`.
        - `limit`: `Optional[int]` Limit the number of samples to process, default is `None`, e.g., `1000`.

## Run Evaluation Task

```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

# Run task
run_task(task_cfg=task_cfg) 
```

### Output Evaluation Results

```{code-block} json
:caption: outputs/chinese-clip-vit-large-patch14-336px/muge_zeroshot_retrieval.json

{"dataset": "muge", "model": "AI-ModelScope/chinese-clip-vit-large-patch14-336px", "task": "zeroshot_retrieval", "metrics": {"image_retrieval_recall@5": 0.8935546875, "text_retrieval_recall@5": 0.876953125}}
```

## Custom Evaluation Dataset

This framework supports custom evaluation datasets. You only need to prepare the dataset and configure it according to the following format.

### Image-Text Retrieval Dataset

**1. Prepare the Dataset**
Prepare an `image_queries.jsonl` (fixed file name) image-text retrieval dataset with the following format:

```{code-block} json
:caption: custom_eval/multimodal/text-image-retrieval/image_queries.jsonl

{"image_path": "custom_eval/multimodal/images/dog.jpg", "query": ["dog"]}
{"image_path": "custom_eval/multimodal/images/AMNH.jpg", "query": ["building"]}
{"image_path": "custom_eval/multimodal/images/tokyo.jpg", "query": ["city", "tokyo"]}
{"image_path": "custom_eval/multimodal/images/tesla.jpg", "query": ["car", "tesla"]}
{"image_path": "custom_eval/multimodal/images/running.jpg", "query": ["man", "running"]}
```

Where:
- `image_path`: Path to the image, supports local paths.
- `query`: Text descriptions for image-text retrieval, supports multiple descriptions, e.g., `["dog", "cat"]`.

**2. Configure Evaluation Parameters**

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

Where:
- `dataset_name`: Dataset name, must be specified as `custom`.
- `data_dir`: Dataset directory, containing the `image_queries.jsonl` file.

**3. Run the Evaluation Task**
```python
from evalscope.run import run_task

run_task(task_cfg=task_cfg)
```

The evaluation results output as follows:
```json
{"dataset": "custom", "model": "AI-ModelScope/chinese-clip-vit-large-patch14-336px", "task": "zeroshot_retrieval", "metrics": {"image_retrieval_recall@5": 1.0, "text_retrieval_recall@5": 1.0}}
```

### Converting Image-Text Retrieval to Text Retrieval
To facilitate the evaluation of different multimodal retrieval methods, this framework supports converting image-text retrieval problems into text retrieval problems using multimodal large models, and then performing text retrieval evaluation.

**1. Prepare the Dataset**

Supports [Image-Text Retrieval Dataset](#image-text-retrieval-dataset) and [Custom Image-Text Retrieval Dataset](#image-text-retrieval-dataset).

**2. Configure Evaluation Parameters**
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
                    "prompt": "Describe this image in English",
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

Parameter Description:
- A multimodal large model must be configured in the `models` list:
    - `model_name`: Name of the multimodal large model, e.g., `internvl2-8b`.
    - `api_base`: API address of the multimodal large model, e.g., `http://localhost:8008/v1`.
    - `api_key`: API key of the multimodal large model, e.g., `xxx`.
    - `prompt` is the prompt input for the multimodal large model, e.g., `"Describe this image in English"`.
- `task`: Evaluation task, must be specified as `image_caption`.

**3. Run the Conversion Task**
```python
from evalscope.run import run_task
run_task(task_cfg=task_cfg)
```

The output results are as follows:
```
2024-10-22 19:56:09,832 - evalscope - INFO - Write files to outputs/internvl2-8b/muge/retrieval_data
2024-10-22 19:56:10,543 - evalscope - INFO - Evaluation results: {'dataset': 'muge', 'model': 'internvl2-8b', 'task': 'image_caption', 'metrics': {'conversion_successful': True, 'save_path': 'outputs/internvl2-8b/muge/retrieval_data'}}
2024-10-22 19:56:10,544 - evalscope - INFO - Dump results to: outputs/internvl2-8b/muge_image_caption.json
```

The output file directory structure is as follows:
```
muge
├── retrieval_data
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels
│       └── test.tsv
└── muge_image_caption.json
```
The specific content of the files is as follows:

```{code-block} json
:caption: outputs/internvl2-8b/muge/retrieval_data/corpus.jsonl

{"_id":0,"text":"This is an advertisement image showcasing the products of the brand Aoyaqi. The image contains six cans of the brand's drink, with the brand name and graphics printed on the cans. The cans are arranged on a carton, which also has the brand name and graphics. The entire package is predominantly red and yellow, giving a striking and attractive impression."}
{"_id":1,"text":"These are fashionable glasses with a metal frame in rose gold color. The temples are black, and the inner side of the temple has a brand logo that looks like 'The Row'. The design of these glasses is quite modern, suitable for everyday wear."}
{"_id":2,"text":"This image shows a woman taking a selfie with her side profile using her phone. She has long brown hair and is wearing a pair of exquisite earrings that resemble the letter 'A'. The background is an indoor environment with light blue walls and light-colored cabinets."}
{"_id":3,"text":"This is an image of a black plastic bottle with a red label on it. The label has white and yellow text, including the product name, brand, and some graphics. The bottle cap is red and gray."}
{"_id":4,"text":"This is a picture of a living room with a single armchair. The backrest and seat cushion of the chair have a zebra print pattern in black and white, and the frame is black wood with curled armrests. The legs of the chair are black, with an elegant shape. The chair is placed on a carpeted floor, with part of a sofa and decorative painting visible in the background. The decor style of the room is warm and modern."}
{"_id":5,"text":"This is an image of a disposable paper cup. The cup is cylindrical, with a smooth wall and no obvious decorations or patterns. The rim of the cup slightly flares outwards for easy gripping. The cup is light gray or off-white and looks relatively thin. This type of paper cup is often used for drinks or cold foods and is suitable for one-time use."}
{"_id":6,"text":"This image shows four cartoon characters with colorful lights in the background. From left to right, the four characters are:\n\n1. A character in blue clothing with a purple headscarf and hair accessories.\n2. A character in blue-green clothing with blue hair accessories and wings.\n3. A character in pink clothing with red headwear and wings.\n4. A character in red and white clothing with red headwear.\n\nThe background has the words 'New Grimm's Fairy Tales' and 'NEW GREEN'."}
{"_id":7,"text":"This image shows a hand holding blue grapes. The person is wearing a green sweater, and the fingers are slender. The grapes are dark blue with a smooth surface, and each grape looks plump and juicy. There are some green leaves and dry twigs for decoration. The background is a wooden table, giving a natural and fresh feeling."}
{"_id":8,"text":"This is an image of a cute little mug, with a light green body and a round handle. The cup features a cute cartoon design, including a bunny wearing headphones and the words 'Love Learning'. There are two small ears and several stars next to it. The overall design of the mug is simple and cute, suitable for daily use."}
{"_id":9,"text":"This is an image showing a large number of thread-like objects in plastic packaging. These objects are stacked together and look like some kind of fiber or hemp rope, possibly for weaving or processing."}
```

```{code-block} json
:caption: outputs/internvl2-8b/muge/retrieval_data/queries.jsonl

{"_id":0,"text":"Tamarind juice drink whole box Yunnan"}
{"_id":1,"text":"Da Vinci glasses"}
{"_id":2,"text":"Rhinestone bow earrings"}
{"_id":3,"text":"Dengzhou yellow wine"}
{"_id":4,"text":"Zebra print armchair"}
{"_id":5,"text":"Pudding cup mold"}
{"_id":6,"text":"Pretty Cure figurine set"}
{"_id":7,"text":"Blueberry model"}
{"_id":8,"text":"Cute drinking cup"}
{"_id":9,"text":"Fried noodles"}
```

```{code-block}
:caption: outputs/internvl2-8b/muge/retrieval_data/qrels/test.tsv

query-id    corpus-id   score
0           0           1
1           1           1
2           2           1
3           3           1
4           4           1
5           5           1
6           6           1
7           7           1
8           8           1
9           9           1
```

**4. Perform Text Retrieval Task**

With the dataset ready, you can perform the text retrieval task following the CMTEB tutorial.
```{seealso}
Refer to [Custom Text Retrieval Evaluation](./mteb.md#custom-text-retrieval-evaluation)
```