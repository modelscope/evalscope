# Offline Evaluation

By default, datasets are hosted on [ModelScope](https://modelscope.cn/datasets), which requires an internet connection to load. However, if you find yourself in an environment without internet access, you can use local datasets. Follow the steps below:

## 1. Download the Dataset Locally
Assuming your current local working path is `/path/to/workdir`, execute the following commands:
```shell
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
```
This will unzip the dataset into the `/path/to/workdir/data` directory, which will be used as the value for the `--dataset-dir` parameter in subsequent steps.

## 2. Create an Evaluation Task Using the Local Dataset
```shell
python evalscope/run.py \
 --model ZhipuAI/chatglm3-6b \
 --template-type chatglm3 \
 --datasets arc \
 --dataset-hub Local \
 --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' \
 --limit 10
```

### Parameter Descriptions
- `--dataset-hub`: Source of the dataset, with possible values: `ModelScope` or `Local`. The default is `ModelScope`.
- `--dataset-dir`: When `--dataset-hub` is set to `Local`, this parameter refers to the local dataset path. If `--dataset-hub` is set to `ModelScope`, this parameter refers to the dataset cache path.

## 3. Evaluate Using a Local Model
Model files are hosted on the ModelScope Hub, requiring internet access for loading. To create an evaluation task in an offline environment, refer to the steps below:

### 3.1 Prepare Local Model Files
Structure your model files similar to the `chatglm3-6b` directory, link: https://modelscope.cn/models/ZhipuAI/chatglm3-6b/files. For example, you can download the entire model folder to the local path `/path/to/ZhipuAI/chatglm3-6b`.

### 3.2 Execute the Offline Evaluation Task
```shell
python evalscope/run.py \
 --model /path/to/ZhipuAI/chatglm3-6b \
 --template-type chatglm3 \
 --datasets arc \
 --dataset-hub Local \
 --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' \
 --limit 10
```