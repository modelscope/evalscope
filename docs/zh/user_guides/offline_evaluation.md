# 离线环境评估

数据集默认托管在[ModelScope](https://modelscope.cn/datasets)上，加载需要联网。如果是无网络环境，可以使用本地数据集，流程如下：

## 1. 下载数据集到本地
假如当前本地工作路径为 `/path/to/workdir`，执行以下命令：
```shell
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
```
则解压后的数据集在：`/path/to/workdir/data` 目录下，该目录在后续步骤将会作为`--dataset-dir`参数的值传入

## 2. 使用本地数据集创建评估任务
```shell
python evalscope/run.py \
 --model ZhipuAI/chatglm3-6b \
 --datasets arc \
 --dataset-hub Local \
 --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' \
 --limit 10
```
### 参数说明
- `--dataset-hub`: 数据集来源，枚举值： `ModelScope` 或 `Local`, 默认为`ModelScope`
- `--dataset-dir`: 当`--dataset-hub`为`Local`时，该参数指本地数据集路径； 如果`--dataset-hub` 为`ModelScope`，则该参数的含义是数据集缓存路径。

## 3. 使用本地模型进行评估
模型文件托管在ModelScope Hub端，需要联网加载，当需要在离线环境创建评估任务时，可参考以下步骤：

### 3.1 准备模型本地文件

文件夹结构参考chatglm3-6b，链接：https://modelscope.cn/models/ZhipuAI/chatglm3-6b/files

例如，将模型文件夹整体下载到本地路径 `/path/to/ZhipuAI/chatglm3-6b`

### 3.2 执行离线评估任务
```shell
python evalscope/run.py \
 --model /path/to/ZhipuAI/chatglm3-6b \
 --datasets arc \
 --dataset-hub Local \
 --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' \
 --limit 10
```