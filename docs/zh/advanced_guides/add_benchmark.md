# 👍 贡献基准评测

EvalScope作为[ModelScope](https://modelscope.cn)的官方评测工具，其基准评测功能正在持续优化中！我们诚邀您参考本教程，轻松添加自己的评测基准，并与广大社区成员分享您的贡献。一起助力EvalScope的成长，让我们的工具更加出色！

下面将介绍如何添加**通用文本推理**和**多项选择**两种基准评测，主要包含上传数据集、注册数据集、编写评测任务三个步骤。

## 基础概念

EvalScope评测流程主要包含以下步骤：
1. 数据准备：通过`DataAdapter`加载和预处理数据集。
2. 任务定义：通过`TaskConfig`定义评测任务的配置，包括模型、数据集、评估指标等。
3. 评测执行：通过`run_task`函数执行评测任务，并输出评测结果。

其中`DataAdapter`是我们需要重点了解的类，包含如下步骤：
（介绍每个步骤和对应数据结构）

## 1. 准备基准评测数据集

您有两种方式准备基准评测数据集：

1. **上传到ModelScope（推荐）**：将数据集上传到ModelScope平台，这样其他用户可以一键加载您的数据集，使用更加便捷，也能让更多用户受益于您的贡献。如需上传到ModelScope，可参考[数据集上传教程](https://www.modelscope.cn/docs/datasets/create)。

2. **本地使用**：您也可以直接使用本地数据集进行评测，适合数据集尚在开发阶段或含有敏感信息的情况。

无论选择哪种方式，请确保数据的格式正确且可被加载。如使用ModelScope，可通过以下代码测试：

```python
from modelscope import MsDataset

dataset = MsDataset.load("/path/to/your/dataset")  # 替换为你的数据集
```

## 2. 创建文件结构

首先[Fork EvalScope](https://github.com/modelscope/evalscope/fork) 仓库，即创建一个自己的EvalScope仓库副本，将其clone到本地。

```bash
git clone https://github.com/your_username/evalscope.git
cd evalscope
```

然后，在`evalscope/benchmarks/`目录下添加基准评测，结构如下：

```text
evalscope/benchmarks/
├── benchmark_name
│   ├── __init__.py
│   ├── benchmark_name_adapter.py
│   └── ...
```
具体到`MMLU-Pro`，结构如下：

```text
evalscope/benchmarks/
├── mmlu_pro
│   ├── __init__.py
│   ├── mmlu_pro_adapter.py
│   └── ...
```

## 3. 编写评测逻辑

下面将以**gsm8k**和**MMLU-Pro**为例，分别进行介绍**通用文本推理**和**多项选择**两种评测任务。

### 通用文本推理
我们需要在`benchmark_name_adapter.py`中注册`Benchmark`，使得EvalScope能够加载我们添加的基准测试。以`MMLU-Pro`为例，主要包含以下内容：

- 导入`Benchmark`和`DataAdapter`
- 注册`Benchmark`，指定：
    - `name`：基准测试名称
    - `pretty_name`：基准测试的可读名称
    - `tags`：基准测试标签，用于分类和搜索
    - `description`：基准测试描述，可以使用Markdown格式，建议使用英文
    - `dataset_id`：基准测试数据集ID，用于加载基准测试数据集
    - `subset_list`：基准测试数据集的子数据集
    - `metric_list`：基准测试评估指标
    - `few_shot_num`：评测的In Context Learning样本数量
    - `train_split`：基准测试训练集，用于采样ICL样例
    - `eval_split`：基准测试评估集
    - `prompt_template`：基准测试提示模板
- 创建`MMLUProAdapter`类，继承自`DataAdapter`。

```{tip}
默认`subset_list`, `train_split`, `eval_split` 可以从数据集预览中获取，例如[MMLU-Pro预览](https://modelscope.cn/datasets/modelscope/MMLU-Pro/dataPeview)

![MMLU-Pro预览](./images/mmlu_pro_preview.png)
```




在完成`Benchmark`注册后，接下来需要编写`DataAdapter`类中的核心方法，以实现评测功能。这些方法控制着数据的加载、处理以及评分流程。



下面是MMLU-Pro适配器的完整实现，包含详细注释：

### 多项选择

## 4. 运行评测

调试代码，看看是否能正常运行。

```python
from evalscope import run_task, TaskConfig
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['mmlu_pro'],
    limit=10,
    dataset_args={'mmlu_pro': {'subset_list': ['computer science', 'math']}},
    debug=True
)
run_task(task_cfg=task_cfg)
```

输出如下：

```text
+-----------------------+-----------+-----------------+------------------+-------+---------+---------+
| Model                 | Dataset   | Metric          | Subset           |   Num |   Score | Cat.0   |
+=======================+===========+=================+==================+=======+=========+=========+
| Qwen2.5-0.5B-Instruct | mmlu_pro  | AverageAccuracy | computer science |     10 |       0.1 | default |
+-----------------------+-----------+-----------------+------------------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | mmlu_pro  | AverageAccuracy | math             |     10 |       0.1 | default |
+-----------------------+-----------+-----------------+------------------+-------+---------+---------+ 
```

## 5. 基准评测文档生成

完成基准评测实现后，您可以使用EvalScope提供的工具生成标准文档。这将确保您的基准评测有一致的文档格式，并能够被其他用户轻松理解和使用。

要生成中英文文档，请运行以下命令，将根据注册信息生成文档：

```bash
make docs
```

## 6. 提交PR
完成这些方法的实现和文档生成后，您的基准评测就准备就绪了！可以提交[PR](https://github.com/modelscope/evalscope/pulls)了。在提交之前请运行如下命令，将自动格式化代码：
```bash
make lint
```
确保没有格式问题后，我们将尽快合并你的贡献，让更多用户来使用你贡献的基准评测。如果你不知道如何提交PR，可以查看我们的[指南](https://github.com/modelscope/evalscope/blob/main/CONTRIBUTING.md)，快来试一试吧🚀
