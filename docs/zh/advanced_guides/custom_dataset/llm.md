# 大语言模型

本框架支持选择题和问答题，两种预定义的数据集格式，使用流程如下：

## 选择题格式（MCQ）
适合用户是选择题的场景，评测指标为准确率（accuracy）。

### 1. 数据准备
准备选择题格式的csv文件，该目录包含了两个文件：
```text
mcq/
├── example_dev.csv  # 名称需为*_dev.csv，用于fewshot评测，如果是0-shot评测，该csv可以为空
└── example_val.csv  # 名称需为*_val.csv，用于实际评测的数据
```

其中csv文件需要为下面的格式：

```text
id,question,A,B,C,D,answer,explanation
1,通常来说，组成动物蛋白质的氨基酸有____,4种,22种,20种,19种,C,1. 目前已知构成动物蛋白质的的氨基酸有20种。
2,血液内存在的下列物质中，不属于代谢终产物的是____。,尿素,尿酸,丙酮酸,二氧化碳,C,"代谢终产物是指在生物体内代谢过程中产生的无法再被利用的物质，需要通过排泄等方式从体内排出。丙酮酸是糖类代谢的产物，可以被进一步代谢为能量或者合成其他物质，并非代谢终产物。"
```
其中：
- `id`是评测序号
- `question`是问题
- `A` `B` `C` `D`是可选项（如果选项少于四个则对应留空）
- `answer`是正确选项
- `explanation`是解释（可选）

### 2. 配置文件
```python
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='qwen/Qwen2-0.5B-Instruct',
    datasets=['ceval'],  # 数据格式，选择题格式固定为 'ceval'
    dataset_args={
        'ceval': {
            "local_path": "custom_eval/text/mcq",  # 自定义数据集路径
            "subset_list": [
                "example"  # 评测数据集名称，上述 *_dev.csv 中的 *
            ]
        }
    },
)
run_task(task_cfg=task_cfg)
```

运行结果：
```text
+---------------------+---------------+
| Model               | mcq           |
+=====================+===============+
| Qwen2-0.5B-Instruct | (mcq/acc) 0.5 |
+---------------------+---------------+ 
```

## 问答题格式（QA）
适合用户是问答题的场景，评测指标是`ROUGE`和`BLEU`。

### 1. 数据准备
准备一个问答题格式的jsonline文件，该目录包含了一个文件：

```text
qa/
└── example.jsonl
```

该jsonline文件需要为下面的格式：

```json
{"query": "中国的首都是哪里？", "response": "中国的首都是北京"}
{"query": "世界上最高的山是哪座山？", "response": "是珠穆朗玛峰"}
{"query": "为什么北极见不到企鹅？", "response": "因为企鹅大多生活在南极"}
```

### 2. 配置文件
```python
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='qwen/Qwen2-0.5B-Instruct',
    datasets=['general_qa'],  # 数据格式，选择题格式固定为 'general_qa'
    dataset_args={
        'general_qa': {
            "local_path": "custom_eval/text/qa",  # 自定义数据集路径
            "subset_list": [
                "example"       # 评测数据集名称，上述 *.jsonl 中的 *
            ]
        }
    },
)

run_task(task_cfg=task_cfg)
```


运行结果：
```text
+---------------------+------------------------------------+
| Model               | qa                                 |
+=====================+====================================+
| Qwen2-0.5B-Instruct | (qa/rouge-1-r) 0.888888888888889   |
|                     | (qa/rouge-1-p) 0.2386966558963222  |
|                     | (qa/rouge-1-f) 0.3434493794481033  |
|                     | (qa/rouge-2-r) 0.6166666666666667  |
|                     | (qa/rouge-2-p) 0.14595543345543344 |
|                     | (qa/rouge-2-f) 0.20751474380718113 |
|                     | (qa/rouge-l-r) 0.888888888888889   |
|                     | (qa/rouge-l-p) 0.23344334652802393 |
|                     | (qa/rouge-l-f) 0.33456027373987435 |
|                     | (qa/bleu-1) 0.23344334652802393    |
|                     | (qa/bleu-2) 0.14571148341640142    |
|                     | (qa/bleu-3) 0.0625                 |
|                     | (qa/bleu-4) 0.05555555555555555    |
+---------------------+------------------------------------+
```

## (可选) 使用ms-swift框架自定义评测

```{seealso}
支持两种pattern的评测集：选择题格式的`CEval`和问答题格式的`General-QA`

参考：[ms-swift评测自定义评测集](../../best_practice/swift_integration.md#自定义评测集)
```
