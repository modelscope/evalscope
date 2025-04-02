# 大语言模型

本框架支持选择题和问答题，两种预定义的数据集格式，使用流程如下：

## 选择题格式（MCQ）
适合用户是选择题的场景，评测指标为准确率（accuracy）。

### 1. 数据准备
准备选择题格式的csv文件，该目录结构如下：

```text
mcq/
├── example_dev.csv  # （可选）文件名组成为`{subset_name}_dev.csv`，用于fewshot评测
└── example_val.csv  # 文件名组成为`{subset_name}_val.csv`，用于实际评测的数据
```

其中csv文件需要为下面的格式：

```text
id,question,A,B,C,D,answer
1,通常来说，组成动物蛋白质的氨基酸有____,4种,22种,20种,19种,C
2,血液内存在的下列物质中，不属于代谢终产物的是____。,尿素,尿酸,丙酮酸,二氧化碳,C
```
其中：
- `id`是序号（可选字段）
- `question`是问题
- `A`, `B`, `C`, `D`等是可选项，最大支持10个选项
- `answer`是正确选项

### 2. 配置评测任务

运行下面的代码，即可开始评测：

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen2-0.5B-Instruct',
    datasets=['general_mcq'],  # 数据格式，选择题格式固定为 'general_mcq'
    dataset_args={
        'general_mcq': {
            "local_path": "custom_eval/text/mcq",  # 自定义数据集路径
            "subset_list": [
                "example"  # 评测数据集名称，上述subset_name
            ]
        }
    },
)
run_task(task_cfg=task_cfg)
```

运行结果：
```text
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Model               | Dataset     | Metric          | Subset   |   Num |   Score | Cat.0   |
+=====================+=============+=================+==========+=======+=========+=========+
| Qwen2-0.5B-Instruct | general_mcq | AverageAccuracy | example  |    12 |  0.5833 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
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
{"system": "你是一位地理学家", "query": "中国的首都是哪里？", "response": "中国的首都是北京"}
{"query": "世界上最高的山是哪座山？", "response": "是珠穆朗玛峰"}
{"query": "为什么北极见不到企鹅？", "response": "因为企鹅大多生活在南极"}
```

其中：
- `system`是系统prompt（可选字段）
- `query`是问题（必须）
- `response`是正确回答（必须）

### 2. 配置评测任务

运行下面的代码，即可开始评测：
```python
from evalscope import TaskConfig, run_task

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
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Model               | Dataset     | Metric          | Subset   |   Num |   Score | Cat.0   |
+=====================+=============+=================+==========+=======+=========+=========+
| Qwen2-0.5B-Instruct | general_qa  | bleu-1          | example  |    12 |  0.2324 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-2          | example  |    12 |  0.1451 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-3          | example  |    12 |  0.0625 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-4          | example  |    12 |  0.0556 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-f       | example  |    12 |  0.3441 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-p       | example  |    12 |  0.2393 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-r       | example  |    12 |  0.8889 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-f       | example  |    12 |  0.2062 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-p       | example  |    12 |  0.1453 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-r       | example  |    12 |  0.6167 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-f       | example  |    12 |  0.333  | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-p       | example  |    12 |  0.2324 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-r       | example  |    12 |  0.8889 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+ 
```