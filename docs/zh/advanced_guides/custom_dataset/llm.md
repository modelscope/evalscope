# 大语言模型

本框架支持选择题、问答题、工具调用三种预定义的数据集格式，使用流程如下：

## 选择题格式（MCQ）
适合用户是选择题的场景，评测指标为准确率（accuracy）。

### 数据准备
准备选择题格式的文件，支持CSV和JSONL两种格式，该目录结构如下：

**CSV格式**

```text
mcq/
├── example_dev.csv   # （可选）文件名组成为`{subset_name}_dev.csv`，用于fewshot评测
└── example_val.csv   # 文件名组成为`{subset_name}_val.csv`，用于实际评测的数据
```

CSV文件需要为下面的格式：

```text
id,question,A,B,C,D,answer
1,通常来说，组成动物蛋白质的氨基酸有____,4种,22种,20种,19种,C
2,血液内存在的下列物质中，不属于代谢终产物的是____。,尿素,尿酸,丙酮酸,二氧化碳,C
```

**JSONL格式**

```text
mcq/
├── example_dev.jsonl # （可选）文件名组成为`{subset_name}_dev.jsonl`，用于fewshot评测
└── example_val.jsonl # 文件名组成为`{subset_name}_val.jsonl`，用于实际评测的数据
```

JSONL文件需要为下面的格式：

```json
{"id": "1", "question": "通常来说，组成动物蛋白质的氨基酸有____", "A": "4种", "B": "22种", "C": "20种", "D": "19种", "answer": "C"}
{"id": "2", "question": "血液内存在的下列物质中，不属于代谢终产物的是____。", "A": "尿素", "B": "尿酸", "C": "丙酮酸", "D": "二氧化碳", "answer": "C"}
```

其中：
- `id`是序号（可选字段）
- `question`是问题
- `A`, `B`, `C`, `D`等是可选项，最大支持10个选项
- `answer`是正确选项

### 配置评测任务

````{note}
默认的`prompt_template`支持四个选项，如下所示，其中`{query}`是prompt填充的位置。如果需要更少或更多选项，可以自定义`prompt_template`。
```text
请回答问题，并选出其中的正确答案。你的回答的最后一行应该是这样的格式：“答案是：LETTER”（不带引号），其中 LETTER 是 A、B、C、D 中的一个。
{query}
```
````

运行下面的代码，即可开始评测：

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen2-0.5B-Instruct',
    datasets=['general_mcq'],  # 数据格式，选择题格式固定为 'general_mcq'
    dataset_args={
        'general_mcq': {
            # 'prompt_template': 'xxx',  # 可以自定义prompt模板
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

本框架支持有参考答案和无参考答案两种问答题格式。

1. **有参考答案**问答：适合有明确的正确答案的问题，默认评测指标为`ROUGE`和`BLEU`，也可以配置LLM裁判，进行基于语义的正确性判断。
2. **无参考答案**问答：适合没有明确的正确答案的问题，例如开放式问答，默认不提供评测指标，可以配置LLM裁判，对生成的答案进行打分。

具体使用方法如下：


### 数据准备

准备问答题格式的jsonline文件，例如下面目录包含一个文件：

```text
qa/
└── example.jsonl
```

该 JSONL 文件为一行一条样本，支持以下三种结构（在同一文件中请选择一种并保持一致）：

```json
{"system": "你是一位地理学家", "query": "中国的首都是哪里？", "response": "中国的首都是北京"}
{"query": "世界上最高的山是哪座山？", "response": "是珠穆朗玛峰"}
{"messages": [{"role": "system", "content": "你是一位地理学家"}, {"role": "user", "content": "世界上最大的沙漠是哪个？"}], "response": "是撒哈拉沙漠"}
```

字段说明与必填项：
- 格式1（system + query [+ response]）
  - query：必填
  - response：有参考答案评测时必填；无参考答案评测时可省略或置空
  - system：可选
- 格式2（query [+ response]）
  - query：必填
  - response：有参考答案评测时必填；无参考答案评测时可省略或置空
- 格式3（messages [+ response]）
  - messages：必填，数组元素为 {"role": "system"|"user"|"assistant", "content": "<文本>"}；建议最后一条为用户提问
  - response：有参考答案评测时必填；无参考答案评测时可省略或置空

### 有参考答案问答

下面展示了如何配置有参考答案的问答题评测，以`Qwen2.5`模型为例，在`example.jsonl`上进行测试。

**方法1. 基于`ROUGE`和`BLEU`评测**

直接运行下面的代码：
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['general_qa'],  # 数据格式，问答题格式固定为 'general_qa'
    dataset_args={
        'general_qa': {
            "local_path": "custom_eval/text/qa",  # 自定义数据集路径
            "subset_list": [
                # 评测数据集名称，上述 *.jsonl 中的 *，可配置多个子数据集
                "example"       
            ]
        }
    },
)

run_task(task_cfg=task_cfg)
```

<details><summary>点击查看评测结果</summary>

```text
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric    | Subset   |   Num |   Score | Cat.0   |
+=======================+============+===========+==========+=======+=========+=========+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-1-R | example  |    12 |  0.6944 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-1-P | example  |    12 |  0.176  | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-1-F | example  |    12 |  0.2276 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-2-R | example  |    12 |  0.4667 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-2-P | example  |    12 |  0.0939 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-2-F | example  |    12 |  0.1226 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-L-R | example  |    12 |  0.6528 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-L-P | example  |    12 |  0.1628 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | Rouge-L-F | example  |    12 |  0.2063 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | bleu-1    | example  |    12 |  0.1644 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | bleu-2    | example  |    12 |  0.0935 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | bleu-3    | example  |    12 |  0.0655 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | general_qa | bleu-4    | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------+----------+-------+---------+---------+ 
```
</details>

**方法2. 基于LLM的评测**

基于LLM的评测可以方便的评测模型的正确性（或其他维度的指标，需要设置自定义的prompt）。下面是一个示例，配置了`judge_model_args`相关的参数，使用预置的`pattern`模式来判断模型输出是否正确。

完整的judge参数说明请[参考文档](../../get_started/parameters.md#judge参数)。

```python
import os
from evalscope import TaskConfig, run_task
from evalscope.constants import JudgeStrategy

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=[
        'general_qa',
    ],
    dataset_args={
        'general_qa': {
            'dataset_id': 'custom_eval/text/qa',
            'subset_list': [
                'example'
            ],
        }
    },
    # judge 相关参数
    judge_model_args={
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096
        },
        # 根据参考答案和模型输出，判断模型输出是否正确
        'score_type': 'pattern',
    },
    # judge 并发数
    judge_worker_num=5,
    # 使用 LLM 进行评测
    judge_strategy=JudgeStrategy.LLM,
)

run_task(task_cfg=task_cfg)
```

<details><summary>点击查看评测结果</summary>

```text
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric          | Subset   |   Num |   Score | Cat.0   |
+=======================+============+=================+==========+=======+=========+=========+
| Qwen2.5-0.5B-Instruct | general_qa | AverageAccuracy | example  |    12 |  0.5833 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+ 
```
</details>

### 无参考答案问答

若数据集没有参考答案，可以使用LLM裁判来评测模型输出的答案，不配置LLM将不会有打分结果。

下面是一个示例，配置了`judge_model_args`相关的参数，使用预置的`numeric`模式，从准确性、相关性、有用性等维度，自动综合判断模型输出得分，分数越高表示模型输出越好。

完整的judge参数说明请[参考文档](../../get_started/parameters.md#judge参数)。
```python
import os
from evalscope import TaskConfig, run_task
from evalscope.constants import JudgeStrategy

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=[
        'general_qa',
    ],
    dataset_args={
        'general_qa': {
            'dataset_id': 'custom_eval/text/qa',
            'subset_list': [
                'example'
            ],
        }
    },
    # judge 相关参数
    judge_model_args={
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 4096
        },
        # 直接打分
        'score_type': 'numeric',
    },
    # judge 并发数
    judge_worker_num=5,
    # 使用 LLM 进行评测
    judge_strategy=JudgeStrategy.LLM,
)

run_task(task_cfg=task_cfg)
```

<details><summary>点击查看评测结果</summary>

```text
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric          | Subset   |   Num |   Score | Cat.0   |
+=======================+============+=================+==========+=======+=========+=========+
| Qwen2.5-0.5B-Instruct | general_qa | AverageAccuracy | example  |    12 |  0.6375 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
```
</details>

## 工具调用格式（FC）

适用于需要函数调用决策与参数结构校验的评测场景，用于测试模型是否能在函数调用时正确决策并传递参数。支持：
- 预置数据集：evalscope/GeneralFunctionCall-Test
- 自定义数据集：通过本地路径传入 JSONL 文件

数据由官方 MoonshotAI [公开样本](https://github.com/MoonshotAI/K2-Vendor-Verifier)合成，包含“是否应触发工具”的标签（K2-Thinking模型作为标签来源），便于复现实验与自动化回归测试。

### 数据格式

数据以 JSONL 提供，每行一个样本，包含对话上下文、工具定义与是否应触发工具的标签。

示例结构（每行一个 JSON 对象）：
```json
{
    "messages": [
        {
            "role": "system",
            "content": "你是助手"
        },
        {
            "role": "user",
            "content": "请把 2 和 3 相加"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "将两个数字相加",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "第一个数字"
                        },
                        "b": {
                            "type": "number",
                            "description": "第二个数字"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ],
                    "additionalProperties": false
                }
            }
        }
    ],
    "should_call_tool": true
}
```

字段说明：
- messages：评测输入上下文（OpenAI 风格 role/content）
- tools：可用函数列表（OpenAI 工具定义，`type=function，function={name, parameters(JSON Schema)}`）
- should_call_tool：是否应触发工具的标签（来源于官方 K2-thinking 模型结果中 `finish_reason == "tool_calls"`）

```{note}
- 评测时，模型若给出 `finish_reason == "tool_calls"` 视为“模型预测调用工具”。
- 参数合法性使用 JSON Schema 校验（必须命名匹配、参数能解析为 JSON 并满足 schema）。
```

### 使用方法

```{note}
推荐**使用模型 API 服务**进行工具调用评测，并确保模型支持工具调用，本地模型推理暂不支持工具调用。
```

方法一：使用预置数据集（[evalscope/GeneralFunctionCall-Test](https://modelscope.cn/datasets/evalscope/GeneralFunctionCall-Test/dataPeview)）

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    datasets=['general_fc'],  # 工具调用格式固定为 'general_fc'
    # 如需显式指定，可加入 dataset_args={"general_fc": {"dataset_id": "evalscope/GeneralFunctionCall-Test"}}
)
run_task(task_cfg=task_cfg)
```

方法二：使用自定义本地数据集

目录结构示例：
```text
fc/
└── example.jsonl  # 每行一个样本，结构见上文
```

简单示例（example.jsonl，3 行）：
```json
{"messages":[{"role":"system","content":"你是助手"},{"role":"user","content":"请把 2 和 3 相加"}],"tools":[{"type":"function","function":{"name":"add","description":"将两个数字相加","parameters":{"type":"object","properties":{"a":{"type":"number","description":"第一个数字"},"b":{"type":"number","description":"第二个数字"}},"required":["a","b"],"additionalProperties":false}}}],"should_call_tool":true}
{"messages":[{"role":"system","content":"你是助手"},{"role":"user","content":"今天天气不错，我们聊聊天"}],"tools":[{"type":"function","function":{"name":"add","description":"将两个数字相加","parameters":{"type":"object","properties":{"a":{"type":"number","description":"第一个数字"},"b":{"type":"number","description":"第二个数字"}},"required":["a","b"],"additionalProperties":false}}}],"should_call_tool":false}
{"messages":[{"role":"system","content":"你是助手"},{"role":"user","content":"把 37 摄氏度转换为华氏度"}],"tools":[{"type":"function","function":{"name":"convert_temperature","description":"将摄氏度转换为华氏度","parameters":{"type":"object","properties":{"celsius":{"type":"number","description":"摄氏温度值"}},"required":["celsius"],"additionalProperties":false}}}],"should_call_tool":true}
```


运行示例：
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    datasets=['general_fc'],
    dataset_args={
        'general_fc': {
            "local_path": "custom_eval/text/fc",  # 自定义数据集根目录
            "subset_list": ["example"]       # 对应 example.jsonl
        }
    },
)
run_task(task_cfg=task_cfg)
```

### 评测指标说明

general_fc 输出以下指标：
- `count_finish_reason_tool_call`：模型预测调用工具的样本数（`finish_reason == "tool_calls"`）
- `count_successful_tool_call`：在尝试调用的样本中，工具名匹配且参数通过 JSON Schema 校验的样本数
- `schema_accuracy`：在尝试调用的样本中，参数校验通过的比例
- `tool_call_f1`：基于标签 `should_call_tool` 与模型是否调用工具的二分类 F1

示例输出：
```text
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| Model     | Dataset    | Metric                        | Subset   |   Num |   Score | Cat.0   |
+===========+============+===============================+==========+=======+=========+=========+
| qwen-plus | general_fc | count_finish_reason_tool_call | default  |    10 |  3      | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | count_successful_tool_call    | default  |    10 |  2      | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | schema_accuracy               | default  |    10 |  0.6667 | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
| qwen-plus | general_fc | tool_call_f1                  | default  |    10 |  0.5    | default |
+-----------+------------+-------------------------------+----------+-------+---------+---------+
```
