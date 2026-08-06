# 👍 贡献基准评测

EvalScope作为[ModelScope](https://modelscope.cn)的官方评测工具，其基准评测功能正在持续优化中！我们诚邀您参考本教程，轻松添加自己的评测基准，并与广大社区成员分享您的贡献。一起助力EvalScope的成长，让我们的工具更加出色！

下面将介绍如何添加**通用文本推理**和**多项选择**两种基准评测，主要包含上传数据集、注册数据集、编写评测任务三个步骤。

## 基础概念

```{tip}
您可以先跳过本节，直接从[准备基准评测数据集](#1-准备基准评测数据集)开始，遇到不理解的代码后再查看具体的实现。
```

EvalScope评测流程主要包含以下步骤：
1. **数据准备**：通过`DataAdapter`加载和预处理数据集。
2. **任务定义**：通过`TaskConfig`定义评测任务的配置，包括模型、数据集、评估指标等。
3. **评测执行**：通过`run_task`函数执行评测任务，并输出评测结果。

其中`DataAdapter`是我们需要重点了解的类，它是基准评测的核心组件。

### DataAdapter架构和调用流程

DataAdapter采用Pipeline架构，支持通过钩子方法自定义行为。以`DefaultDataAdapter`为例，完整的评测流程如下：

```
1. 数据加载阶段
   load_dataset() 
   ├── load() 
   │   ├── load_from_remote() / load_from_disk()
   │   │   ├── load_subsets()
   │   │   │   └── load_subset() / load_fewshot_subset()
   │   │   │       └── record_to_sample() [用户实现]
   │   │   └── _post_process_samples()
   │   │       └── process_sample_input()
   │   │           ├── sample_to_fewshot() [用户实现]
   │   │           ├── format_fewshot_template() [用户可选实现]
   │   │           └── format_prompt_template() [用户可选实现]
   │   └── 返回 DatasetDict

2. 模型推理阶段（每个样本）
   run_inference()
   ├── _on_inference_start() [钩子方法]
   ├── _on_inference() [钩子方法]
   └── _on_inference_end() [钩子方法]
       └── 返回 TaskState

3. 指标计算阶段（每个样本）
   calculate_metrics()
   ├── filter_prediction()
   │   └── extract_answer() [用户可选实现]
   ├── match_score() / llm_match_score()
   └── 返回 SampleScore

4. 结果聚合阶段
   aggregate_scores()
   └── 返回 List[AggScore]

5. 报告生成阶段
   generate_report()
   ├── _on_generate_report() [钩子方法]
   └── _on_generate_report_end() [钩子方法]
       └── 返回 Report
```

### 核心数据结构

#### 1. Sample对象
表示单个评测样本，包含输入、目标答案和元数据：

```python
@dataclass
class Sample:
    input: Any                    # 输入内容（问题文本或聊天消息列表）
    target: str                   # 目标答案（正确答案）
    choices: Optional[List[str]] = None    # 选择项（多选题使用）
    subset_key: Optional[str] = None       # 子集划分键（用于按类别分组）
    metadata: Optional[Dict] = None        # 元数据（推理过程、ID等）
    tools: Optional[List] = None           # 工具调用信息
```

#### 2. TaskState对象
表示单次推理任务的完整状态：

```python
@dataclass
class TaskState:
    model: str                    # 模型名称
    sample: Sample               # 输入样本
    messages: List[ChatMessage]  # 聊天消息历史
    output: ModelOutput          # 模型原始输出
    completed: bool              # 任务是否完成
    sample_id: Optional[str] = None      # 样本ID
    group_id: Optional[str] = None       # 分组ID
    metadata: Optional[Dict] = None      # 任务元数据
```

#### 3. ModelOutput对象
表示模型的原始输出：

```python
@dataclass
class ModelOutput:
    completion: str              # 模型生成的文本
    message: ChatMessage         # 格式化的聊天消息
    # 其他模型特定字段...
```

#### 4. Score对象
表示单个样本的评分结果：

```python
@dataclass
class Score:
    value: Dict[str, float]      # 各指标的得分 {"acc": 1.0, "f1": 0.8}
    extracted_prediction: str    # 提取的预测答案
    prediction: str              # 原始预测文本
    metadata: Dict = None        # 评分元数据
```

#### 5. SampleScore对象
封装单个样本的完整评分信息：

```python
@dataclass
class SampleScore:
    score: Score                 # 评分对象
    sample_id: Optional[str]     # 样本唯一标识
    group_id: Optional[str]      # 分组标识
    sample_metadata: Optional[Dict] = None  # 样本元数据
```

#### 6. AggScore对象
表示聚合后的评分统计：

```python
@dataclass
class AggScore:
    metric: str                  # 指标名称
    value: float                 # 聚合值（如平均分）
    subset: str                  # 子集名称
    num_samples: int             # 样本数量
    agg_method: str              # 聚合方法（mean, median等）
    metadata: Dict = None        # 聚合元数据
```

#### 7. DatasetDict对象
管理多个数据集子集：

```python
class DatasetDict(dict):
    """数据集字典，键为子集名称，值为Dataset对象"""
    
    @classmethod
    def from_dataset(cls, dataset, subset_list=None, limit=None, repeats=1):
        """从单个数据集创建多子集数据集字典"""
        pass
```

### DataAdapter核心方法详解

基于上述调用流程，以下是需要用户实现或可选重写的关键方法：

#### 必须实现的方法

1. **`record_to_sample(record: Dict[str, Any]) -> Sample`**
   - **作用**：将原始数据记录转换为标准Sample对象
   - **输入**：数据集中的原始记录字典
   - **输出**：标准化的Sample对象
   - **示例**：
   ```python
   def record_to_sample(self, record: Dict[str, Any]) -> Sample:
       return Sample(
           input=record['question'],
           target=record['answer'],
           metadata={'reasoning': record.get('explanation', '')}
       )
   ```

#### 可选实现的方法

2. **`sample_to_fewshot(sample: Sample) -> str`**
   - **作用**：将样本转换为few-shot示例字符串
   - **输入**：Sample对象
   - **输出**：格式化的few-shot示例文本
   - **调用时机**：构建few-shot提示时

3. **`extract_answer(prediction: str, task_state: TaskState) -> str`**
   - **作用**：从模型原始输出中提取最终答案
   - **输入**：模型预测文本和任务状态
   - **输出**：提取的答案字符串
   - **调用时机**：计算指标前的答案清理

4. **`format_prompt_template(sample: Sample) -> str`**
   - **作用**：格式化基础提示模板
   - **输入**：Sample对象
   - **输出**：格式化的提示文本
   - **默认实现**：使用`prompt_template.format(question=sample.input)`

5. **`format_fewshot_template(fewshot: str, sample: Sample) -> str`**
   - **作用**：格式化包含few-shot的提示模板
   - **输入**：few-shot示例字符串和Sample对象
   - **输出**：完整的few-shot提示
   - **默认实现**：使用`few_shot_prompt_template.format()`

6. **`sample_filter(sample: Sample) -> bool`**
   - **作用**：过滤数据集样本
   - **输入**：Sample对象
   - **输出**：是否保留该样本
   - **默认实现**：返回True（保留所有样本）

### 钩子方法系统

DataAdapter提供了钩子方法系统，支持在关键节点插入自定义逻辑：

#### 推理阶段钩子
- **`_on_inference_start(model, sample)`**：推理开始前
- **`_on_inference(model, sample)`**：执行推理
- **`_on_inference_end(model, sample, model_output, output_dir)`**：推理结束后

#### 报告生成钩子
- **`_on_generate_report(scores, model_name)`**：生成报告
- **`_on_generate_report_end(report, output_dir)`**：报告生成后

### 适配器类型

EvalScope提供了两种主要的适配器基类：

1. **`DefaultDataAdapter`**：通用文本推理任务的基础适配器
   - 适用于开放式问答、数学推理、代码生成等任务
   - 需要自定义答案提取逻辑

2. **`MultiChoiceAdapter`**：多项选择任务的专用适配器
   - 继承自`DefaultDataAdapter`
   - 内置选择项格式化和答案提取逻辑
   - 支持单选和多选模式

选择适配器类型的原则：
- 如果任务涉及从固定选项中选择答案 → 使用`MultiChoiceAdapter`
- 如果任务需要生成开放式答案 → 使用`DefaultDataAdapter`

## 1. 准备基准评测数据集

您有两种方式准备基准评测数据集：

1. **上传到ModelScope（推荐）**：将数据集上传到ModelScope平台，这样其他用户可以一键加载您的数据集，使用更加便捷，也能让更多用户受益于您的贡献。如需上传到ModelScope，可参考[数据集上传教程](https://www.modelscope.cn/docs/datasets/create)。

2. **本地使用**：您也可以直接使用本地数据集进行评测，适合数据集尚在开发阶段或含有敏感信息的情况。

无论选择哪种方式，请确保数据的格式正确且可被加载。如使用本地数据集，可通过以下代码测试：

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
具体到`GSM8K`和`MMLU-Pro`，结构如下：

```text
evalscope/benchmarks/
├── gsm8k
│   ├── __init__.py
│   ├── gsm8k_adapter.py
├── mmlu_pro
│   ├── __init__.py
│   ├── mmlu_pro_adapter.py
│   └── ...
```

## 3. 编写评测逻辑

下面将以**GSM8K**和**MMLU-Pro**为例，分别介绍**通用文本推理**和**多项选择**两种评测任务。

### 通用文本推理

通用文本推理任务通常要求模型对给定问题进行分析和推理，然后生成答案。以GSM8K（数学推理）为例：

我们需要在`gsm8k_adapter.py`中注册`Benchmark`并实现`GSM8KAdapter`类：

```python
from typing import Any, Dict
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

# 定义提示模板
PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".lstrip()

# 注册基准评测
@register_benchmark(
    BenchmarkMeta(
        name='gsm8k',                          # 基准测试名称
        pretty_name='GSM8K',                   # 可读名称
        dataset_id='AI-ModelScope/gsm8k',      # 数据集ID 或 本地路径
        tags=[Tags.MATH, Tags.REASONING],      # 标签
        description='GSM8K (Grade School Math 8K) is a dataset of grade school math problems, designed to evaluate the mathematical reasoning abilities of AI models.',
        subset_list=['main'],                  # 子数据集列表
        few_shot_num=4,                       # few-shot示例数量
        train_split='train',                  # 训练集split名称
        eval_split='test',                    # 评测集split名称
        metric_list=['acc'],                  # 评估指标
        prompt_template=PROMPT_TEMPLATE,      # 提示模板
    )
)
class GSM8KAdapter(DefaultDataAdapter):
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """将原始数据记录转换为Sample对象"""
        DELIM = '####'
        question = record['question']
        answer = record['answer'].split(DELIM)
        target = answer.pop().strip()  # 提取最终答案
        reasoning = DELIM.join(answer)  # 提取推理过程
        
        return Sample(
            input=question,
            target=target,
            metadata={'reasoning': reasoning.strip()}
        )
    
    def sample_to_fewshot(self, sample: Sample) -> str:
        """将样本转换为few-shot示例"""
        if sample.metadata:
            return (
                f'{sample.input}\n\nReasoning:\n' + 
                f"{sample.metadata['reasoning']}\n\n" + 
                f'ANSWER: {sample.target}'
            )
        else:
            return ''
    
    def extract_answer(self, prediction: str, task_state: TaskState):
        """从模型预测中提取答案"""
        from evalscope.filters.extraction import RegexFilter
        
        # 使用正则表达式提取数字答案
        regex = RegexFilter(regex_pattern=r'(-?[0-9.,]{2,})|(-?[0-9]+)', group_select=-1)
        res = regex(prediction)
        return res.replace(',', '').replace('+', '').strip().strip('.')
```

### 多项选择

多项选择任务要求模型从给定选项中选择正确答案。以MMLU-Pro为例，我们需要继承`MultiChoiceAdapter`：

```python
from typing import Any, Dict
from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

# 定义提示模板
USER_PROMPT_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}
""".lstrip()

SUBSET_LIST = [
    'computer science', 'math', 'chemistry', 'engineering', 'law', 'biology', 
    'health', 'physics', 'business', 'philosophy', 'economics', 'other', 
    'psychology', 'history'
]

@register_benchmark(
    BenchmarkMeta(
        name='mmlu_pro',
        pretty_name='MMLU-Pro',
        tags=[Tags.MULTIPLE_CHOICE, Tags.KNOWLEDGE],
        description='MMLU-Pro is a benchmark for evaluating language models on multiple-choice questions across various subjects.',
        dataset_id='modelscope/MMLU-Pro',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        few_shot_num=5,
        train_split='validation',
        eval_split='test',
        prompt_template=USER_PROMPT_TEMPLATE,
    )
)
class MMLUProAdapter(MultiChoiceAdapter):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True  # 启用子集划分
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """将原始数据记录转换为Sample对象"""
        return Sample(
            input=record['question'],
            choices=record['options'],      # 选择项列表
            target=record['answer'],        # 正确答案（如'A'）
            subset_key=record['category'].lower(),  # 用于子集划分的key
            metadata={
                'cot_content': record['cot_content'],
                'subject': record['category'].lower(),
                'question_id': record['question_id'],
            },
        )
    
    def sample_to_fewshot(self, sample: Sample) -> str:
        """将样本转换为few-shot示例"""
        q_str = f"""Question:\n{str(sample.input)}"""
        options = sample.choices if sample.choices is not None else []
        
        # 格式化选择项
        opt_str_list = []
        for i, opt in enumerate(options):
            opt_str_list.append(f"""{chr(65 + i)} {opt}""")
        opt_str = f"""Options:\n{'\n'.join(opt_str_list)}"""
        
        # 处理答案和推理过程
        ans_str = sample.metadata['cot_content'] if sample.metadata is not None else ''
        ans_str = ans_str.replace('The answer is', 'ANSWER:')
        ans_opt = ans_str.split('ANSWER:')[-1].split('.')[0].strip().strip('(').strip(')')
        ans_str = ans_str.replace(f'ANSWER: ({ans_opt})', f'ANSWER: {ans_opt}')
        
        final_str = '\n'.join([q_str, opt_str, ans_str])
        return final_str
```

### 关键差异说明

**通用文本推理** vs **多项选择**：

1. **继承的基类**：
   - 通用文本推理：继承`DefaultDataAdapter`
   - 多项选择：继承`MultiChoiceAdapter`

2. **Sample对象结构**：
   - 通用文本推理：主要包含`input`和`target`
   - 多项选择：额外包含`choices`（选择项列表）

3. **答案提取方法**：
   - 通用文本推理：需要自定义`extract_answer()`方法
   - 多项选择：`MultiChoiceAdapter`提供了标准的答案提取逻辑

4. **提示模板**：
   - 通用文本推理：更注重推理过程的引导
   - 多项选择：专注于选择项的展示和答案格式

### 指标语义与主指标

报告不会猜测指标的含义。指标如何展示——名称、优化方向、单位、刻度与精度——统一来自
`evalscope/metrics/semantics/catalog.py` 中的集中目录；每个 benchmark 则声明自己的哪个指标承载结论。

**大多数新 benchmark 无需改动目录。** 复用已有指标名（`acc`、`f1_score`、`exact_match`、
`pass_rate` 等）时，语义已经声明好了：

```python
metric_list=['acc'],   # 'acc' 已在目录中：无需其它改动
```

只有两种情况需要你添一行：

1. **产出多个指标时**：声明哪个是主指标，填 `metric_list` 中的原始名。其余指标自动解析为
   auxiliary，占比类、计数类指标解析为 diagnostic：

   ```python
   metric_list=['precision', 'recall', 'f1_score', 'accuracy'],
   primary_metric='f1_score',   # metric_list 多于一个指标时必填
   ```

   这一点是强制的：多指标却不写 `primary_metric` 会在构造 `BenchmarkMeta` 时报错——否则每个报告
   视图都只能任选一个指标当结论。

2. **引入了新的指标名时**：在 `METRIC_NAME_SEMANTICS` 中加一行，引用描述它的基线：

   ```python
   # evalscope/metrics/semantics/catalog.py
   'my_new_score': MetricEntry(baseline='quality.accuracy.ratio'),
   ```

跑审计即可看到还缺什么，它会给出 benchmark 名、指标名与待改文件，无需人工猜测：

```bash
make metric-audit
```

仓库外的第三方 benchmark 不改目录也能跑：其未声明的指标降级为 diagnostic，原样显示数值，不伪造
方向与单位。内置 benchmark 则不行——未声明的指标会让报告生成失败，否则目录的空洞就会表现为一个被
静默错误渲染的分数。

## 4. 运行评测

调试代码，看看是否能正常运行。

**GSM8K示例**：
```python
from evalscope import run_task, TaskConfig

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['gsm8k'],
    limit=10,
    debug=True
)
run_task(task_cfg=task_cfg)
```

**MMLU-Pro示例**：
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

输出示例：

```text
+-----------------------+-----------+-----------------+------------------+-------+---------+---------+
| Model                 | Dataset   | Metric          | Subset           |   Num |   Score | Cat.0   |
+=======================+===========+=================+==================+=======+=========+=========+
| Qwen2.5-0.5B-Instruct | gsm8k     | mean_acc        | main             |    10 |     0.3 | default |
+-----------------------+-----------+-----------------+------------------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | mmlu_pro  | mean_acc        | computer science |    10 |     0.1 | default |
+-----------------------+-----------+-----------------+------------------+-------+---------+---------+
| Qwen2.5-0.5B-Instruct | mmlu_pro  | mean_acc        | math             |    10 |     0.1 | default |
+-----------------------+-----------+-----------------+------------------+-------+---------+---------+
```

## 5. 基准评测文档生成

完成基准评测实现后，您可以使用EvalScope提供的工具生成标准文档。这将确保您的基准评测有一致的文档格式，并能够被其他用户轻松理解和使用。

要生成中英文文档，请运行以下命令，将根据注册信息生成文档：

```bash
pip install -e '.[docs]'
make docs
```

## 6. 提交PR
完成这些方法的实现和文档生成后，您的基准评测就准备就绪了！可以提交[PR](https://github.com/modelscope/evalscope/pulls)了。在提交之前请运行如下命令，将自动格式化代码：
```bash
make lint
```
确保没有格式问题后，我们将尽快合并你的贡献，让更多用户来使用你贡献的基准评测。如果你不知道如何提交PR，可以查看我们的[指南](https://github.com/modelscope/evalscope/blob/main/CONTRIBUTING.md)，快来试一试吧🚀
