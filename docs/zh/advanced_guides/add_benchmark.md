# 添加基准评测

EvalScope作为[ModelScope](https://modelscope.cn)的官方评测工具，其基准评测功能正在持续优化中！我们诚邀您参考本教程，轻松添加自己的评测基准，并与广大社区成员分享您的贡献。一起助力EvalScope的成长，让我们的工具更加出色！

下面以`MMLU-Pro`为例，介绍如何添加基准评测，主要包含上传数据集、注册数据集、编写评测任务三个步骤。

## 上传基准评测数据集

上传基准评测数据集到ModelScope，这可以让用户一键加载数据集，让更多用户受益。当然，如果数据集已经存在，可以跳过这一步。

```{seealso}
例如：[modelscope/MMLU-Pro](https://modelscope.cn/datasets/modelscope/MMLU-Pro/summary)，参考[数据集上传教程](https://www.modelscope.cn/docs/datasets/create)。
```

请确保数据可以被modelscope加载，测试代码如下：

```python
from modelscope import MsDataset

dataset = MsDataset.load("modelscope/MMLU-Pro")  # 替换为你的数据集
```

## 注册基准评测

在EvalScope中添加基准评测。

### 创建文件结构

首先[Fork EvalScope](https://github.com/modelscope/evalscope/fork) 仓库，即创建一个自己的EvalScope仓库副本，将其clone到本地。

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

### 注册`Benchmark`

我们需要在`benchmark_name_adapter.py`中注册`Benchmark`，使得EvalScope能够加载我们添加的基准测试。以`MMLU-Pro`为例，主要包含以下内容：

- 导入`Benchmark`和`DataAdapter`
- 注册`Benchmark`，指定：
    - `name`：基准测试名称
    - `dataset_id`：基准测试数据集ID，用于加载基准测试数据集
    - `model_adapter`：基准测试模型适配器。改模型适配器用于本地加载模型推理，支持三种：
        - `ChatGenerationModelAdapter`：通用文本生成模型评测，通过输入prompt，返回模型生成的文本
        - `MultiChoiceModelAdapter`：多选题评测，通过logits来计算选项的概率，返回最大概率选项
        - `ContinuationLogitsModelAdapter`：多选文本评测，通过loglikelihood来计算每个上下文-延续对的对数似然值，返回对数似然值列表
    - `subset_list`：基准测试数据集的子数据集
    - `metric_list`：基准测试评估指标
    - `few_shot_num`：CoT评测的样本数量
    - `train_split`：基准测试训练集，用于采样CoT样例
    - `eval_split`：基准测试评估集
    - `prompt_template`：基准测试提示模板
- 创建`MMLUProAdapter`类，继承自`DataAdapter`。

```{tip}
`subset_list`, `train_split`, `eval_split` 可以从数据集预览中获取，例如[MMLU-Pro预览](https://modelscope.cn/datasets/modelscope/MMLU-Pro/dataPeview)

![MMLU-Pro预览](./images/mmlu_pro_preview.png)
```

代码示例如下：

```python
from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import WeightedAverageAccuracy
from evalscope.models import MultiChoiceModelAdapter


@Benchmark.register(
    name='mmlu_pro',
    dataset_id='modelscope/mmlu-pro',
    model_adapter=MultiChoiceModelAdapter,
    subset_list=['default'],
    metric_list=[WeightedAverageAccuracy],
    few_shot_num=0,
    train_split='validation',
    eval_split='test',
    prompt_template='You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`.',
)
class MMLUProAdapter(DataAdapter):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
```


## 编写评测逻辑

完成`DataAdapter`的编写，即可在EvalScope中添加评测任务。需要实现如下方法：

- `gen_prompt`：生成模型输入prompt
    - 对于类 `ChatGenerationModelAdapter`，输出格式为：`{'data': [full_prompt], 'system_prompt': (str, optional)}` 其中 `full_prompt: str`，每个数据样本构造的提示。

    - 对于类 `MultiChoiceModelAdapter`，输出格式为：`{'data': [full_prompt], 'multi_choices': self.choices}` 其中 `full_prompt: str`，每个数据样本构造的提示。

    - 对于类 `ContinuationEvalModelAdapter`，输出格式为：`{'data': ctx_continuation_pair_list, 'multi_choices': self.choices}` 其中 `ctx_continuation_pair_list: list`，上下文-延续对的列表。

```{note}
若`gen_prompt`提供的逻辑不符合预期，可以重写`gen_prompts`方法，来自定义从数据集到prompt的转换逻辑。
```

- `get_gold_answer`：解析数据集的标准答案
- `parse_pred_result`：解析模型输出，可以根据不同的eval_type返回不同的答案解析方式
- `match`：匹配模型输出和数据集标准答案，给出打分

完整示例代码如下：

```python
class MMLUProAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                            'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                            'psychology', 'history']
        
    
    def gen_prompts(self, data_dict: dict, **kwargs) -> Dict[str, list]:
        """
        Generate model prompt from raw input, unify the prompt format for MMLU-Pro benchmark.
        Return a dict with category as key and list of prompts as value.
        """
        
        data_dict = data_dict[self.subset_list[0]]  # Only one subset for MMLU-Pro
        fewshot_prompts = self.get_fewshot_examples(data_dict)
        
        #  Use the category as key to group the prompts
        res_dict = defaultdict(list)
        # generate prompts for each test sample
        for entry in data_dict[self.eval_split]:
            prefix = fewshot_prompts[entry['category']]
            query = prefix + 'Q: ' + entry['question'] + '\n' + \
                self.__form_options(entry['options']) + '\n'
            
            prompt_d = {
                'data': [query],
                'system_prompt': self.prompt_template,
                AnswerKeys.RAW_INPUT: entry
            }
            
            res_dict[entry['category']].append(prompt_d)
        return res_dict
    
    def get_fewshot_examples(self, data_dict: dict):
        # load 5-shot prompts for each category
        prompts = {c: '' for c in self.categories}
        for d in data_dict[self.train_split]:
            prompts[d['category']] += 'Q:' + ' ' + d['question'] + '\n' + \
                self.__form_options(d['options']) + '\n' + \
                d['cot_content'] + '\n\n'
        return prompts
    
    
    def __form_options(self, options: list):
        option_str = 'Options are:\n'
        for opt, choice in zip(options, self.choices):
            option_str += f'({choice}): {opt}' + '\n'
        return option_str
    
    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).

        Args:
            input_d: input raw data. Depending on the dataset.

        Returns:
            The parsed input. e.g. gold answer ... Depending on the dataset.
        """
        return input_d['answer']


    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or `custom`, default: 'checkpoint'

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        return ResponseParser.parse_first_option(result)


    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.

        Args:
            gold (Any): The golden answer. Usually a string for chat/multiple-choice-questions.
                        e.g. 'A', extracted from get_gold_answer method.
            pred (Any): The predicted answer. Usually a string for chat/multiple-choice-questions.
                        e.g. 'B', extracted from parse_pred_result method.

        Returns:
            The match result. Usually a score (float) for chat/multiple-choice-questions.
        """
        return exact_match(gold=gold, pred=pred)

```

## 运行评测

调试代码，看看是否能正常运行。

```python
from evalscope import run_task
task_cfg = {'model': 'qwen/Qwen2-0.5B-Instruct',
            'datasets': ['mmlu_pro'],
            'limit': 2,
            'debug': True}
run_task(task_cfg=task_cfg)
```

输出如下：

```text
+---------------------+-------------------------------------------+
| Model               | mmlu-pro                                  |
+=====================+===========================================+
| Qwen2-0.5B-Instruct | (mmlu-pro/WeightedAverageAccuracy) 0.1429 |
+---------------------+-------------------------------------------+ 
```

运行没问题的话，就可以提交[PR](https://github.com/modelscope/evalscope/pulls)了，我们将尽快合并你的贡献，让更多用户来使用你贡献的基准评测，快来试一试吧🚀
