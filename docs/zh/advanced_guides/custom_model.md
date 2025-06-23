# 自定义模型评测


EvalScope 默认支持兼容 OpenAI API 格式的模型评测。但对于不支持 OpenAI API 格式的模型，您可以通过自定义模型适配器（CustomModel）来实现评测。本文将指导您如何创建自定义模型适配器并集成到评测流程中。

## 什么情况下需要自定义模型适配器？

在以下情况下，您可能需要创建自定义模型适配器：

1. 您的模型不支持标准的 OpenAI API 格式
2. 您需要对模型输入输出进行特殊处理
3. 您需要使用特定的推理参数或配置

## 自定义模型适配器的实现方法

您需要创建一个继承自`CustomModel`的类，并实现`predict`方法：

```python
from evalscope.models import CustomModel
from typing import List

class MyCustomModel(CustomModel):
    def __init__(self, config: dict = None, **kwargs):
        # 初始化您的模型，可以在config中传入模型参数
        super(MyCustomModel, self).__init__(config=config, **kwargs)
        
        # 根据需要初始化模型资源
        # 例如：加载模型权重、连接到模型服务等
        
    def predict(self, prompts: List[dict], **kwargs):
        """
        模型推理的核心方法，接收输入提示并返回模型响应
        
        Args:
            prompts: 输入提示列表，每个元素是一个字典
            **kwargs: 额外的推理参数
            
        Returns:
            与OpenAI API格式兼容的响应列表
        """
        # 1. 处理输入提示
        # 2. 调用您的模型进行推理
        # 3. 将模型输出转换为OpenAI API格式
        # 4. 返回格式化后的响应
```


## 示例：DummyCustomModel

以下是一个完整的`DummyCustomModel`示例，展示了如何创建和使用自定义模型适配器：

```python
import time
from typing import List

from evalscope.utils.logger import get_logger
from evalscope.models import CustomModel

logger = get_logger()

class DummyCustomModel(CustomModel):

    def __init__(self, config: dict = {}, **kwargs):
        super(DummyCustomModel, self).__init__(config=config, **kwargs)
        
    def make_request_messages(self, input_item: dict) -> list:
        """
        Make request messages for OpenAI API.
        """
        if input_item.get('messages', None):
            return input_item['messages']

        data: list = input_item['data']
        if isinstance(data[0], tuple):  # for truthful_qa and hellaswag
            query = '\n'.join(''.join(item) for item in data)
            system_prompt = input_item.get('system_prompt', None)
        else:
            query = data[0]
            system_prompt = input_item.get('system_prompt', None)

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': query})

        return messages
    
    def predict(self, prompts: List[dict], **kwargs):

        original_inputs = kwargs.get('origin_inputs', None)
        infer_cfg = kwargs.get('infer_cfg', None)
        
        logger.debug(f'** Prompts: {prompts}')
        if original_inputs is not None:
            logger.debug(f'** Original inputs: {original_inputs}')
        if infer_cfg is not None:
            logger.debug(f'** Inference config: {infer_cfg}')

        # Simulate a response based on the prompts
        # Must return a list of dicts with the same format as the OpenAI API.
        responses = []
        for input_item in original_inputs:
            message = self.make_request_messages(input_item)

            # You can replace this with actual model inference logic
            # For demonstration, we will just return a dummy response
            response = f"Dummy response for prompt: {message}"

            res_d = {
                'choices': [{
                    'index': 0,
                    'message': {
                        'content': response,
                        'role': 'assistant'
                    }
                }],
                'created': time.time(),
                'model': self.config.get('model_id'),
                'object': 'chat.completion',
                'usage': {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0
                }
            }
            
            responses.append(res_d)
            
        return responses
```

**以下是使用`DummyCustomModel`进行评测的完整示例**：

```python
from evalscope import run_task, TaskConfig
from evalscope.models.custom.dummy_model import DummyCustomModel

# 实例化DummyCustomModel
dummy_model = DummyCustomModel()

# 配置评测任务
task_config = TaskConfig(
    model=dummy_model,
    model_id='dummy-model',  # 自定义模型ID
    datasets=['gsm8k'],
    eval_type='custom',  # 必须为custom
    generation_config={
        'max_new_tokens': 100,
        'temperature': 0.0,
        'top_p': 1.0,
        'top_k': 50,
        'repetition_penalty': 1.0
    },
    debug=True,
    limit=5,
)

# 运行评测任务
eval_results = run_task(task_cfg=task_config)
```

## 自定义模型实现的注意事项

1. **输入格式**：`predict`方法接收的`prompts`参数是一个列表，包含一个batch的输入提示。您需要确保将这些提示转换为模型可以接受的格式。

2. **输出格式**：`predict`方法必须返回与OpenAI API格式兼容的响应列表。每个响应必须包含`choices`字段，其中包含模型生成的内容。

3. **错误处理**：确保您的实现包含适当的错误处理逻辑，以防模型推理过程中出现异常。


## 总结

通过创建自定义模型适配器，您可以将任何LLM模型集成到EvalScope评测框架中，即使它不原生支持OpenAI API格式。自定义模型适配器的核心是实现`predict`方法，将输入提示转换为模型可接受的格式，调用模型进行推理，然后将模型输出转换为OpenAI API格式。
