# 自定义模型评测

EvalScope 基于 `ModelAPI` 抽象接口支持各种模型的评测。本文将介绍如何参考 `MockLLM` 实现自定义模型适配器。

## 什么情况下需要自定义模型适配器？

在以下情况下，您可能需要创建自定义模型适配器：

- 您的模型不支持标准的 OpenAI API 格式
- 您需要对模型输入输出进行特殊处理
- 您需要使用特定的推理参数或配置

## ModelAPI 接口定义

所有模型都需要实现 `ModelAPI` 抽象基类：

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo

class ModelAPI(ABC):
    """模型API提供者的基础接口"""
    
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.config = config

    @abstractmethod
    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """生成模型输出（必须实现）"""
        pass
```

## MockLLM 参考实现

`MockLLM` 是一个简单的测试模型实现，展示了 `ModelAPI` 的基本结构：

```python
from typing import Any, Dict, List, Optional
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo

class MockLLM(ModelAPI):
    """测试用的模拟模型实现"""
    
    default_output = 'Default output from mockllm/model'

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        custom_output: Optional[str] = None,
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        self.custom_output = custom_output

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 使用自定义输出（如果提供），否则使用默认输出
        output_text = self.custom_output if self.custom_output is not None else self.default_output
        
        return ModelOutput.from_content(
            model=self.model_name,
            content=output_text
        )
```

## 自定义模型实现

参考 `MockLLM` 的结构创建您的模型：

```python
from typing import List, Optional, Dict, Any
from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.api.registry import register_model_api

# 1. 使用register_model_api注册模型
@register_model_api(name='my_custom_model')
class MyCustomModel(ModelAPI):
    """自定义模型实现"""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        print(self.model_args)
        
        # 2. 在这里初始化您的模型
        # 例如：加载模型文件、建立连接等

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 3. 实现模型推理逻辑

        # 3.1 处理输入消息
        input_text = self._process_messages(input)
        
        # 3.2 调用您的模型
        response = self._call_model(input_text, config)
        
        # 3.3 返回标准化输出
        return ModelOutput.from_content(
            model=self.model_name,
            content=response
        )

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        """将聊天消息转换为文本"""
        text_parts = []
        for message in messages:
            role = getattr(message, 'role', 'user')
            content = getattr(message, 'content', str(message))
            text_parts.append(f"{role}: {content}")
        return '\n'.join(text_parts)

    def _call_model(self, input_text: str, config: GenerateConfig) -> str:
        """调用您的模型进行推理"""
        # 在这里实现您的模型调用逻辑
        # 例如：调用 API、本地模型推理等
        return f"Response to: {input_text}"
```



## 使用自定义模型

### 直接使用

```python
from evalscope import run_task, TaskConfig

# 创建模型实例
custom_model = MyCustomModel(
    model_name='my-model',
    model_args={'test': 'test'}
)

# 配置评测任务
task_config = TaskConfig(
    model=custom_model,
    datasets=['gsm8k'],
    limit=5
)

# 运行评测
results = run_task(task_cfg=task_config)
```

### 通过注册使用

```python
from evalscope import run_task, TaskConfig
from xxx import MyCustomModel # 需预先import模型，实现自动注册

# 使用注册的模型
task_config = TaskConfig(
    model='my-model',
    eval_type='my_custom_model', # register_model_api 使用的名称
    datasets=['gsm8k'],
    model_args={'test': 'test'},
    limit=5
)

results = run_task(task_cfg=task_config)
```

## 关键要点

1. **继承 `ModelAPI`** 并实现 `generate` 方法
2. **返回 `ModelOutput`** 对象，可以使用 `ModelOutput.from_content()` 创建
3. **处理异常** 确保模型调用的稳定性
4. **注册模型** 以便在配置中使用
