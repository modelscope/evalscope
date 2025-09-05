# Custom Model Evaluation

EvalScope supports the evaluation of various models through the `ModelAPI` abstract interface. This document will guide you on how to implement a custom model adapter by referencing `MockLLM`.

## When Do You Need a Custom Model Adapter?

You might need to create a custom model adapter in the following scenarios:

- Your model does not support the standard OpenAI API format.
- You need to perform special processing on model inputs and outputs.
- You require specific inference parameters or configurations.

## ModelAPI Interface Definition

All models must implement the `ModelAPI` abstract base class:

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo

class ModelAPI(ABC):
    """Base interface for model API providers"""
    
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
        """Generate model output (must be implemented)"""
        pass
```

## MockLLM Reference Implementation

`MockLLM` is a simple test model implementation that demonstrates the basic structure of `ModelAPI`:

```python
from typing import Any, Dict, List, Optional
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo

class MockLLM(ModelAPI):
    """Mock model implementation for testing purposes"""
    
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
        # Use custom output if provided, otherwise use default output
        output_text = self.custom_output if self.custom_output is not None else self.default_output
        
        return ModelOutput.from_content(
            model=self.model_name,
            content=output_text
        )
```

## Custom Model Implementation

Create your model by referencing the structure of `MockLLM`:

```python
from typing import List, Optional, Dict, Any
from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.api.registry import register_model_api

# 1. Register the model using register_model_api
@register_model_api(name='my_custom_model')
class MyCustomModel(ModelAPI):
    """Custom model implementation"""

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
        
        # 2. Initialize your model here
        # For example: load model files, establish connections, etc.

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 3. Implement model inference logic

        # 3.1 Process input messages
        input_text = self._process_messages(input)
        
        # 3.2 Call your model
        response = self._call_model(input_text, config)
        
        # 3.3 Return standardized output
        return ModelOutput.from_content(
            model=self.model_name,
            content=response
        )

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to text"""
        text_parts = []
        for message in messages:
            role = getattr(message, 'role', 'user')
            content = getattr(message, 'content', str(message))
            text_parts.append(f"{role}: {content}")
        return '\n'.join(text_parts)

    def _call_model(self, input_text: str, config: GenerateConfig) -> str:
        """Invoke your model for inference"""
        # Implement your model invocation logic here
        # For example: API calls, local model inference, etc.
        return f"Response to: {input_text}"
```

## Using the Custom Model

### Direct Usage

```python
from evalscope import run_task, TaskConfig

# Create a model instance
custom_model = MyCustomModel(
    model_name='my-model',
    model_args={'test': 'test'}
)

# Configure the evaluation task
task_config = TaskConfig(
    model=custom_model,
    datasets=['gsm8k'],
    limit=5
)

# Run the evaluation
results = run_task(task_cfg=task_config)
```

### Usage Through Registration

```python
from evalscope import run_task, TaskConfig
from xxx import MyCustomModel # Import the model beforehand for automatic registration

# Use the registered model
task_config = TaskConfig(
    model='my-model',
    eval_type='my_custom_model', # Name used in register_model_api
    datasets=['gsm8k'],
    model_args={'test': 'test'},
    limit=5
)

results = run_task(task_cfg=task_config)
```

## Key Points

1. **Inherit from `ModelAPI`** and implement the `generate` method.
2. **Return a `ModelOutput`** object, which can be created using `ModelOutput.from_content()`.
3. **Handle exceptions** to ensure the stability of model calls.
4. **Register the model** for use in configurations.