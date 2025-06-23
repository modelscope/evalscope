# Custom Model Evaluation

EvalScope supports model evaluation compatible with the OpenAI API format by default. However, for models that do not support the OpenAI API format, you can implement evaluations through custom model adapters (CustomModel). This document will guide you on how to create a custom model adapter and integrate it into the evaluation workflow.

## When Do You Need a Custom Model Adapter?

You might need to create a custom model adapter in the following situations:

1. Your model does not support the standard OpenAI API format.
2. You need to handle special processing for model input and output.
3. You need to use specific inference parameters or configurations.

## How to Implement a Custom Model Adapter

You need to create a class that inherits from `CustomModel` and implement the `predict` method:

```python
from evalscope.models import CustomModel
from typing import List

class MyCustomModel(CustomModel):
    def __init__(self, config: dict = None, **kwargs):
        # Initialize your model, you can pass model parameters in config
        super(MyCustomModel, self).__init__(config=config, **kwargs)
        
        # Initialize model resources as needed
        # For example: load model weights, connect to model service, etc.
        
    def predict(self, prompts: List[dict], **kwargs):
        """
        The core method for model inference, which takes input prompts and returns model responses
        
        Args:
            prompts: List of input prompts, each element is a dictionary
            **kwargs: Additional inference parameters
            
        Returns:
            A list of responses compatible with the OpenAI API format
        """
        # 1. Process input prompts
        # 2. Call your model for inference
        # 3. Convert model output to OpenAI API format
        # 4. Return formatted responses
```

## Example: DummyCustomModel

Below is a complete example of `DummyCustomModel` that demonstrates how to create and use a custom model adapter:

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

**Here is a complete example of evaluating using `DummyCustomModel`:**

```python
from evalscope import run_task, TaskConfig
from evalscope.models.custom.dummy_model import DummyCustomModel

# Instantiate DummyCustomModel
dummy_model = DummyCustomModel()

# Configure evaluation task
task_config = TaskConfig(
    model=dummy_model,
    model_id='dummy-model',  # Custom model ID
    datasets=['gsm8k'],
    eval_type='custom',  # Must be custom
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

# Run evaluation task
eval_results = run_task(task_cfg=task_config)
```

## Considerations for Implementing a Custom Model

1. **Input Format**: The `predict` method receives the `prompts` parameter as a list that contains a batch of input prompts. You need to ensure these prompts are converted into a format that the model can accept.

2. **Output Format**: The `predict` method must return a response list compatible with the OpenAI API format. Each response must include the `choices` field containing the content generated by the model.

3. **Error Handling**: Ensure your implementation contains appropriate error handling logic to prevent exceptions during the model inference process.

## Summary

By creating a custom model adapter, you can integrate any LLM model into the EvalScope evaluation framework, even if it does not natively support the OpenAI API format. The core of a custom model adapter is implementing the `predict` method, which converts input prompts into a format acceptable by the model, calls the model for inference, and then converts the model output to OpenAI API format.