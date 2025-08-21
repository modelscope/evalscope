from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Union

from evalscope.api.dataset import Dataset
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.function_utils import thread_safe


class MockLLM(ModelAPI):
    """A mock implementation of the ModelAPI class for testing purposes.

    Always returns default_output, unless you pass in a model_args
    key "custom_outputs" with a value of an Iterable[ModelOutput]
    """

    default_output = 'Default output from mockllm/model'

    outputs: Iterator[ModelOutput]

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        custom_outputs: Iterable[ModelOutput] = None,
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        if custom_outputs is not None:
            # We cannot rely on the user of this model giving custom_outputs
            # the correct type since they do not call this constructor
            # Hence this type check and the one in generate.
            if not isinstance(custom_outputs, (Iterable, Generator)):
                raise ValueError(
                    f"model_args['custom_outputs'] must be an Iterable or a Generator, got {custom_outputs}"
                )
            self.outputs = iter(custom_outputs)
        else:
            self.outputs = iter((
                ModelOutput.from_content(model='mockllm', content=self.default_output)
                for _ in iter(int, 1)  # produce an infinite iterator
            ))

    @thread_safe
    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        try:
            output = next(self.outputs)
        except StopIteration:
            raise ValueError('custom_outputs ran out of values')

        if not isinstance(output, ModelOutput):
            raise ValueError(f'output must be an instance of ModelOutput; got {type(output)}; content: {repr(output)}')
        return output

    def batch_generate(inputs: Dataset, config: GenerateConfig) -> List[ModelOutput]:
        return super().batch_generate(inputs, config)
