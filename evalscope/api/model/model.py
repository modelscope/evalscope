import abc
from pydantic_core import to_jsonable_python
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Optional, Sequence, Union

from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from evalscope.api.registry import get_model_api
from evalscope.api.tool import ToolChoice, ToolFunction, ToolInfo
from evalscope.utils import get_logger
from evalscope.utils.function_utils import thread_safe
from .generate_config import GenerateConfig
from .model_output import ModelOutput

if TYPE_CHECKING:
    from evalscope.config import TaskConfig

logger = get_logger()


class ModelAPI(abc.ABC):
    """Model API provider."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs
    ) -> None:
        """Create a model API provider.

        Args:
           model_name (str): Model name.
           base_url (str | None): Alternate base URL for model.
           api_key (str | None): API key for model.
           api_key_vars (list[str]): Environment variables that
              may contain keys for this provider (used for override)
           config (GenerateConfig): Model configuration.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.config = config

    @abc.abstractmethod
    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate output from the model.

        Args:
          input (str | list[ChatMessage]): Chat message
            input (if a `str` is passed it is converted
            to a `ChatUserMessage`).
          tools (list[ToolInfo]): Tools available for the
            model to call.
          tool_choice (ToolChoice): Directives to the model
            as to which tools to prefer.
          config (GenerateConfig): Model configuration.

        Returns:
           ModelOutput
        """
        ...

    def batch_generate(
        self,
        inputs: List[List[ChatMessage]],
        tools: List[List[ToolInfo]],
        tool_choices: List[ToolChoice],
        configs: List[GenerateConfig],
    ) -> Generator[ModelOutput, None, None]:
        """Default batch implementation using individual generate calls.

        ModelAPI implementations can override this for optimized batch processing.

        Args:
          inputs: List of preprocessed chat message inputs.
          tools: List of tools for each input.
          tool_choices: List of tool choices for each input.
          configs: List of configs for each input.

        Returns:
            Generator yielding ModelOutput for each input.
        """
        from concurrent.futures import ThreadPoolExecutor

        def single_generate(args):
            input_msgs, input_tools, tool_choice, config = args
            return self.generate(input_msgs, input_tools, tool_choice, config)

        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            futures = []
            for input_msgs, input_tools, tool_choice, config in zip(inputs, tools, tool_choices, configs):
                future = executor.submit(single_generate, (input_msgs, input_tools, tool_choice, config))
                futures.append(future)

            for future in futures:
                yield future.result()

    def supports_batch(self) -> bool:
        """Whether this ModelAPI supports optimized batch processing."""
        return False

    def max_tokens(self) -> Optional[int]:
        """Default max_tokens."""
        return None

    def max_tokens_for_config(self, config: GenerateConfig) -> Optional[int]:
        """Default max_tokens for a given config.

        Args:
           config: Generation config.

        Returns:
           Default maximum tokens for specified configuration.
        """
        return None

    def tools_required(self) -> bool:
        """Any tool use in a message stream means that tools must be passed."""
        return False

    def tool_result_images(self) -> bool:
        """Tool results can contain images"""
        return False


class Model:
    """Model interface.

    Use `get_model()` to get an instance of a model.
    """

    api: ModelAPI
    """Model API."""

    config: GenerateConfig
    """Generation config."""

    def __init__(self, api: ModelAPI, config: GenerateConfig, model_args: Dict[str, Any] = {}) -> None:
        """Create a model.

        Args:
           api: Model API provider.
           config: Model configuration.
           model_args: Optional model args
        """
        self.api = api
        self.config = config
        self.model_args = model_args

    @property
    def name(self) -> str:
        """Model name or path to model."""
        return self.api.model_name

    @property
    def role(self) -> Optional[str]:
        """Model role."""
        return self._role

    @role.setter
    def role(self, role: str) -> None:
        self._role = role

    def __str__(self) -> str:
        return f'Model(name={self.model_id}, role={self.role})'

    def generate(
        self,
        input: Union[str, List[ChatMessage]],
        tools: Optional[Sequence[ToolInfo]] = None,
        tool_choice: Optional[ToolChoice] = None,
        config: Optional[GenerateConfig] = None,
    ) -> ModelOutput:
        """Generate output from the model.

        Args:
          input: Chat message input (if a `str` is passed it is converted
            to a `ChatMessageUser`).
          tools: Tools available for the model to call.
          tool_choice: Directives to the model as to which tools to prefer.
          config: Model configuration.

        Returns:
           ModelOutput
        """
        processed_input, processed_tools, processed_tool_choice, processed_config = self._preprocess_input(
            input, tools, tool_choice, config
        )

        # Call the model's generate method
        output = self.api.generate(
            input=processed_input,
            tools=processed_tools,
            tool_choice=processed_tool_choice,
            config=processed_config,
        )

        # return output
        return output

    def batch_generate(
        self,
        inputs: List[List[ChatMessage]],
        tools: List[List[ToolInfo]],
        tool_choices: List[ToolChoice],
        configs: List[GenerateConfig],
    ) -> Generator[ModelOutput, None, None]:
        """Generate output from the model for a batch of inputs.

        Args:
          inputs (List[List[ChatMessage]]): Batch of chat message inputs.
          tools (List[List[ToolInfo]]): Batch of tools for each input.
          tool_choices (List[ToolChoice]): Batch of tool choices for each input.
          configs (List[GenerateConfig]): Batch of configs for each input.
        """
        preprocessed_data = []

        for input_item, input_tools, input_tool_choice, input_config in zip(inputs, tools, tool_choices, configs):
            processed_input, processed_tools, processed_tool_choice, processed_config = self._preprocess_input(
                input=input_item, tools=input_tools, tool_choice=input_tool_choice, config=input_config
            )
            preprocessed_data.append((processed_input, processed_tools, processed_tool_choice, processed_config))

        # check if ModelAPI supports batch processing
        if self.api.supports_batch() and len(preprocessed_data) > 1:
            # use the batch_generate method of the ModelAPI
            inputs, tools, tool_choices, configs = zip(*preprocessed_data)
            batch_results = self.api.batch_generate(
                inputs=list(inputs), tools=list(tools), tool_choices=list(tool_choices), configs=list(configs)
            )
            for result in batch_results:
                yield result
        else:
            # fall back to processing each input individually
            for input_msgs, input_tools, tool_choice, config in preprocessed_data:
                result = self.api.generate(input_msgs, input_tools, tool_choice, config)
                yield result

    def _preprocess_input(
        self,
        input: Union[str, List[ChatMessage]],
        tools: Optional[Sequence[ToolInfo]] = None,
        tool_choice: Optional[ToolChoice] = None,
        config: Optional[GenerateConfig] = None,
    ) -> tuple[List[ChatMessage], List[ToolInfo], ToolChoice, GenerateConfig]:
        """pre process input for generate."""

        # merge passed config
        if config is not None:
            config = self.config.merge(config)
        else:
            config = self.config.model_copy(deep=True)

        # provide max_tokens from the model api if required
        if config.max_tokens is None:
            config.max_tokens = self.api.max_tokens_for_config(config)
            if config.max_tokens is None:
                config.max_tokens = self.api.max_tokens()

        # normalize input to chat
        if isinstance(input, str):
            input = [ChatMessageUser(content=input)]

        # handle tools and tool_choice
        tool_choice = tool_choice if tool_choice is not None else 'auto'
        tools_info = list(tools) if tools is not None else []

        if isinstance(tool_choice, ToolFunction):
            tools_info = [tool for tool in tools_info if tool.name == tool_choice.name]

        if tool_choice == 'none' or len(tools_info) == 0:
            if not self.api.tools_required():
                tools_info = []
            tool_choice = 'none'

        return input, tools_info, tool_choice, config


class ModelCache:
    _models: Dict[str, 'Model'] = {}

    @classmethod
    def get(cls, key: str) -> Optional['Model']:
        return cls._models.get(key, None)

    @classmethod
    def set(cls, key: str, model: 'Model') -> None:
        cls._models[key] = model


def get_model_with_task_config(task_config: 'TaskConfig') -> Model:
    """Get an instance of a model with the specified task configuration.

    Args:
        task_config (TaskConfig): Task configuration.

    Returns:
        Model: An instance of the model.
    """
    model = task_config.model
    eval_type = task_config.eval_type
    base_url = task_config.api_url
    api_key = task_config.api_key
    config = task_config.generation_config
    model_args = task_config.model_args or {}

    return get_model(
        model=model, eval_type=eval_type, base_url=base_url, api_key=api_key, config=config, model_args=model_args
    )


@thread_safe
def get_model(
    model: Union[str, Model, ModelAPI],
    eval_type: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    config: GenerateConfig = GenerateConfig(),
    model_args: dict = {},
    role: Optional[str] = None,
    memoize: bool = True,
) -> Model:
    """Get an instance of a model.

    Calls to get_model() are memoized (i.e. a call with the same arguments
    will return an existing instance of the model rather than creating a
    new one). You can disable this with `memoize=False`.

    Args:
        task_config (TaskConfig): Task configuration.
        memoize (bool): Whether to memoize the model instance.

    Returns:
        Model instance.

    """

    # start with seeing if a model was passed
    if isinstance(model, Model):
        return model

    if isinstance(model, ModelAPI):
        return Model(model, config, model_args)

    # see if we can return a memoized model instance
    # (exclude mockllm since custom_outputs is an infinite generator)
    model_cache_key: str = ''
    if eval_type.startswith('mock_llm'):
        memoize = False
    if memoize:
        model_cache_key = (
            model + str(role) + config.model_dump_json(exclude_none=True) + str(base_url) + str(api_key)
            + str(to_jsonable_python(model_args, fallback=lambda _: None))
        )
        cached = ModelCache.get(model_cache_key)
        if cached is not None:
            return cached

    logger.info(
        f'Creating model {model} with eval_type={eval_type} '
        f'base_url={base_url}, config={config.model_dump(exclude_none=True)}, model_args={model_args}'
    )

    # find a matching model type
    modelapi_type = get_model_api(eval_type)

    modelapi_instance = modelapi_type(
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        config=config,
        **model_args,
    )
    m = Model(modelapi_instance, config, model_args)
    if role is not None:
        m.role = role
    if memoize:
        ModelCache.set(model_cache_key, m)
    return m
