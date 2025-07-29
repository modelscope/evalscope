import abc
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic_core import to_jsonable_python

from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from evalscope.api.registry import get_model_api
from evalscope.api.tool import ToolChoice, ToolInfo, ToolFunction
from .generate_config import GenerateConfig
from .model_output import ModelOutput

from evalscope.utils import get_logger

logger = get_logger()

class ModelAPI(abc.ABC):
    """Model API provider."""

    def __init__(self,
                 model_name: str,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: GenerateConfig = GenerateConfig(),
                 **kwargs) -> None:
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

        # set any explicitly specified api key
        self.api_key = api_key

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

    Use `get_model()` to get an instance of a model. Model provides an
    async context manager for closing the connection to it after use.
    For example:

    ```python
    async with get_model("openai/gpt-4o") as model:
        response = await model.generate("Say hello")
    ```
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
        """Model name."""
        return self.api.model_name

    @property
    def role(self) -> Optional[str]:
        """Model role."""
        return self._role
    
    @role.setter
    def role(self, role: str) -> None:
        self._role = role

    def __str__(self) -> str:
        return f"Model(name={self.name}, role={self.role})"
    
    def generate(
        self,
        input: Union[str, List[ChatMessage]],
        tools: Sequence[ToolInfo] = [],
        tool_choice: Optional[ToolChoice] = None,
        config: GenerateConfig = GenerateConfig(),
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
        # base config for this model
        base_config = self.config

        # merge passed config
        config = base_config.merge(config)

        # provide max_tokens from the model api if required
        if config.max_tokens is None:
            config.max_tokens = self.api.max_tokens_for_config(config)
            if config.max_tokens is None:
                config.max_tokens = self.api.max_tokens()

        # normalize input to chat
        if isinstance(input, str):
            input = [ChatMessageUser(content=input)]

        # insert any system message provided in config
        if config.system_message:
            input = [ChatMessageSystem(content=config.system_message)] + input

        # generate
        output = self._generate(
            input=input,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
        )

        # return output
        return output
        
    def _generate(
        self,
        input: List[ChatMessage],
        tools: Sequence[ToolInfo],
        tool_choice: Optional[ToolChoice],
        config: GenerateConfig,
    ) -> ModelOutput:

        # default to 'auto' for tool_choice (same as underlying model apis)
        tool_choice = tool_choice if tool_choice is not None else "auto"

        # resolve all tools into tool_info
        tools_info = tools

        # if we have a specific tool selected then filter out the others
        if isinstance(tool_choice, ToolFunction):
            tools_info = [tool for tool in tools_info if tool.name == tool_choice.name]

        # if tool_choice is "none" or if there are no tools then fully purge
        # the tools (as some models (e.g. openai and mistral) get confused
        # if you pass them tool definitions along with tool_choice == "none"
        # (they both 'semi' use the tool by placing the arguments in JSON
        # in their output!). on the other hand, anthropic actually errors if
        # there are tools anywhere in the message stream and no tools defined.
        if tool_choice == "none" or len(tools_info) == 0:
            # allow model providers to implement a tools_required() method to
            # force tools to be passed (we need this for anthropic)
            if not self.api.tools_required():
                tools_info = []
            tool_choice = "none"

        model_output = self.api.generate(
                            input=input,
                            tools=tools_info,
                            tool_choice=tool_choice,
                            config=config,
                        )

        # return results
        return model_output
    
def get_model(
    model: Union[str, Model],
    *,
    role: Optional[str] = None,
    config: GenerateConfig = GenerateConfig(),
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    memoize: bool = True,
    **model_args: Any,
) -> Model:
    """Get an instance of a model.

    Calls to get_model() are memoized (i.e. a call with the same arguments
    will return an existing instance of the model rather than creating a
    new one). You can disable this with `memoize=False`.

    Args:
       model: Model specification.
          If `Model` is passed it is returned unmodified.
    
       role: Optional named role for model (e.g. for roles specified
          at the task or eval level). Provide a `default` as a fallback
          in the case where the `role` hasn't been externally specified.
       config: Configuration for model.
       base_url: Optional. Alternate base URL for model.
       api_key: Optional. API key for model.
       memoize: Use/store a cached version of the model based on
          the parameters to `get_model()`
       **model_args: Additional args to
          pass to model constructor.

    Returns:
        Model instance.

    """
    # start with seeing if a model was passed
    if isinstance(model, Model):
        return model

    # see if we can return a memoized model instance
    # (exclude mockllm since custom_outputs is an infinite generator)
    model_cache_key: str = ""
    if model.startswith("mockllm/"):
        memoize = False
    if memoize:
        model_cache_key = (
            model
            + str(role)
            + config.model_dump_json(exclude_none=True)
            + str(base_url)
            + str(api_key)
            + str(to_jsonable_python(model_args, fallback=lambda _: None))
        )
        cached = cached_model(model_cache_key)
        if cached is not None:
            return cached
    
    # split model into api name and model name if necessary
    api_name = None
    original_model = model
    parts = model.split("/")
    if len(parts) > 1:
        api_name = parts[0]
        model = "/".join(parts[1:])
    
    logger.info(
        f"Creating model {model} with api_name={api_name}, base_url={base_url}, api_key={api_key}, config={config}, model_args={model_args}"
    )
    
    # find a matching model type
    modelapi_type = get_model_api(api_name)

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
        _models[model_cache_key] = m
    return m


# cache for memoization of get_model
_models: Dict[str, Model] = {}


def cached_model(key: str) -> Optional[Model]:

    # read from the cache
    return _models.get(key, None)