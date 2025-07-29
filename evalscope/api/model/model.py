import abc
from typing import Any, Dict, List, Literal, Optional, Union

from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolInfo, ToolChoice
from .generate_config import GenerateConfig
from .model_output import ModelOutput

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
    async def generate(
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
