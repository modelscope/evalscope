from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class CustomLLM(LLM):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    n: int
    """The number of characters from the last message of the prompt to be echoed."""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        for char in prompt[: self.n]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"