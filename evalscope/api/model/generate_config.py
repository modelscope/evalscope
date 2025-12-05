# flake8: noqa: E501
from copy import deepcopy
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Literal, Optional, Union

from evalscope.utils.json_schema import JSONSchema


class ResponseSchema(BaseModel):
    """Schema for model response when using Structured Output."""

    name: str
    """The name of the response schema. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64."""

    json_schema: JSONSchema
    """The schema for the response format, described as a JSON Schema object."""

    description: Optional[str] = Field(default=None)
    """A description of what the response format is for, used by the model to determine how to respond in the format."""

    strict: Optional[bool] = Field(default=None)
    """Whether to enable strict schema adherence when generating the output. If set to true, the model will always follow the exact schema defined in the schema field.
    OpenAI and Mistral only."""


class GenerateConfig(BaseModel):
    """Model generation options."""
    model_config = {'extra': 'allow'}

    timeout: Optional[int] = Field(default=None)
    """Request timeout (in seconds)."""

    retries: Optional[int] = Field(default=5)
    """Number of retries for the request. Only supported by OpenAI compatible models."""

    retry_interval: Optional[int] = Field(default=10)
    """Retry interval between retries (in seconds). Only supported by OpenAI compatible models."""

    batch_size: Optional[int] = Field(default=None)
    """Maximum number of concurrent connections to Model API (default is model specific) or batch size for generation."""

    stream: Optional[bool] = Field(default=None)
    """Whether to stream the response (default is model specific)."""

    max_tokens: Optional[int] = Field(default=None)
    """The maximum number of tokens that can be generated in the completion (default is model specific)."""

    top_p: Optional[float] = Field(default=None)
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass."""

    temperature: Optional[float] = Field(default=None)
    """What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."""

    stop_seqs: Optional[List[str]] = Field(default=None)
    """Sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence."""

    best_of: Optional[int] = Field(default=None)
    """Generates best_of completions server-side and returns the 'best' (the one with the highest log probability per token). vLLM only."""

    frequency_penalty: Optional[float] = Field(default=None)
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. OpenAI, Google, Grok, Groq, vLLM, and SGLang only."""

    presence_penalty: Optional[float] = Field(default=None)
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. OpenAI, Google, Grok, Groq, vLLM, and SGLang only."""

    repetition_penalty: Optional[float] = Field(default=None)
    """Exponential penalty applied to existing tokens in the generated text. 1.0 means no penalty. OpenAI, HuggingFace, and vLLM only."""

    logit_bias: Optional[Dict[int, float]] = Field(default=None)
    """Map token Ids to an associated bias value from -100 to 100 (e.g. "42=10,43=-10"). OpenAI, Grok, Grok, and vLLM only."""

    seed: Optional[int] = Field(default=None)
    """Random seed. OpenAI, Google, Mistral, Groq, HuggingFace, and vLLM only."""

    do_sample: Optional[bool] = Field(default=None)
    """Whether to use sampling; use greedy decoding otherwise. Only transformers models support this parameter."""

    top_k: Optional[int] = Field(default=None)
    """Randomly sample the next word from the top_k most likely next words. Anthropic, Google, HuggingFace, vLLM, and SGLang only."""

    n: Optional[int] = Field(default=None)
    """How many chat completion choices to generate for each input message. OpenAI, Grok, Google, TogetherAI, vLLM, and SGLang only."""

    logprobs: Optional[bool] = Field(default=None)
    """Return log probabilities of the output tokens. OpenAI, Grok, TogetherAI, Huggingface, llama-cpp-python, vLLM, and SGLang only."""

    top_logprobs: Optional[int] = Field(default=None)
    """Number of most likely tokens (0-20) to return at each token position, each with an associated log probability. OpenAI, Grok, Huggingface, vLLM, and SGLang only."""

    parallel_tool_calls: Optional[bool] = Field(default=None)
    """Whether to enable parallel function calling during tool use (defaults to True). OpenAI and Groq only."""

    internal_tools: Optional[bool] = Field(default=None)
    """Whether to automatically map tools to model internal implementations (e.g. 'computer' for anthropic)."""

    max_tool_output: Optional[int] = Field(default=None)
    """Maximum tool output (in bytes). Defaults to 16 * 1024."""

    cache_prompt: Union[Literal['auto'], bool, None] = Field(default=None)
    """Whether to cache the prompt prefix. Defaults to "auto", which will enable caching for requests with tools. Anthropic only."""

    reasoning_effort: Optional[Literal['low', 'medium', 'high']] = Field(default=None)
    """Constrains effort on reasoning for reasoning models (defaults to `medium`). Open AI o1 models only."""

    reasoning_tokens: Optional[int] = Field(default=None)
    """Maximum number of tokens to use for reasoning. Anthropic Claude models only."""

    reasoning_summary: Optional[Literal['concise', 'detailed', 'auto']] = Field(default=None)
    """Provide summary of reasoning steps (defaults to no summary). Use 'auto' to access the most detailed summarizer available for the current model. OpenAI reasoning models only."""

    reasoning_history: Optional[Literal['none', 'all', 'last', 'auto']] = Field(default=None)
    """Include reasoning in chat message history sent to generate."""

    response_schema: Optional[ResponseSchema] = Field(default=None)
    """Request a response format as JSONSchema (output should still be validated). OpenAI, Google, and Mistral only."""

    extra_body: Optional[Dict[str, Any]] = Field(default=None)
    """Extra body to be sent with requests to OpenAI compatible servers. OpenAI, vLLM, and SGLang only."""

    extra_query: Optional[Dict[str, Any]] = Field(default=None)
    """Extra query parameters to be sent with requests to OpenAI compatible servers. OpenAI, vLLM, and SGLang only."""

    extra_headers: Optional[Dict[str, str]] = Field(default=None)
    """Extra headers to be sent with requests to OpenAI compatible servers. OpenAI, vLLM, and SGLang only."""

    height: Optional[int] = Field(default=None)
    """Image height for image generation model only"""

    width: Optional[int] = Field(default=None)
    """Image width for image generation model only"""

    num_inference_steps: Optional[int] = Field(default=None)
    """Number of inference steps for image generation model only"""

    guidance_scale: Optional[float] = Field(default=None)
    """Guidance scale for image generation model only"""

    # migrate reasoning_history as a bool
    @model_validator(mode='before')
    @classmethod
    def migrate_reasoning(cls, data: Any) -> Any:
        if isinstance(data, dict):
            reasoning_history = data.get('reasoning_history', None)
            if reasoning_history is True:
                data['reasoning_history'] = 'all'
            elif reasoning_history is False:
                data['reasoning_history'] = 'none'

        return data

    def merge(self, other: 'GenerateConfig') -> 'GenerateConfig':
        """Merge another model configuration into this one.

        Args:
           other (GenerateConfig):
              Configuration to merge.

        Returns:
           Merged configuration.
        """
        config_keys = [field for field in self.__class__.model_fields.keys()]
        config = deepcopy(self)
        for key in config_keys:
            value = getattr(other, key, None)
            if value is not None:
                setattr(config, key, value)
        return config
