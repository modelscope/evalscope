# Copyright (c) Alibaba, Inc. and its affiliates.
from pydantic import Field, field_validator
from typing import List, Literal, Optional

from evalscope.utils.argument_utils import BaseArgument


class RAGASLLMConfig(BaseArgument):
    """RAGAS LLM configuration - adapts to ragas 0.4.x llm_factory() pattern."""

    model_name: str
    provider: str = 'openai'
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class RAGASEmbeddingConfig(BaseArgument):
    """RAGAS Embedding configuration - adapts to ragas 0.4.x embedding_factory()."""

    model_name_or_path: str
    provider: str = 'huggingface'
    api_base: Optional[str] = None
    api_key: Optional[str] = None


class RAGASEvalConfig(BaseArgument):
    """RAGAS evaluation configuration - adapts to 0.4.x aevaluate() API."""

    testset_file: str
    critic_llm: RAGASLLMConfig
    embeddings: RAGASEmbeddingConfig
    metrics: List[str] = Field(default=['answer_relevancy', 'faithfulness'])
    language: str = 'english'
    batch_size: Optional[int] = None
    raise_exceptions: bool = False

    @field_validator('critic_llm', mode='before')
    @classmethod
    def parse_critic_llm(cls, v):
        if isinstance(v, dict):
            return RAGASLLMConfig(**v)
        return v

    @field_validator('embeddings', mode='before')
    @classmethod
    def parse_embeddings(cls, v):
        if isinstance(v, dict):
            return RAGASEmbeddingConfig(**v)
        return v


class RAGASTestsetConfig(BaseArgument):
    """RAGAS testset generation configuration."""

    docs: List[str]
    test_size: int = 10
    output_file: str = 'outputs/testset.json'
    generator_llm: RAGASLLMConfig
    embeddings: RAGASEmbeddingConfig
    language: str = 'english'

    @field_validator('generator_llm', mode='before')
    @classmethod
    def parse_generator_llm(cls, v):
        if isinstance(v, dict):
            return RAGASLLMConfig(**v)
        return v

    @field_validator('embeddings', mode='before')
    @classmethod
    def parse_embeddings(cls, v):
        if isinstance(v, dict):
            return RAGASEmbeddingConfig(**v)
        return v


class RAGASToolConfig(BaseArgument):
    """Complete configuration for tool='ragas' in eval_config."""

    tool: Literal['ragas'] = 'ragas'
    testset_generation: Optional[RAGASTestsetConfig] = None
    eval: Optional[RAGASEvalConfig] = None

    @field_validator('tool', mode='before')
    @classmethod
    def normalize_tool(cls, v):
        return v.lower() if isinstance(v, str) else v

    @field_validator('testset_generation', mode='before')
    @classmethod
    def parse_testset_generation(cls, v):
        if isinstance(v, dict):
            return RAGASTestsetConfig(**v)
        return v

    @field_validator('eval', mode='before')
    @classmethod
    def parse_eval(cls, v):
        if isinstance(v, dict):
            return RAGASEvalConfig(**v)
        return v
