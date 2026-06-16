"""Models package for rag_eval - encoder and reranker implementations.

Provides a factory function `load_model` and direct access to model classes:
- SentenceTransformerEncoder: Local sentence-transformers encoder
- APIEncoder: OpenAI-compatible embedding API encoder
- CrossEncoderReranker: Local sentence-transformers cross-encoder
- APIReranker: OpenAI-compatible rerank API
"""
from evalscope.backend.rag_eval.models.encoder import APIEncoder, SentenceTransformerEncoder
from evalscope.backend.rag_eval.models.reranker import APIReranker, CrossEncoderReranker

__all__ = [
    'SentenceTransformerEncoder',
    'APIEncoder',
    'CrossEncoderReranker',
    'APIReranker',
    'load_model',
]


def load_model(config):
    """Factory function to load the appropriate model based on config.

    Args:
        config: ModelArguments dataclass or dict with model configuration.
            Key fields: model_name_or_path, is_cross_encoder, api_base, model_name, hub, etc.

    Returns:
        An encoder or reranker model instance.
    """
    if isinstance(config, dict):
        from evalscope.backend.rag_eval.mteb.arguments import MTEBModelConfig
        config = MTEBModelConfig(**config)

    # Extract common kwargs from config
    kwargs = {}
    if hasattr(config, 'to_dict'):
        kwargs = config.to_dict()
    elif hasattr(config, 'model_dump'):
        kwargs = config.model_dump()
    elif hasattr(config, 'dict'):
        kwargs = config.dict()
    else:
        kwargs = {k: v for k, v in vars(config).items() if not k.startswith('_')}

    # Route to API models if model_name is specified (indicating API usage)
    if getattr(config, 'model_name', None):
        if getattr(config, 'is_cross_encoder', False):
            return APIReranker(
                model_name=config.model_name,
                api_base=config.api_base,
                api_key=getattr(config, 'api_key', None),
                prompt=getattr(config, 'prompt', None),
                prompts=getattr(config, 'prompts', None),
                revision=getattr(config, 'revision', 'master'),
                batch_size=kwargs.get('encode_kwargs', {}).get('batch_size', 10),
                max_seq_length=getattr(config, 'max_seq_length', 512),
            )
        return APIEncoder(
            model_name=config.model_name,
            api_base=config.api_base,
            api_key=getattr(config, 'api_key', None),
            dimensions=getattr(config, 'dimensions', None),
            max_seq_length=getattr(config, 'max_seq_length', 512),
            prompt=getattr(config, 'prompt', None),
            prompts=getattr(config, 'prompts', None),
            revision=getattr(config, 'revision', 'master'),
            batch_size=kwargs.get('encode_kwargs', {}).get('batch_size', 10),
        )

    # Route to local models
    model_name_or_path = getattr(config, 'model_name_or_path', '')
    hub = getattr(config, 'hub', 'modelscope')
    revision = getattr(config, 'revision', 'master')

    if getattr(config, 'is_cross_encoder', False):
        return CrossEncoderReranker(
            model_name_or_path=model_name_or_path,
            max_seq_length=getattr(config, 'max_seq_length', 512),
            prompt=getattr(config, 'prompt', None),
            prompts=getattr(config, 'prompts', None),
            revision=revision,
            hub=hub,
            model_kwargs=getattr(config, 'model_kwargs', None),
            config_kwargs=getattr(config, 'config_kwargs', None),
            encode_kwargs=getattr(config, 'encode_kwargs', None),
        )

    return SentenceTransformerEncoder(
        model_name_or_path=model_name_or_path,
        pooling_mode=getattr(config, 'pooling_mode', None),
        max_seq_length=getattr(config, 'max_seq_length', 512),
        prompt=getattr(config, 'prompt', None),
        prompts=getattr(config, 'prompts', None),
        revision=revision,
        hub=hub,
        model_kwargs=getattr(config, 'model_kwargs', None),
        config_kwargs=getattr(config, 'config_kwargs', None),
        encode_kwargs=getattr(config, 'encode_kwargs', None),
    )
