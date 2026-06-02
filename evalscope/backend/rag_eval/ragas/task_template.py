import asyncio
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def import_metric(metric_name: str):
    """Import a metric by name. Supports both ragas built-in and custom metrics.

    Args:
        metric_name: Either a ragas built-in metric name (e.g., 'answer_relevancy')
                    or a full Python import path (e.g., 'mypackage.metrics.CustomMetric')

    Returns:
        The metric instance or class.
    """
    # Try as ragas built-in first
    try:
        import re
        from ragas import metrics as ragas_metrics

        # Build candidate names: exact, snake_case, lowercase
        candidates = [metric_name]
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', metric_name).lower()
        if snake != metric_name:
            candidates.append(snake)
        if metric_name.lower() not in candidates:
            candidates.append(metric_name.lower())

        for name in candidates:
            if hasattr(ragas_metrics, name):
                obj = getattr(ragas_metrics, name)
                if isinstance(obj, type):
                    return obj()
                return obj
    except ImportError:
        pass

    # Try as full import path
    if '.' in metric_name:
        module_path, cls_name = metric_name.rsplit('.', 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, cls_name)
        if isinstance(obj, type):
            return obj()
        return obj

    raise ImportError(f'Cannot import metric: {metric_name}')


def _build_llm(llm_config) -> Any:
    """Build LLM from config using ragas 0.4.x llm_factory pattern.

    Args:
        llm_config: RAGASLLMConfig instance.

    Returns:
        A BaseRagasLLM instance compatible with ragas 0.4.x.
    """
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory

        client_kwargs: Dict[str, Any] = {}
        if llm_config.api_base:
            client_kwargs['base_url'] = llm_config.api_base
        if llm_config.api_key:
            client_kwargs['api_key'] = llm_config.api_key

        client = OpenAI(**client_kwargs)
        return llm_factory(llm_config.model_name, client=client)
    except (ImportError, AttributeError, TypeError):
        logger.warning('Failed to use llm_factory, falling back to LangchainLLMWrapper', exc_info=True)
        from langchain_openai import ChatOpenAI
        from ragas.llms.base import LangchainLLMWrapper

        langchain_llm = ChatOpenAI(
            model=llm_config.model_name,
            base_url=llm_config.api_base,
            api_key=llm_config.api_key or 'EMPTY',
            temperature=llm_config.temperature,
        )
        return LangchainLLMWrapper(langchain_llm)


def _build_embeddings(embedding_config) -> Any:
    """Build embeddings from config using ragas 0.4.x embedding_factory pattern.

    Args:
        embedding_config: RAGASEmbeddingConfig instance.

    Returns:
        A BaseRagasEmbeddings or BaseRagasEmbedding instance.
    """
    if embedding_config.provider == 'openai':
        try:
            from openai import OpenAI
            from ragas.embeddings import embedding_factory

            client_kwargs: Dict[str, Any] = {}
            if embedding_config.api_base:
                client_kwargs['base_url'] = embedding_config.api_base
            if embedding_config.api_key:
                client_kwargs['api_key'] = embedding_config.api_key

            client = OpenAI(**client_kwargs)
            return embedding_factory('openai', model=embedding_config.model_name_or_path, client=client)
        except (ImportError, AttributeError, TypeError):
            logger.warning('Failed to use embedding_factory for openai', exc_info=True)
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings.base import LangchainEmbeddingsWrapper

            return LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model=embedding_config.model_name_or_path,
                    base_url=embedding_config.api_base,
                    api_key=embedding_config.api_key or 'EMPTY',
                )
            )
    else:
        # Local model — resolve via ModelScope first
        from evalscope.backend.rag_eval.models.utils import resolve_model_path
        local_path = resolve_model_path(embedding_config.model_name_or_path)
        try:
            from ragas.embeddings import embedding_factory
            return embedding_factory('huggingface', model=local_path)
        except (ImportError, AttributeError, TypeError):
            logger.warning('Failed to use embedding_factory for huggingface', exc_info=True)
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings.base import LangchainEmbeddingsWrapper
            return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=local_path))


def _load_dataset(testset_file: str):
    """Load evaluation dataset from JSON/JSONL file.

    Args:
        testset_file: Path to the dataset file (JSON or JSONL format).

    Returns:
        EvaluationDataset instance.
    """
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    file_path = Path(testset_file)

    if file_path.suffix == '.jsonl':
        return EvaluationDataset.from_jsonl(file_path)

    with open(testset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = []
        for item in data:
            raw_ctx = item.get('contexts', item.get('retrieved_contexts', []))
            if raw_ctx is None:
                retrieved_contexts = []
            elif isinstance(raw_ctx, list):
                retrieved_contexts = raw_ctx
            else:
                retrieved_contexts = [str(raw_ctx)]

            sample = SingleTurnSample(
                user_input=item.get('question', item.get('user_input', '')),
                response=item.get('answer', item.get('response', '')),
                retrieved_contexts=retrieved_contexts,
                reference=item.get('ground_truth', item.get('reference', '')),
            )
            samples.append(sample)
        return EvaluationDataset(samples=samples)

    # Single dict, wrap in a list
    return EvaluationDataset.from_list([data])


async def _run_ragas_eval(args) -> Any:
    """Async RAGAS evaluation using 0.4.x aevaluate() API.

    Args:
        args: RAGASEvalConfig instance.

    Returns:
        EvaluationResult from ragas.
    """
    from ragas import aevaluate
    from ragas.run_config import RunConfig

    # 1. Build LLM
    llm = _build_llm(args.critic_llm)

    # 2. Build Embeddings
    embeddings = _build_embeddings(args.embeddings)

    # 3. Load dataset
    dataset = _load_dataset(args.testset_file)

    # 4. Load metrics
    metrics = [import_metric(m) for m in args.metrics]

    # 5. Build run config
    run_config = RunConfig(timeout=600, max_retries=3, max_wait=60)

    # 6. Run evaluation
    eval_kwargs: Dict[str, Any] = {
        'dataset': dataset,
        'metrics': metrics,
        'llm': llm,
        'embeddings': embeddings,
        'run_config': run_config,
        'raise_exceptions': args.raise_exceptions,
    }
    if args.batch_size is not None:
        eval_kwargs['batch_size'] = args.batch_size

    result = await aevaluate(**eval_kwargs)
    return result


def rag_eval(args) -> dict:
    """Main entry point for RAGAS evaluation.

    Uses ragas 0.4.x aevaluate() API with asyncio.run() for synchronous invocation.

    Args:
        args: RAGASEvalConfig instance or dict.

    Returns:
        Dictionary of metric scores.
    """
    from evalscope.backend.rag_eval.ragas.arguments import RAGASEvalConfig

    if isinstance(args, dict):
        args = RAGASEvalConfig(**args)

    logger.info(f'Starting RAGAS evaluation with metrics: {args.metrics}')

    result = asyncio.run(_run_ragas_eval(args))

    # Save results
    output_path = Path(args.testset_file).with_name(Path(args.testset_file).stem + '_score.json')
    result_df = result.to_pandas()
    result_df.to_json(str(output_path), orient='records', force_ascii=False, indent=2)

    logger.info(f'RAGAS evaluation complete. Results saved to {output_path}')
    logger.info(f'Scores: {result}')

    return dict(result._repr_dict)
