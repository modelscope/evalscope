import logging
import os
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Optional

from evalscope.backend.rag_eval.ragas.arguments import RAGASTestsetConfig

logger = logging.getLogger(__name__)


def _build_ragas_llm(llm_config) -> Any:
    """Build a ragas-native LLM for testset generation.

    Args:
        llm_config: RAGASLLMConfig instance.

    Returns:
        A BaseRagasLLM instance compatible with ragas 0.4.x TestsetGenerator.
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


def _build_ragas_embeddings(embedding_config) -> Any:
    """Build a ragas-native embeddings model for testset generation.

    Args:
        embedding_config: RAGASEmbeddingConfig instance.

    Returns:
        A BaseRagasEmbeddings instance compatible with ragas 0.4.x TestsetGenerator.
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


def load_documents(file_paths: List[str]):
    """Load documents from file paths using LangChain document loaders.

    Args:
        file_paths: List of file paths to load.

    Returns:
        List of LangChain Document objects.
    """
    from langchain_community.document_loaders import UnstructuredFileLoader

    all_docs = []
    for file_path in file_paths:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


def generate_testset(args: RAGASTestsetConfig) -> None:
    """Generate a test set using ragas 0.4.x TestsetGenerator API.

    This function loads documents, builds a knowledge graph, and generates
    evaluation samples using the ragas TestsetGenerator.

    Args:
        args: RAGASTestsetConfig instance with generation parameters.
    """
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator

    # Load documents
    documents = load_documents(args.docs)
    logger.info(f'Loaded {len(documents)} documents')

    # Build LLM and embeddings
    llm = _build_ragas_llm(args.generator_llm)
    embeddings = _build_ragas_embeddings(args.embeddings)

    # Create generator
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    # Configure run
    run_config = RunConfig(timeout=600, max_retries=10, max_wait=120, max_workers=1, log_tenacity=True)

    # Generate testset using generate_with_langchain_docs (handles transforms + KG internally)
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=args.test_size,
        run_config=run_config,
        with_debugging_logs=True,
        raise_exceptions=True,
    )

    # Save testset
    testset_df = testset.to_pandas()
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    testset_df.to_json(args.output_file, indent=4, index=False, orient='records', force_ascii=False)
    logger.info(f'Testset saved to {args.output_file}')

    # Generate answers for the testset
    testset_with_answer = _generate_answers(testset_df, args.generator_llm, args.language)
    answer_file = args.output_file.replace('.json', '_with_answer.json')
    testset_with_answer.to_json(answer_file, indent=4, index=False, orient='records', force_ascii=False)
    logger.info(f'Testset with answers saved to {answer_file}')


def _generate_answers(testset_df: pd.DataFrame, llm_config, language: str) -> pd.DataFrame:
    """Generate answers for the testset using the LLM.

    Args:
        testset_df: DataFrame with test samples.
        llm_config: RAGASLLMConfig for the generator LLM.
        language: Target language for answers.

    Returns:
        DataFrame with answers added.
    """
    from langchain_openai import ChatOpenAI

    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Answer in {language}.
Question: {question}
Context: {contexts}
Answer:
"""
    # Build a simple LangChain LLM for answer generation
    llm = ChatOpenAI(
        model=llm_config.model_name,
        base_url=llm_config.api_base,
        api_key=llm_config.api_key or 'EMPTY',
        temperature=llm_config.temperature,
    )

    items = []
    max_retries = 3
    for i in tqdm(range(len(testset_df)), desc='Generating Answers'):
        row = testset_df.iloc[i]
        question = row.get('user_input', '')
        contexts_raw = row.get('reference_contexts', [])
        contexts = '\n'.join(contexts_raw) if isinstance(contexts_raw, list) else str(contexts_raw)

        input_text = template.format(language=language, question=question, contexts=contexts)
        answer = ''
        for attempt in range(max_retries):
            try:
                response = llm.invoke(input_text)
                answer = response.content if hasattr(response, 'content') else str(response)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f'Failed to generate answer for row {i} (attempt {attempt + 1}): {e}')
                    import time
                    time.sleep(2**attempt)
                else:
                    logger.warning(f'Failed to generate answer for row {i} after {max_retries} attempts: {e}')

        items.append({
            'user_input': question,
            'retrieved_contexts': contexts_raw,
            'response': answer,
            'reference': row.get('reference', ''),
        })

    return pd.DataFrame.from_dict(items)
